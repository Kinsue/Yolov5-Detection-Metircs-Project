from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from multiprocessing.pool import Pool, ThreadPool
from tqdm import tqdm
from PIL import Image, ImageOps, ExifTags
from itertools import repeat

import glob
import os
import sys
import contextlib
import cv2
import numpy as np
import torch
import math
import hashlib
import logging

from utils import (
    segments2boxes,
    letterbox,
    xyn2xy,
    xywhn2xyxy,
    xyxy2xywhn,
    img2label_paths,
    xyxy2xywh,
)

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes

NUM_THREADS = min(
    8, max(1, os.cpu_count() - 1)
)  # number of YOLOv5 multiprocessing threads
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
LOGGING_NAME = "YOLOv5"  # logger name
LOGGER = logging.getLogger(
    LOGGING_NAME
)  # define globally (used in train.py, val.py, detect.py, etc.)
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class LoadImagesAndLabels(Dataset):
    """An Custom Yolo like Dataset Function for data preprocess"""

    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        rect=False,
        stride=32,
        pad=0.0,
        single_cls=False,
    ) -> None:
        super().__init__()

        self.path = path
        self.img_size = img_size

        self.stride = stride
        self.pad = pad
        self.rect = rect

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent, 1) if x.startswith("./") else x
                            for x in t
                        ]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f"does not exist")
            self.im_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, "No images found"
        except Exception as e:
            raise Exception(f"Error loading data from {path}: {e}\n") from e

        # Check cache
        self.label_files, self.detection_files = img2label_paths(self.im_files)  # labels
        label_cache_path = (
            p if p.is_file() else Path(self.label_files[0]).parent
        ).with_suffix(".cache")

        try:
            label_cache = np.load(label_cache_path, allow_pickle=True).item()
            assert label_cache["version"] == self.cache_version # matches current version
            assert label_cache["hash"] == get_hash(
                self.label_files + self.im_files + self.detection_files
            )  # identical hash
        except Exception:
            label_cache = self.cache_labels(label_cache_path, "")

        nf, nm, ne, nc, n = label_cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total

        # Read cache
        [label_cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, detections, shapes, self.segments = zip(*label_cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert (
            nl > 0
        ), f"All labels empty in {label_cache_path}, can not start training. {HELP_URL}"

        self.labels = list(labels)
        self.detections = list(detections)
        self.shapes = np.array(shapes)
        self.im_files = list(label_cache.keys())  # update
        self.label_files, self.detection_files = img2label_paths(label_cache.keys())  # update

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        self.ims = [None] * n

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.detection_files = [self.detection_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.detections = [self.detections[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
            )

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):

        augment = False
        index = self.indices[index]
        img, (h0, w0), (h, w) = self.load_image(index)
        # Letterbox
        shape = (
            self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        )  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        detections = self.detections[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
            )
        if detections.size:  # normalized xywh to pixel xyxy format
            detections[:, 1:-1] = xywhn2xyxy(
                detections[:, 1:-1], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
            )

        nl = len(labels)  # number of labels
        nd = len(detections)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
            )
        if nd:
            detections[:, 1:-1] = xyxy2xywhn(
                detections[:, 1:-1], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
            )

        labels_out = torch.zeros((nl, 6))
        detections_out = torch.zeros((nd, 7))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        if nd:
            detections_out[:, 1:] = torch.from_numpy(detections)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, detections_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        im, label, detection, path, shapes = zip(*batch)  # transposed
        for i, (lb, dt) in enumerate(zip(label, detection)):
            lb[:, 0] = i  # add target image index for build_targets()
            dt[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), torch.cat(detection, 0), path, shapes

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f = (self.ims[i], self.im_files[i])
        if im is None:  # not cached in RAM
            im = cv2.imread(f)  # BGR
            assert im is not None, f"Image Not Found {f}"

            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                # Modify
                # interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
                im = cv2.resize(
                    im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp
                )
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(
                    verify_image_label,
                    zip(self.im_files, self.label_files, self.detection_files, repeat(prefix)),
                ),
                desc=desc,
                total=len(self.im_files),
                bar_format=TQDM_BAR_FORMAT,
            )
            for im_file, lb, dt, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, dt, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files + self.detection_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(
                f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}"
            )  # not writeable
        return x


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, dt_file, prefix = args
    nm, nf, ne, nc, msg, segments = (
        0,
        0,
        0,
        0,
        "",
        [],
    )  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg = (
                        f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
                    )

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                    ]  # (cls, xy1...)
                    lb = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert (
                    lb.shape[1] == 5
                ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (
                    lb[:, 1:] <= 1
                ).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        
        # verify detections
        if os.path.isfile(dt_file):
            ndf = 1
            with open(dt_file) as f:
                dt = [x.split() for x in f.read().strip().splitlines() if len(x)]
                dt = np.array(dt, dtype=np.float32)
            dnl = len(dt)
            if dnl:
                assert (
                    dt.shape[1] == 6
                ), f"detections require 5 columns, {dt.shape[1]} columns detected"
                assert (dt >= 0).all(), f"negative detection values {dt[dt < 0]}"
                assert (
                    dt[:, 1:] <= 1
                ).all(), f"non-normalized or out of bounds coordinates and conf {dt[:, 1:][dt[:, 1:] > 1]}"
        else:
            dnl = np.zeros((0, 6), dtype=np.float32)

        return im_file, lb, dt, shape, segments, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]

# def verify_detection_result(args):
#     # Verify one image-label pair
# 
#     dt_file = args
#     nm, nf, ne, nc, msg, segments = (
#         0,
#         0,
#         0,
#         0,
#         "",
#         [],
#     )  # number (missing, found, empty, corrupt), message, segments
# 
#     try:
#         # verify labels
#         if os.path.isfile(dt_file):
#             nf = 1  # label found
#             with open(dt_file) as f:
#                 dt = [x.split() for x in f.read().strip().splitlines() if len(x)]
#                 if any(len(x) > 6 for x in dt):  # is segment
#                     classes = np.array([x[0] for x in dt], dtype=np.float32)
#                     segments = [
#                         np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in dt
#                     ]  # (cls, xy1...)
#                     dt = np.concatenate(
#                         (classes.reshape(-1, 1), segments2boxes(segments)), 1
#                     )  # (cls, xywh)
#                 dt = np.array(dt, dtype=np.float32)
#             nl = len(dt)
#             if nl:
#                 assert (
#                     dt.shape[1] == 5
#                 ), f"labels require 5 columns, {dt.shape[1]} columns detected"
#                 assert (dt >= 0).all(), f"negative label values {dt[dt < 0]}"
#                 assert (
#                     dt[:, 1:] <= 1
#                 ).all(), f"non-normalized or out of bounds coordinates {dt[:, 1:][dt[:, 1:] > 1]}"
#                 _, i = np.unique(dt, axis=0, return_index=True)
#                 if len(i) < nl:  # duplicate row check
#                     dt = dt[i]  # remove duplicates
#                     if segments:
#                         segments = [segments[x] for x in i]
#                     msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
#             else:
#                 ne = 1  # label empty
#                 dt = np.zeros((0, 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             dt = np.zeros((0, 5), dtype=np.float32)
#         return im_file, dt, shape, segments, nm, nf, ne, nc, msg
#     except Exception as e:
#         nc = 1
#         msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
#         return [None, None, None, None, nm, nf, ne, nc, msg]

def create_dataloader(
    path, imgsz, batch_size=16, stride=32, single_cls=False, pad=0.0, rect=False, workers=8, seed=42
):
    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        rect=rect,  # rectangular batches
        stride=int(stride),
        pad=pad,
    )

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        collate_fn=dataset.collate_fn,
    ), dataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = create_dataloader(
        ROOT / "data" / "val.txt",
        imgsz=640,
        batch_size=32,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=8,
    )[0]

    for batch_i, (im, targets, detections, paths, shapes) in enumerate(dataloader):
        print(batch_i, targets, detections, sep='\n============================\n')
        print("=========")
    
