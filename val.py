import sys
import argparse
import numpy as np
import torch
import os

from dataloader import create_dataloader
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import box_iou, ap_per_class, scale_boxes, TQDM_BAR_FORMAT

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 Detection Metircs Project Root directory

batch_idx = 32
device = "cpu"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """

    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)

    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def error(msg):
    print("\033[1;31m\t" + msg + "\033[0m")
    sys.exit(0)


def main(
    data,
    name_file,
    img_size=640,
    batch_size=32,
    stride=32,
    single_cls=False,
    pad=0.5,
    rect=True,
    device="cpu",
    workers=8,
):
    assert os.path.isfile(name_file), "{} not found".format(name_file)
    with open(name_file) as f:
        names = f.read().strip().splitlines()
        assert len(names) > 0, "No lables names in file {}".format(name_file)
        if isinstance(names, list):
            names = dict(enumerate(names)) 

    dataloader = create_dataloader(
        data,
        img_size,
        batch_size,
        stride,
        single_cls,
        pad,
        rect,
        workers,
    )[0]


    # Read original data from txt files
    jdict, stats, ap, ap_class = [], [], [], []
    dt, gt = [], []
    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    s = ("%20s" + "%11s" * 9) % (
        "Class",
        "Images",
        "Labels",
        "P",
        "R",
        "mAP@.5",
        "mAP@.85",
        "mAP@0.9",
        "mAP@0.95",
        "mAP@.5:.95",
    )
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

    # Config
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc = len(names)
    seen = 0

    for batch_i, (im, targets, detections, paths, shapes) in enumerate(pbar):
        im = im.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        bn = int(targets[-1][0])

        targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels
        for i, dt in enumerate(detections):
            detections[i][:, :4] = dt[:, :4] * torch.Tensor([width, height, width, height])
        # detections[:, 0:4] *= torch.Tensor([width, height, width, height])  # to pixels

        for si, pred in enumerate(detections):
            labels = targets[targets[:, 0] == si][:, 1:]
            path, shape = Path(paths[si]), shapes[si][0]
            nl, npr = labels.shape[0], pred.shape[0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            predn = pred.clone()
            predn[:, :4] = xywh2xyxy(predn[:, :4])
            scale_boxes(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space preds

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)

            stats.append(
                (correct, pred[:, 4], pred[:, 5], labels[:, 0])
            )  # (correct, conf, pcls, tcls)

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap55to80 = ap[:, 1:7]
        ap85, ap90, ap95 = ap[:, -3], ap[:, -2], ap[:, -1]
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        map85, map90, map95 = ap85.mean(), ap90.mean(), ap95.mean()
        map55to80 = ap55to80.mean(0)
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    pf = "%20s" + "%11i" * 2 + "%11.3g" * 7  # print format

    # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map85, map90, map95, map))
    if nt.sum() == 0:
        print(
            f"WARNING: no labels found in dataset, can not compute metrics without labels ⚠️"
        )

    # Print results per class
    for si, c in enumerate(ap_class):
        # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        print(
            pf
            % (
                names[c],
                seen,
                nt[c],
                p[si],
                r[si],
                ap50[si],
                ap85[si],
                ap90[si],
                ap95[si],
                ap[si],
            )
        )

    smAP = ("%20s" + "%11s" * 8) % (
        "Class",
        "Images",
        "Labels",
        "mAP@.55",
        "mAP@.60",
        "mAP@.65",
        "mAP@.70",
        "mAP@.75",
        "mAP@.80",
    )
    title_split = ("%40s" + "%28s" + "%40s") % (30 * "=", "mAP@.55 - mAP@.8", 30 * "=")
    print("\n", title_split, "\n\n", smAP)
    apf = "%20s" + "%11i" * 2 + "%11.3g" * 6

    print(
        apf
        % (
            "all",
            seen,
            nt.sum(),
            map55to80[0],
            map55to80[1],
            map55to80[2],
            map55to80[3],
            map55to80[4],
            map55to80[5],
        )
    )

    for si, c in enumerate(ap_class):
        print(
            apf
            % (
                names[c],
                seen,
                nt[c],
                ap55to80[si][0],
                ap55to80[si][1],
                ap55to80[si][2],
                ap55to80[si][3],
                ap55to80[si][4],
                ap55to80[si][5],
            )
        )


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/val.txt",
        help="val.txt path",
    )
    parser.add_argument(
        "--name_file",
        type=str,
        default=ROOT / "data/names.txt",
        help="names.txt path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="inference size (pixels)",
    )

    parser.add_argument("--batch_size", type=int, default=32, help="size of each batch")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="number of dataloader workers"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse()
    print(vars(opt))
    main(**vars(opt))
