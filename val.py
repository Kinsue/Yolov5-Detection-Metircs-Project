import sys
import argparse
import numpy as np
import torch

from utils import box_iou, ap_per_class
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

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
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def error(msg):
    print("\033[1;31m\t" + msg + "\033[0m")
    sys.exit(0)


def main(opt):
    image_path = Path(opt.images).resolve()
    det_path = Path(opt.detection_path).resolve()
    gt_path = Path(opt.groundtruth_path).resolve()
    lable_path = Path(opt.labels).resolve()

    names = {}
    with lable_path.open(mode="r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            names[idx] = line

    print(image_path, det_path, gt_path)

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
    pbar = tqdm(
        sorted(det_path.glob("*.txt")),
        desc=s,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(list(det_path.glob("*.txt"))),
    )
    for i, file in enumerate(pbar):
        # [N, 6] : class x y w h conf
        dt.append(np.insert(np.loadtxt(file, ndmin=2), 0, i, axis=1))
        file_name = file.parts[-1].split(".txt", 1)[0]

        if not dt:
            error("* There is no detection result in directory " + str(det_path))

        if gt_path.joinpath(file_name + ".txt").exists():
            gt.append(
                np.insert(
                    np.loadtxt(gt_path / (file_name + ".txt"), ndmin=2), 0, i, axis=1
                )
            )
        else:
            error(
                "* Do not find the groundtruth file "
                + str(gt_path / (file_name + ".txt"))
            )

        if image_path.joinpath(file_name + ".jpg").exists():
            im = np.array(plt.imread(image_path.joinpath(file_name + ".jpg")))
        else:
            error(
                "* Do not find the image file " + str(image_path / (file_name + ".jpg"))
            )

        gt[-1][:, 2:] *= np.array(im.shape[:2] * 2)
        dt[-1][:, 2:-1] *= np.array(im.shape[:2] * 2)

    batch_gt = np.vstack(gt)
    batch_det = np.vstack(dt)

    # Config
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc = len(names)
    seen = 0

    # Metrics
    for si in range(len(list(det_path.glob("*.txt")))):
        # for si in range(batch_idx):
        labels = batch_gt[batch_gt[:, 0] == si, 1:]
        pred = batch_det[batch_det[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        seen += 1

        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))
            continue

        # Evaluate
        # 将 target 的 xywh 转换成绝对坐标形式
        tbox_gt = xywh2xyxy(labels[:, 1:])
        tbox_det = xywh2xyxy(pred[:, 1:-1])

        # lables and detections in a single image with xyxy format
        labelsn = np.concatenate((labels[:, 0:1], tbox_gt), 1)
        predn = np.concatenate((tbox_det, pred[:, -1:], pred[:, 0:1]), axis=1)

        correct = process_batch(
            torch.tensor(predn, device=device),
            torch.tensor(labelsn, device=device),
            iouv,
        )
        stats.append(
            (
                correct,
                torch.tensor(predn[:, -2], device=device),
                torch.tensor(predn[:, -1], device=device),
                torch.tensor(labels[:, 0], device=device),
            )
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
    for i, c in enumerate(ap_class):
        # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        print(
            pf
            % (
                names[c],
                seen,
                nt[c],
                p[i],
                r[i],
                ap50[i],
                ap85[i],
                ap90[i],
                ap95[i],
                ap[i],
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

    for i, c in enumerate(ap_class):
        print(
            apf
            % (
                names[c],
                seen,
                nt[c],
                ap55to80[i][0],
                ap55to80[i][1],
                ap55to80[i][2],
                ap55to80[i][3],
                ap55to80[i][4],
                ap55to80[i][5],
            )
        )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-det",
        "--detection-path",
        default=ROOT / "input/DR",
        help="Detection result path.",
    )
    parser.add_argument(
        "-gt", "--groundtruth-path", default=ROOT / "input/GT", help="GroundTruth path."
    )
    parser.add_argument(
        "-img", "--images", default=ROOT / "input/image", help="images path "
    )
    parser.add_argument(
        "-l", "--labels", default=ROOT / "input/label.txt", help="label path"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse()
    main(opt)
