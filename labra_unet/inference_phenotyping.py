#!/usr/bin/env python3
"""
Lesion phenotypes of leaf photos with a trained LabRA-UNet.

Each photo is converted to CIELAB, resized to the network input (256 x 256) and segmented;
the class probabilities are resized back to the photo's resolution and the mask is taken
from them. The lesion pixels (BP and LC classes) are analysed with connected components at
that resolution:
  N         lesion count: components of at least --min-area px;
  CL        combined lesion-component count after the density-based spatial merge: the
            lesion mask is closed with an elliptical structuring element of --merge-dist px
            and its components of at least --min-area px are counted;
  BP / LC   merged components of at most / more than --tau px.
Writes analysis_report.csv, overlays (BP red, LC blue) and the predicted class masks
(0 background, 1 BP, 2 LC) to runs/inference/expN/.

Usage: python inference_phenotyping.py --weights best_model.pth --source photos/
"""
import argparse, os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models import LabRAUNet

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def predict(model, img_bgr, imgsz, dev):
    """Class probabilities resized (bilinear) to the photo's resolution, and the class-index
    mask taken from them (0 background, 1 BP, 2 LC)."""
    h, w = img_bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB), (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5                  # same normalization as in training
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = F.interpolate(torch.softmax(model(x).float(), 1), size=(h, w), mode="bilinear", align_corners=False)[0]
    return prob.argmax(0).cpu().numpy().astype(np.uint8), prob.cpu().numpy()


def phenotype(mask, min_area=100, tau=500, merge_dist=30):
    """Lesion traits of one class-index mask. Returns the traits and a per-pixel map of the
    merged-component class (1 BP, 2 LC, 0 background or component below min_area)."""
    lesion = (mask > 0).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(lesion)
    n_lesions = int((st[1:, cv2.CC_STAT_AREA] >= min_area).sum())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_dist, merge_dist))
    closed = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(closed)
    area = st[:, cv2.CC_STAT_AREA]
    cls = np.zeros(n, np.uint8)
    cls[(area >= min_area) & (area <= tau)] = 1
    cls[area > tau] = 2
    cls[0] = 0                                                       # background label
    comp = cls[lbl]
    traits = {"N": n_lesions, "CL": int((cls > 0).sum()), "BP_count": int((cls == 1).sum()),
              "LC_count": int((cls == 2).sum()), "BP_area": int(((comp == 1) & (lesion == 1)).sum()),
              "LC_area": int(((comp == 2) & (lesion == 1)).sum()), "lesion_area": int(lesion.sum())}
    return traits, comp


def overlay(img, comp, lesion, traits):
    vis = img.copy()
    vis[(comp == 1) & lesion] = (0, 0, 255)                          # BP red
    vis[(comp == 2) & lesion] = (255, 0, 0)                          # LC blue
    vis = cv2.addWeighted(vis, 0.6, img, 0.4, 0)
    txt = f"N {traits['N']}  CL {traits['CL']}  BP {traits['BP_count']}  LC {traits['LC_count']}"
    cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="folder of photos")
    ap.add_argument("--imgsz", type=int, default=256, help="network input size used in training")
    ap.add_argument("--min-area", type=int, default=100, help="smallest component counted as a lesion (px)")
    ap.add_argument("--tau", type=int, default=500, help="merged components larger than this are LC (px)")
    ap.add_argument("--merge-dist", type=int, default=30, help="size of the elliptical structuring element (px)")
    ap.add_argument("--output", default="runs/inference")
    a = ap.parse_args()

    os.makedirs(a.output, exist_ok=True)
    n = 1 + max([int(d[3:]) for d in os.listdir(a.output) if d.startswith("exp") and d[3:].isdigit()] or [0])
    out = os.path.join(a.output, f"exp{n}")
    os.makedirs(os.path.join(out, "masks")); os.makedirs(os.path.join(out, "visualizations"))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LabRAUNet(n_cls=3).to(dev)
    model.load_state_dict(torch.load(a.weights, map_location=dev)); model.eval()

    rows = []
    for f in sorted(os.listdir(a.source)):
        if not f.lower().endswith(EXTS): continue
        img = cv2.imread(os.path.join(a.source, f))
        if img is None: continue
        mask, _ = predict(model, img, a.imgsz, dev)
        traits, comp = phenotype(mask, a.min_area, a.tau, a.merge_dist)
        name = os.path.splitext(f)[0]
        cv2.imwrite(os.path.join(out, "masks", name + ".png"), mask)
        cv2.imwrite(os.path.join(out, "visualizations", name + ".jpg"), overlay(img, comp, mask > 0, traits))
        rows.append({"Filename": name, **traits})

    pd.DataFrame(rows).to_csv(os.path.join(out, "analysis_report.csv"), index=False)
    print(f"{len(rows)} photos -> {out}  (min-area {a.min_area} px, tau {a.tau} px, merge-dist {a.merge_dist} px)")


if __name__ == "__main__":
    main()
