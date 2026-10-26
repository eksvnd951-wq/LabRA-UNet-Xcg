#!/usr/bin/env python3
"""
Download one version of the Roboflow dataset (YOLOv8 polygon export) and rasterize the
polygons into class-index masks: <split>/masks/<image name>.png with 0 background, 1 BP, 2 LC.

Usage: python prepare_data.py --api_key YOUR_ROBOFLOW_API_KEY --workspace park-yuhyeon \
           --project bacteria-dzvwf --version 35
"""
import argparse, glob, os

import cv2
import numpy as np
import roboflow


def write_masks(root):
    for split in ("train", "valid", "test"):
        mdir = os.path.join(root, split, "masks"); os.makedirs(mdir, exist_ok=True)
        for img in glob.glob(os.path.join(root, split, "images", "*")):
            mask = np.zeros(cv2.imread(img).shape[:2], np.uint8)
            name = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(root, split, "labels", name + ".txt")
            if os.path.exists(lbl):
                for line in open(lbl):
                    parts = list(map(float, line.split()))
                    if not parts: continue
                    cls = min(int(parts[0]) + 1, 2)                       # 0 BP -> 1, 1 LC (SP) -> 2
                    pts = np.array(parts[1:]).reshape(-1, 2) * [mask.shape[1], mask.shape[0]]
                    cv2.fillPoly(mask, [pts.astype(int)], cls)
            cv2.imwrite(os.path.join(mdir, name + ".png"), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True); ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True); ap.add_argument("--version", type=int, required=True)
    a = ap.parse_args()
    ds = roboflow.Roboflow(api_key=a.api_key).workspace(a.workspace).project(a.project).version(a.version).download("yolov8")
    write_masks(ds.location)
    print(f"masks written to {ds.location}/<split>/masks")


if __name__ == "__main__":
    main()
