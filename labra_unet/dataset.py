"""Leaf images converted to CIELAB, with class-index masks (0 background, 1 BP, 2 LC)."""
import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, tf=None, oversample_lc=0):
        """oversample_lc: extra copies of every image whose mask contains LC (class 2)."""
        self.items = []
        for p in sorted(glob.glob(os.path.join(img_dir, "*"))):
            m = os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            self.items.append((p, m if os.path.exists(m) else None))
        if oversample_lc:
            lc = [it for it in self.items if it[1] is not None and 2 in cv2.imread(it[1], cv2.IMREAD_GRAYSCALE)]
            self.items += lc * oversample_lc
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2LAB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else np.zeros(img.shape[:2], np.uint8)
        if self.tf:
            out = self.tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask.long()
