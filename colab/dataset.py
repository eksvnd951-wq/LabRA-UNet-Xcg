
import cv2, glob, os, torch, numpy as np
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, imgsz=512, tf=None, is_train=False):
        self.data_list = []
        imgs = sorted(glob.glob(f"{img_dir}/*"))
        target_indices = []

        for i, p in enumerate(imgs):
            base = os.path.splitext(os.path.basename(p))[0]
            m_path = os.path.join(mask_dir, base+'.png')
            if not os.path.exists(m_path): self.data_list.append((p,None)); continue
            self.data_list.append((p, m_path))

            # Oversampling for SP(Large) - Class 2
            if is_train:
                m = cv2.imread(m_path, 0)
                if m is not None and 2 in m: target_indices.append(i)

        if is_train and target_indices:
            for i in target_indices:
                for _ in range(4): self.data_list.append(self.data_list[i])

        self.tf = tf

    def __len__(self): return len(self.data_list)
    def __getitem__(self, i):
        p, mp = self.data_list[i]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2LAB)
        mask = cv2.imread(mp, 0) if mp else np.zeros(img.shape[:2], np.uint8)
        if self.tf:
            aug = self.tf(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask.long()
