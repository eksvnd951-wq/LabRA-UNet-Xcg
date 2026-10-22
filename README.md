# LabRA-UNet-Xcg

Lesion segmentation code and trained weights for the manuscript

> **Deep Learning-Based Lesion Phenotyping via LabRA-UNet for Genetic Dissection of Soybean Resistance to *Xanthomonas citri* pv. *glycines***
> (submitted to *Plant Phenomics*)

The network segments soybean leaf photos into background, bacterial pustule (`BP`) and `SP` lesions. It then counts the lesions of each class and measures their area. The scripts in [`colab/`](colab) are the ones used on Google Colab to train the released model (2026-02-23) and to extract the lesion phenotypes.

## Files

| File | What it does |
|---|---|
| `colab/prepare_data.py` | Downloads the dataset from Roboflow (YOLOv8 polygon export) and converts the polygons into class-index masks in `<split>/masks/`: 0 = background, 1 = BP, 2 = SP. |
| `colab/dataset.py` | `LeafDataset`. Reads each image and converts BGR to CIELAB (`cv2.COLOR_BGR2LAB`). For training, images that contain SP are oversampled ×4. |
| `colab/models.py` | `AttentionUNet`: 4-level U-Net with double 3 × 3 conv blocks (64–1024 channels) and transposed-conv upsampling. The class defines attention blocks (`att1`–`att4`), but `forward()` does not call them. 31.4 M parameters. |
| `colab/losses.py` | `ComboLoss`: focal loss (α 0.25, γ 2) + Tversky loss (α 0.3, β 0.7). |
| `colab/metrics.py` | Per-class pixel IoU, F1, precision, recall and AP, used to monitor training. |
| `colab/train.py` | Trains at 512 × 512 with AdamW (lr 3e-4). Augmentation: flips, RandomRotate90, GridDistortion, ElasticTransform. Normalization: (x − 0.5)/0.5. Saves `runs/train/expN/best_model.pth` (best 0.6 mIoU + 0.4 mAP on the training batches), `final_metrics.csv` and TensorBoard logs. |
| `colab/inference_phenotyping.py` | Runs a checkpoint on a folder of photos. Writes the per-photo BP/SP lesion count and area (`analysis_report.csv`), overlays, and raw masks to `runs/inference/expN/`. Lesions are connected components after a 3 × 3 closing; components under 10 px are dropped. |
| `colab/analysis_utils.py` | Plotting helpers (training curves). |

## Data

Roboflow Universe, workspace `park-yuhyeon`, project `bacteria-dzvwf`, version 35 (CC BY 4.0). Images are resized to 640 × 640, and the labels are polygons with two classes, `BP` and `SP`. The images are not stored in this repository.

| Split | Files | Source photos |
|---|---|---|
| train | 2,304 | 192 (the train splits of versions 32–35 merged: 4 Roboflow exports × 3 augmented copies per photo) |
| valid | 10 | 10 |
| test | 5 | 5 |

No source photo appears in more than one split.

## Setup

```bash
pip install -r requirements.txt
```

Training needs a CUDA GPU. Inference also runs on a CPU.

## Usage

Run the scripts from inside `colab/`, because they import each other.

```bash
cd colab

# 1. download the dataset and write the masks (needs your own Roboflow API key)
python prepare_data.py --api_key YOUR_ROBOFLOW_API_KEY --workspace park-yuhyeon --project bacteria-dzvwf --version 35

# 2. train (the released model: 200 epochs)
python train.py --data Bacteria-35 --epochs 200

# 3. lesion counts and areas for a folder of photos
python inference_phenotyping.py --weights final_v35_exp2_best_model.pth --source path/to/photos
```

The phenotypes in the manuscript were extracted with step 3 as shown, without `--auto-merge`. The optional `--auto-merge` (`--merge-dist` 30 px, `--min-cluster` 500 px) closes the BP mask with an elliptical kernel and relabels clusters larger than the threshold as SP.

## Weights

The released checkpoint is attached to the GitHub release [`weights-colab`](../../releases/tag/weights-colab):

| File | Source | SHA-256 |
|---|---|---|
| `final_v35_exp2_best_model.pth` | Colab run of 2026-02-23, `runs/train/exp2/best_model.pth` (200 epochs, 20.7 h) | `a77854e3686831a08d4f67e8a1307fa13f211feda45ff0e7f8f5169f29b96bd6` |

```python
import torch
from models import AttentionUNet

model = AttentionUNet(n_cls=3)
model.load_state_dict(torch.load("final_v35_exp2_best_model.pth", map_location="cpu"))
model.eval()
# input: BGR photo -> cv2.COLOR_BGR2LAB -> 512 x 512 -> (x / 255 - 0.5) / 0.5, channels first
```

## Citation

Please cite the manuscript above. The full reference will be added on publication.
