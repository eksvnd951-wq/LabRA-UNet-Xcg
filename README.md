# LabRA-UNet-Xcg

LabRA-UNet code and trained weights for the manuscript

> **Deep Learning-Based Lesion Phenotyping via LabRA-UNet for Genetic Dissection of Soybean Resistance to *Xanthomonas citri* pv. *glycines***
> (submitted to *Plant Phenomics*)

LabRA-UNet segments soybean leaf photos into background, discrete bacterial pustule (`BP`) and lesion-coalescence (`LC`) pixels. The lesion masks are then turned into lesion traits by connected-component analysis and a density-based spatial merge.

## Model

- **Input:** the RGB photo is converted to CIELAB (L\*, a\*, b\*), and the three channels are stacked as the network input (256 × 256).
- **Network:** a 4-level U-Net encoder–decoder (64–1024 channels, 29.6 M parameters), laid out as in Fig. S1 of the manuscript.
  - **Residual blocks:** every encoder stage, the bottleneck and every decoder stage is a residual block (two 3 × 3 conv–BN layers plus an identity or 1 × 1 conv shortcut).
  - **Attention gates:** the three upper levels (64, 128 and 256 channels) have skip connections, and each passes through an attention gate. The deepest level (512 channels) has no skip connection. In each gate, the encoder features x and the decoder gating signal g are projected with 1 × 1 convolutions and added. ReLU, a 1 × 1 convolution and a sigmoid then give coefficients α ∈ [0, 1], which multiply x. The refined features are concatenated with the decoder features.
- **Loss:** focal loss (α 0.25, γ 2) + Tversky loss (α 0.3, β 0.7).

## Files

| File | What it does |
|---|---|
| `labra_unet/models.py` | `LabRAUNet`, `ResidualBlock`, `AttentionGate` |
| `labra_unet/losses.py` | Focal–Tversky loss |
| `labra_unet/dataset.py` | Reads images, converts them to CIELAB, and pairs them with class-index masks (0 background, 1 BP, 2 LC) |
| `labra_unet/prepare_data.py` | Downloads a Roboflow dataset version (YOLOv8 polygons) and writes the class-index masks |
| `labra_unet/make_split.py` | Splits the images into train / valid / test (7:2:1) by source photo |
| `labra_unet/train.py` | Training with validation-based checkpoint selection and early stopping |
| `labra_unet/inference_phenotyping.py` | Segments a folder of photos and writes the lesion traits per photo |
| `data/split.csv` | Split of every image file used to train the released model |

## Data

The photos are from Roboflow Universe, workspace `park-yuhyeon`, project `bacteria-dzvwf` (CC BY 4.0, 640 × 640). The images are not stored in this repository.

The training data combine the train splits of versions 32, 33, 34 and 35 in the folder of version 35. Files whose names already exist are renamed `Bacteria-<version>_<name>`. The merged folder holds 207 source photos:

- **192 photos from the train splits:** 12 files each, namely the un-augmented image (identical in all four versions) and 8 augmented copies.
- **15 photos from the valid and test splits of version 35:** one file each.

The 4 test photos without annotations are dropped. `make_split.py` (seed 0) then splits the remaining 203 photos 7:2:1 by photo:

| Split | Photos | Files |
|---|---|---|
| train | 142 | 1,627 (all files of each photo) |
| valid | 41 | 41 (un-augmented image only) |
| test | 20 | 20 (un-augmented image only) |

`data/split.csv` lists every file.

## Setup

```bash
pip install -r requirements.txt
```

Training needs a CUDA GPU. Inference also runs on a CPU. Run the scripts from inside `labra_unet/`.

## Training

```bash
python make_split.py --src Bacteria-35 --out Bacteria-35-split
python train.py --data Bacteria-35-split
```

`train.py` defaults:

- **Input:** 256 × 256.
- **Optimizer:** AdamW (learning rate 0.001, weight decay 0.01) with cosine annealing over 200 epochs; batch size 16; mixed precision (bfloat16 where the GPU supports it, otherwise float16).
- **Augmentation:** horizontal/vertical flips, 90° rotations, grid distortion and elastic deformation. Training images that contain LC are oversampled ×4.
- **Checkpoint selection and stopping:** at most 200 epochs. After every epoch the model is evaluated on the validation set, and the checkpoint with the highest validation Dice (mean of BP and LC) is kept. Training stops after 30 epochs without improvement.

The output goes to `runs/train/expN/`: `best_model.pth`, `log.csv`, `training_curves.png` and `train_info.json`.

## Lesion traits

```bash
python inference_phenotyping.py --weights labra_unet_best.pth --source path/to/photos
```

The class probabilities are resized (bilinear) back to the photo's resolution, and the mask is taken from them. Connected components are then taken from the lesion pixels (BP and LC):

| Column | Definition |
|---|---|
| `N` | Lesion count: components of at least `--min-area` px (default 100 px) |
| `CL` | Combined lesion-component count after the density-based spatial merge. The BP pixels are closed with an elliptical structuring element of `--merge-dist` px (default 30 px), the LC pixels are added and holes are filled. Components of at least `--min-area` px are counted. |
| `BP_count`, `LC_count` | Merged components of at most / more than `--tau` px (default 500 px) |
| `BP_area`, `LC_area`, `lesion_area` | Lesion pixels in BP components, in LC components, and in total |

Near-view photos give the traits N and CLN of the manuscript (`N`, `CL`). Wide-view photos were processed by progressive learning (manuscript Figs. 3 and S2): the target leaf was first localized and cropped, and the leaf crops were then segmented in the same way. `N` and `CL` of the crops give W and CLW. The leaf-cropping stage is not part of this repository.

The output goes to `runs/inference/expN/`: `analysis_report.csv`, overlays (BP red, LC blue) and the predicted class masks.

## Weights

This version is still being trained. Its weights will be attached to the GitHub release `labra-unet-v2`.

The release [`labra-unet-v1`](../../releases/tag/labra-unet-v1) holds an earlier version with an attention gate at all four levels. It matches the code at tag `labra-unet-v1`, not the current code.

```python
import torch
from models import LabRAUNet

model = LabRAUNet(n_cls=3)
model.load_state_dict(torch.load("labra_unet_best.pth", map_location="cpu"))
model.eval()
# input: BGR photo -> cv2.COLOR_BGR2LAB -> 256 x 256 -> (x / 255 - 0.5) / 0.5, channels first
```

## Citation

Please cite the manuscript above. The full reference will be added on publication.
