# MORPH

Official implementation of:

**MORPH: Morphology-aware Frequency Prompting with Cross-branch Attention Fusion for Polyp Segmentation**

## Requirements

Python 3.10 or later. Install a compatible CUDA build of PyTorch on your GPU server,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Model

Download the official [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
(`sam_vit_b_01ec64.pth`) and place it in `checkpoints/`.
The ResNet-18 ImageNet weights are downloaded automatically by torchvision.

## Code

```text
models/morph.py             MORPH, backbones, prompt encoder and decoder
models/whfc.py              Wavelet high-frequency cues
models/meaa.py              Multi-source feature aggregation
models/cbaf.py              Cross-branch attention fusion
train.py / test.py          Training and evaluation
configs/ / scripts/        Settings and commands
splits/                    Fixed sample lists
utils/                     Data loading, clicks, loss and metrics
```

## Datasets

Training: Kvasir-SEG and CVC-ClinicDB.

Evaluation: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS and CVC-300.
BKAI-IGH NeoPolyp is used only for strict zero-shot evaluation, never for
training, hyperparameter tuning or checkpoint selection.

Download the datasets separately. Keep the filenames in `splits/pranet_*.csv`.
Organize your data root as follows:

```text
dataset_root/
├── TrainDataset/
│   ├── image/
│   └── masks/
└── TestDataset/
    ├── Kvasir/{images,masks}/
    ├── CVC-ClinicDB/{images,masks}/
    ├── CVC-ColonDB/{images,masks}/
    ├── ETIS-LaribPolypDB/{images,masks}/
    └── CVC-300/{images,masks}/
```

Masks are binarized with `mask > 0`; images and masks are resized to 1024 × 1024.

## Training

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/train.sh
```

Or train one seed directly:

```bash
python train.py --data_root /path/to/dataset --seed 1234
```

The script trains seeds 1234, 2345 and 3456. Checkpoints and logs are saved under
`runs/morph/<seed>/`. Resume using `--resume runs/morph/1234/checkpoint_last.pth`.
Shared settings are in `configs/base.yaml`; overrides use
`--set train.epochs=200` or `--set execution.num_workers=4`.

## Evaluation

Use the same final checkpoint for all three click protocols:

```bash
export DATA_ROOT=/path/to/dataset
export CHECKPOINT=runs/morph/1234/checkpoint_last.pth
bash scripts/test_0m.sh
bash scripts/test_1m.sh
bash scripts/test_2m.sh
```

0-m uses no clicks. 1-m samples one positive click inside the ground-truth mask.
2-m adds a correction in the largest false-negative or false-positive component
after the first prediction. An error-free prediction needs no second click.
The positive/negative click maps have shape 2 × 128 × 128, with radius 5 on
the 128 × 128 grid. The encoder projects them to 512 × 32 × 32.

Each run saves predictions, per-image metrics, dataset summaries and click records.
Repeat evaluation for all three training seeds. For BKAI, supply
`python test.py --data_root /path/to/dataset --checkpoint /path/to/checkpoint.pth --bkai_split /path/to/bkai.csv`.

## Reproducibility

Training seeds: **1234, 2345, 3456**. Prompt evaluation seed: **1234**.
Per-image click seeds depend on dataset name and image ID, so sample order does
not change the prompts. Evaluation uses the final epoch and threshold 0.5.

The shared config exposes the settings currently available in the implementation,
including one-click training and optional numerical-stability layers. These are
implementation defaults; this code-only release does not certify paper scores
or a completed three-seed experiment. Check the final experiment configuration
when reproducing a particular reported checkpoint.

## Acknowledgements

The frozen ViT backbone uses Meta's [Segment Anything](https://github.com/facebookresearch/segment-anything).
The CNN backbone uses torchvision. The fixed data organization follows the
[PraNet](https://github.com/DengPingFan/PraNet) benchmark convention.
