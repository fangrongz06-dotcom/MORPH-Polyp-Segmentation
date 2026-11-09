# MORPH

Implementation of **MORPH: Morphology-aware Frequency Prompting with
Cross-branch Attention Fusion for Polyp Segmentation**.

## Requirements

Python 3.10–3.12, PyTorch 2.5.1 and pytorch-wavelets 1.3.0.
The paper uses CUDA 12.1. Install the appropriate [PyTorch build](https://pytorch.org/get-started/previous-versions/), then:

```console
pip install -r requirements.txt
```

Download the official [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
and place `sam_vit_b_01ec64.pth` in `checkpoints/`.

## Datasets

Training uses Kvasir-SEG and CVC-ClinicDB. Evaluation uses Kvasir-SEG,
CVC-ClinicDB, CVC-ColonDB, ETIS and CVC-300, following the fixed
1,450-image training / 798-record test protocol.

```text
dataset_root/
  TrainDataset/{image,masks}/
  TestDataset/{Kvasir,CVC-ClinicDB,CVC-ColonDB,ETIS-LaribPolypDB,CVC-300}/{images,masks}/
```

Prepare local manifests from the original PraNet-style partition:

```console
python scripts/prepare_data.py --data_root /path/to/dataset
```

BKAI-IGH is reserved for zero-shot evaluation and requires prepared binary GT.
Datasets, per-image lists, weights and experimental results are not included.

## Training

```console
python train.py --data_root /path/to/dataset --seed 1234
```

Repeat with seeds 2345 and 3456. Settings are in `configs/base.yaml`.
Checkpoints are saved under `runs/morph/final/<seed>/`.
Use `--phase development` for the 1,305/145 development split.

## Evaluation

```console
python test.py --data_root /path/to/dataset --checkpoint /path/to/checkpoint.pth --protocol 0-m
```

Select `0-m`, `1-m` or `2-m` for zero, one or two clicks using the same
final-epoch checkpoint. The prompt evaluation seed is 1234.

## Code

The model is in `models/morph.py`; WHFC, MEAA and CBAF are in their
corresponding files under `models/`. Data processing, prompts and losses are
in `utils/`. Preparation, content auditing, statistics and profiling utilities
are in `scripts/`. Run each Python entry point with `--help` for options.
