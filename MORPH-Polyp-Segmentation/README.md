# MORPH

Implementation of **MORPH: Morphology-aware Frequency Prompting with
Cross-branch Attention Fusion for Polyp Segmentation**.

## Installation

Use Python 3.10–3.12. The paper environment is PyTorch 2.5.1 with CUDA 12.1.
Install the corresponding build using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For CPU checks, use the CPU PyTorch index instead. Download the official
[SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
and place `sam_vit_b_01ec64.pth` in `checkpoints/`, or pass `--sam_checkpoint`.
Torchvision obtains ResNet-18 `IMAGENET1K_V1` weights through its usual cache.
Neither pretrained nor trained weights are included here.

## Code

```text
models/morph.py       frozen encoders, MORPH, dense prompt projection, decoder
models/whfc.py        fixed Haar DWT and trainable projections
models/meaa.py        multi-source edge-aware adapters
models/cbaf.py        cross-branch attention fusion
train.py / test.py    training and measured per-image evaluation
configs/             shared settings and 0/1/2-click evaluation protocols
scripts/             local preparation, auditing, statistics and profiling
utils/               preprocessing, prompts, supervision and runtime
tests/               generated-fixture CPU checks
```

## Data preparation

Obtain the original [PraNet-style partition](https://github.com/DengPingFan/PraNet)
separately. Preserve its existing training/test membership:

```text
dataset_root/
  TrainDataset/{image,masks}/
  TestDataset/
    Kvasir/{images,masks}/
    CVC-ClinicDB/{images,masks}/
    CVC-ColonDB/{images,masks}/
    ETIS-LaribPolypDB/{images,masks}/
    CVC-300/{images,masks}/
```

The nested `TrainDataset/TrainDataset/` download layout is also accepted.
Create private manifests and an RGB/GT content audit:

```bash
python scripts/prepare_data.py --data_root /path/to/dataset
```

The script requires 900 Kvasir-SEG and 550 ClinicDB training records, and
100/62/380/196/60 test records, respectively. Original numeric ClinicDB IDs
and Kvasir CUID filenames identify the source dataset. For renamed files,
provide `--source_train_manifest /path/to/train.csv` with `dataset,filename`
columns. This command validates an already supplied partition; it does not
randomly repartition the source and formal test sets.

Development IDs are ranked separately within each source dataset by the
SHA-256 of UTF-8 `1234|dataset|image_id`. The first 90 Kvasir and 55 ClinicDB
records form the 145-image holdout. Its complement contains 1,305 records.
Final training uses all original 1,450 records. Formal test membership remains
unchanged. Generated IDs, content hashes and audit reports stay in the ignored
`splits/local/` directory; no actual per-image lists are distributed.

## Model and preprocessing

Images are resized with their aspect ratio preserved and longest side 1024.
WHFC receives RGB floats in [0,1], padded on the right/bottom with
`(0.485, 0.456, 0.406)`. The encoders normalize the valid RGB region using
their ImageNet/SAM statistics and explicitly set normalized padding to zero.
Binary GT uses nearest-neighbor resizing and zero background padding.
Flips, rotation and brightness/contrast follow the shared configuration;
brightness and contrast use independent probability draws.

WHFC uses `pytorch_wavelets.DWTForward(J=3, wave="haar", mode="symmetric")`.
It independently decomposes RGB, retains signed coefficients, resizes each
level to 64×64, and concatenates scales finest-to-coarsest. Each level uses
direction-major `LH_RGB, HL_RGB, HH_RGB` ordering. Four trainable 1×1
convolutions map the 27 channels to 64/128/256/512. The DWT has no trainable
parameters.

MEAA prompts enter frozen SAM blocks 2/5/8/11. CBAF combines the final ViT
feature and deepest CNN feature at 32×32×512. The separate two-channel
128×128 click map is encoded by two stride-2 2×2 Conv–BN–GELU stages and a
1×1 projection to 32×32×512. The five-stage decoder upsamples to 1024×1024.
The original SAM prompt encoder and mask decoder are not used.

## Training

```bash
python train.py --data_root /path/to/dataset --phase development --seed 1234
python test.py --data_root /path/to/dataset --scope development --protocol 0-m \
  --checkpoint runs/morph/development/1234/checkpoint_last.pth
```

Use the development holdout for configuration/loss selection. After fixing
settings, retrain independently on the complete source pool:

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/train.sh
```

The script runs seeds 1234, 2345 and 3456. Defaults are Adam, batch size 2,
200 epochs, learning rate 0.01, weight decay 1e-4, StepLR(50, 0.1), and
`0.3 * BCE + 1.2 * soft_Dice` with Dice smoothing 1e-6. Training supplies
one positive click per nonempty augmented GT. All three evaluation protocols
use the same final-epoch model. There is no early stopping or formal-test
checkpoint selection. Ablations use seed 1234 unless explicitly changed.

Training uses `drop_last=True` for CBAF's globally pooled BatchNorm. The
1,305-image development pool therefore omits the final single shuffled record
of each epoch; the 1,450-image final pool has no remainder at batch size 2.

Checkpoints are written to `runs/morph/final/<seed>/`. Resume with `--resume`
and the same configuration, source code and manifest. Old square-resize,
custom-DWT or differently ordered feature checkpoints require their original
implementation; retrain after changing these settings.

## Evaluation

```bash
export CHECKPOINT=runs/morph/final/1234/checkpoint_last.pth
bash scripts/test_0m.sh
bash scripts/test_1m.sh
bash scripts/test_2m.sh
```

Repeat for each training seed. Threshold is 0.5, click seed is 1234, and each
per-image RNG is derived from dataset and image ID. The same first positive
GT-oracle click is used for 1-m and 2-m. The second click samples the largest
8-connected FN/FP component, chooses FN on equal area, and is omitted if
there is no error. It is prediction-dependent and need not have identical
coordinates across different models. Binary disks have radius 5 on the
128×128 prompt grid. Zero-click evaluation bypasses dense prompt projection.

Seeded data/click sampling and deterministic cuDNN settings are used.
PyTorch determinism runs in warning mode because CUDA bilinear-interpolation
backward is nondeterministic; bitwise-identical GPU training is not claimed.

Losses, click-error maps, predictions and metrics use the padded 1024 grid.
Per-image Dice/IoU are averaged within each dataset; the five-dataset macro
is an unweighted average of dataset means. Predictions, measured CSVs, click
records and provenance are saved locally under the run output directory.
Test-set duplicates remain in descriptive benchmark results.

## BKAI zero-shot evaluation

Only the labeled official training partition is an external evaluation pool;
BKAI must not enter MORPH fitting, development or checkpoint selection.
Arrange its 1,000 image records under `TestDataset/BKAI/images/` and their
explicitly prepared lossless binary GT under `TestDataset/BKAI/masks/`.
Image and mask extensions may differ if their stems match.

```bash
python scripts/prepare_bkai.py --data_root /path/to/dataset
python scripts/audit_content.py --data_root /path/to/dataset \
  --evaluation_manifests splits/local/bkai.csv --output results/bkai_audit
python test.py --data_root /path/to/dataset --protocol 1-m \
  --checkpoint runs/morph/final/1234/checkpoint_last.pth \
  --bkai_split splits/local/bkai.csv --output results/bkai/1234/1-m
```

Repeat 1-m/2-m for all three checkpoints. Raw BKAI JPEG masks contain color
labels and are rejected as binary GT. Supply the binary conversion used in
the experiment; this implementation does not infer an undocumented color
threshold from JPEG background noise. The competition test images without
GT are not substituted for this labeled evaluation pool.

## Repeated runs and paired statistics

`scripts/analyze_results.py` accepts measured `per_image_metrics.csv` files
and their sibling `evaluation_metadata.json`, including baseline exports
with the same format. ASPS and SegNext architectures are not substituted by
MORPH; their reproduced predictions must be supplied separately.

The metadata includes the CSV checksum, seed, protocol, training settings,
preprocessing, click radius and manifest hashes. Baseline exports must provide
these same provenance fields. Within-method runs may differ only in seed.

```bash
python scripts/analyze_results.py --family main_0m \
  --left /path/to/morph_seed1.csv /path/to/morph_seed2.csv /path/to/morph_seed3.csv \
  --right /path/to/asps_seed1.csv /path/to/asps_seed2.csv /path/to/asps_seed3.csv \
  --audit splits/local/content_audit.jsonl --output results/main_0m_statistics.json
```

Average each record across seeds first, then average record differences
within each RGB content group. Unresolved GT conflicts and source-training
overlap are excluded from inference. The report distinguishes `N_pair`,
`N_zero` and `n=N_pair-N_zero`. Tied absolute differences receive average
ranks. Two-sided signed-rank p-values use the conditional sign-permutation
distribution; rank-biserial effect sizes use signed rank sums. Mean/median
differences include zero pairs. Repeated-run CIs use sample SD (ddof=1) and
t(0.975,2)=4.3026527299. Three seeds are not additional image samples.

Holm families are separate: `main_0m` has 2 Dice/IoU comparisons;
`main_prompt` and `bkai_prompt` each have 4 metric-by-protocol comparisons.
For prompt families pass the six CSVs per method covering 1-m/2-m and all
three seeds. Content groups do not establish patient/video independence.

## Profiling and checks

```bash
pip install -r requirements-profile.txt
python scripts/profile_model.py --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --checkpoint /path/to/trained_model.pth --device cuda --output results/profile.json
pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider
python scripts/package_release.py --check
python scripts/package_release.py --output /path/outside/repository/MORPH-code.zip
```

Profiling uses FP32, batch size 1, 0-m, fvcore, MAC=1 FLOP, 20 warmups and
100 synchronized timed forwards. Memory is peak allocated memory, not
reserved memory. Unsupported profiler operations are reported; a partial
FLOP count must not be presented as a complete count. Hardware/environment
are measured at runtime rather than replaced by paper table values.

The symmetric library DWT is not JIT-traceable. For the FLOP trace its fixed
basis is precomputed, and separable two-tap Haar filtering MACs are added
analytically. The real library transform is restored before timing. Use
`--device cpu --count_only` for a complexity-only diagnostic without latency
or GPU-memory measurements.

The release packager includes only source/configuration/documentation and
generated-fixture tests. It excludes local datasets, per-image IDs, audit
results, predictions, weights, credentials, private paths and Git history.

## Acknowledgements

The encoders use Meta's [Segment Anything](https://github.com/facebookresearch/segment-anything)
and torchvision. Dataset partitioning follows PraNet. Dataset access and
pretrained weight distribution remain subject to their respective providers.
