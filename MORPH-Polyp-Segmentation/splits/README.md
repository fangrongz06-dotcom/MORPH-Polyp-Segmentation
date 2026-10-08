# Fixed PraNet-style split

The CSV files contain dataset names and filenames from the existing local fixed
split. They contain no images, masks or machine-specific paths. No random
re-splitting is performed.

| Split | Images |
| --- | ---: |
| Training (900 Kvasir + 550 ClinicDB) | 1450 |
| Kvasir-SEG test | 100 |
| CVC-ClinicDB test | 62 |
| CVC-ColonDB test | 380 |
| ETIS test | 196 |
| CVC-300 test | 60 |

The original fixed benchmarks contain repeated images within/across some test
sets. Keep dataset-level metrics separate when interpreting independent samples.

For BKAI zero-shot evaluation, supply a separate CSV with the same
`dataset,filename` header and `bkai` as the dataset name. Place files in
`TestDataset/BKAI/images/` and `TestDataset/BKAI/masks/` under your data root.
