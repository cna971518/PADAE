# PADAE: Poisoning Attack Defense and Accuracy Enhancement Based on the Kolmogorov–Smirnov Statistic for Federated Intrusion Detection

## Overview

PADAE is a poisoning defense framework for federated intrusion detection. It consists of three modules:

1. **MQV — Model Quality Validation:** evaluates local models using K–S scores on a trusted server validation set.
2. **MPDD — Model Parameter Deviation Detection:** analyzes first-layer weight updates using cross-round average cosine distances, clustering, and consecutive-round confirmation.
3. **CMA — Contribution-based Model Aggregation:** assigns aggregation weights according to the retained models' quality.

## Datasets

- **UNSW-NB15:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CIC-IDS2017:** https://www.unb.ca/cic/datasets/ids-2017.html

Download the original CSV files from the official sources. Datasets are not included.

## Supported Attacks

| Attack | Operation | Scenarios |
|---|---|---|
| Partial Data Tampering (PDT) | Replaces selected features using paired-class samples; labels remain unchanged. | `pdt_0p5`, `pdt_1p0` |
| Label flipping | Exchanges labels between predefined class pairs; features remain unchanged. | `lf_0p5`, `lf_1p0` |
| Random weight | Replaces all parameter tensors before upload with Gaussian random values having a mean of 0 and a standard deviation of 1. | `rw_1p0` |

`clean` disables attacks. Client configurations are `1-out-of-10`, `4-out-of-10`, and `4-out-of-20`.

## Requirements

Use Python 3.9 in a separate environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The training stack uses TensorFlow 2.10.0. GPU execution also requires compatible driver, CUDA, and cuDNN installations.

## Project Structure

```text
PADAE/
├── main.py / args.py / configuration.py
├── server.py / client.py / model.py
├── metrics.py / mpdd.py / aggregators.py / attacks.py
├── calibrate_mpdd.py / calibration_io.py
├── run_experiments.py / summarize_results.py / benchmark.py
├── preprocess/
├── configs/
├── tests/ / docs/
└── data/
    ├── raw/UNSW-NB15/       raw/CIC-IDS2017/
    └── processed/UNSW-NB15/  processed/CIC-IDS2017/
```

## File Description

| File / Path | Description |
|---|---|
| `main.py`, `args.py`, `configuration.py` | Single-run interface and experiment settings. |
| `server.py`, `client.py`, `model.py` | Federated training and DNN classifier. |
| `metrics.py`, `mpdd.py`, `aggregators.py` | MQV, MPDD, CMA, and baseline aggregation methods. |
| `attacks.py`, `data_process.py`, `preprocess/` | Offline poisoning, data loading, and training-only preprocessing. |
| `calibrate_mpdd.py`, `calibration_io.py` | Clean-data threshold calculation and loading. |
| `run_experiments.py`, `summarize_results.py` | Batch experiments and complete-seed summaries. |
| `benchmark.py` | Controlled server-load measurement. |

## Data Preparation

Place the following files in their dataset directories:

```text
data/raw/UNSW-NB15/
    UNSW-NB15_1.csv
    UNSW-NB15_2.csv
    UNSW-NB15_3.csv
    UNSW-NB15_4.csv
data/raw/CIC-IDS2017/
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
```

UNSW-NB15 uses the 49-column headerless files; CIC-IDS2017 uses the 79-column MachineLearningCSV files.

```bash
python preprocess/preprocess_unsw_nb15.py
python preprocess/Preprocess_cicids2017.py
```

Each script creates `clients_10/` and `clients_20/` under `data/processed/<dataset>/`, containing local train/test CSVs, `Val.csv`, `Test.csv`, and preprocessing metadata. Data are split before fitting transformations. UNSW-NB15 uses training-only one-hot encoding, PCA, StatAvg, and local SMOTE; CIC-IDS2017 uses StatAvg without PCA or SMOTE. Existing nonempty output directories are not overwritten.

## Basic Usage

### MPDD Threshold Calibration

Generate clean monitoring trajectories and calculate a separation threshold for each dataset:

```bash
python calibrate_mpdd.py --execute
```

Defaults: both datasets, K = 10 and 20, development seeds 42–46, 12 rounds, and margin `0.000001`. This produces 20 clean trajectories and `results/mpdd_thresholds.json`. Without `--execute`, only a training plan is written.

After two initial observation rounds, identify records in which the same benign client belongs to the smaller cluster of a valid partition for two consecutive rounds. Take the smaller separation ratio from each pair of rounds, then set the dataset threshold to the maximum of these values plus the margin. Independent test data are neither loaded nor evaluated by monitoring runs. See `docs/IMPLEMENTATION.md` for the formula.

An explicit single-dataset command is:

```bash
python calibrate_mpdd.py --datasets UNSW-NB15 --clients 10 20 --seeds 42 43 44 45 46 --margin 0.000001 --execute
```

Use a new `--output-root` and `--output` when changing the calibration plan. To recalculate from the clean trajectories just generated:

```bash
python calibrate_mpdd.py --results results/calibration --margin 0.000001 --output results/mpdd_thresholds_check.json
```

### Single Experiment

```bash
python main.py --config configs/unsw_4of10.json --calibration-file results/mpdd_thresholds.json
python main.py --config configs/cic_4of10.json --calibration-file results/mpdd_thresholds.json --scenario lf_0p5 --output results/cic_lf05_s100
```

Use `--scenario clean`, `pdt_1p0`, `lf_1p0`, or `rw_1p0` to change the condition; select a new output directory for each run. Without `--calibration-file`, the fixed experiment thresholds in `configuration.py` are used.

| Parameter | UNSW-NB15 | CIC-IDS2017 |
|---|---:|---:|
| MQV threshold | 0.40 | 0.70 |
| CMA β | 0.2 | 0.2 |
| CMA λ | 0.5 | 0.9 |

CMA sorts retained models by K–S score and uses `ceil(β × retained_count)` for the high-contribution group, with at least one member. A single retained model is used unchanged.

### Method Comparison and Ablation

```bash
python run_experiments.py --preset comparison --calibration-file results/mpdd_thresholds.json --execute
python run_experiments.py --preset ablation --datasets UNSW-NB15 --calibration-file results/mpdd_thresholds.json --execute
```

Comparison includes PADAE, FedAvg, Median, Trimmed-mean, DDaBA, FLTrust, and FLEST. Ablation covers all eight combinations of MQV, MPDD, and CMA. Defaults use training seeds 100–104: 1,260 comparison runs or 200 UNSW-NB15 ablation runs. Omit `--execute` to inspect the plan first. `--resume` skips verified completed runs from the same plan; it does not resume unfinished training rounds.

## Important Arguments

| Argument | Description |
|---|---|
| `--dataset`, `--client-setting` | Dataset and client configuration. |
| `--scenario`, `--method`, `--seed` | Attack, method, and training seed. |
| `--rounds`, `--local-epochs`, `--batch-size`, `--learning-rate` | Defaults: 12, 40, 500, and 0.01. |
| `--calibration-file` | Load computed dataset-level MPDD thresholds. |
| `--tau-mqv`, `--tau-r` | Explicit threshold settings; a conflicting calibrated threshold is rejected. |
| `--beta`, `--lam` | CMA group proportion and group weight. |
| `--data-root`, `--output` | Processed-data directory and single-run output directory. |

Run each entry point with `--help` for its full interface.

## Output

Runs save settings, data/environment metadata, per-round decisions, client removal records, ACC, Macro-F1, per-class metrics, and final model weights. `DONE.json` records output integrity. Calibration runs save monitoring records without independent-test metrics.

```bash
python summarize_results.py --results results/ablation --seeds 100 101 102 103 104 --output results/ablation_summary
```

Summaries report the mean and sample standard deviation across complete seed groups. Results from incomplete conditions or incompatible data or execution environments are not automatically pooled.

## Citation

Please cite the related PADAE manuscript when using this code or method.

## License

Provided for academic and research use.

## Contact

Please contact the corresponding author for questions.
