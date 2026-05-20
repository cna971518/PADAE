# PADAE: Poisoning Attack Defense and Accuracy Enhancement Based on Kolmogorov-Smirnov Test for Federated Learning in Intrusion Detection

## Overview

PADAE is a federated learning defense framework for intrusion detection. It is designed to detect and reduce the impact of poisoning attacks from malicious clients during federated model training.

The framework includes three main modules:

1. **MQV: Model Quality Validation**  
   Uses the K-S statistic to evaluate whether a local client model performs normally on the server-side validation set.

2. **MPDD: Model Parameter Distribution Detection**  
   Uses the two-sample K-S test to compare parameter distributions between client models and identify abnormal local models.

3. **CMA: Contribution-based Model Aggregation**  
   Aggregates retained secure client models with contribution-based weights to reduce the influence of poisoned models and improve global model robustness.

---

## Dataset

This project uses two public intrusion detection datasets.

### UNSW-NB15

Official source:  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

### CIC-IDS2017

Official source:  
https://www.unb.ca/cic/datasets/ids-2017.html

The original datasets are not included in this repository. Please download them from the official websites and place them under the `data/raw/` directory.

---

## Supported Attacks

This project supports three poisoning attacks.

### 1. Partial Data Tampering (PDT)

PDT modifies selected feature values while keeping the original labels unchanged. It is designed to simulate a clean-label poisoning attack.

Supported PDT modes:

```text
mean_shift
swap
```

Example:

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type pdt --malicious_clients 0 --tamper_ratio 0.4 --alpha 0.0 --pdt_mode swap

python main.py --dataset UNSW-NB15 --K 20 --attack_type pdt --malicious_clients 0,1,2,3 --tamper_ratio 0.4 --alpha 0.0 --pdt_mode swap
```

### 2. Label-flipping Attack

Label flipping changes selected class labels while keeping feature values unchanged.

Supported modes:

```text
random
targeted_pair
```

Example:

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type label_flip --malicious_clients 0 --flip_ratio 0.1 --label_flip_mode targeted_pair

python main.py --dataset UNSW-NB15 --K 20 --attack_type label_flip --malicious_clients 0,1,2,3 --flip_ratio 0.1 --label_flip_mode targeted_pair
```

### 3. Random Weight Attack

Random weight attack modifies local model weights before upload.

Supported modes:

```text
random
noise
```

Example:

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type random_weight --malicious_clients 0 --weight_attack_mode random --weight_noise_scale 1.0

python main.py --dataset UNSW-NB15 --K 20 --attack_type random_weight --malicious_clients 0,1,2,3 --weight_attack_mode random --weight_noise_scale 1.0
```

---

## Requirements

Python 3.8 or later is recommended.

Main packages:

```text
numpy
pandas
scikit-learn
scipy
tensorflow
matplotlib
imbalanced-learn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install numpy pandas scikit-learn scipy tensorflow matplotlib imbalanced-learn
```

---

## Project Structure

```text
PADAE/
│
├── main.py
├── args.py
├── server.py
├── client.py
├── model.py
├── data_process.py
├── attacks.py
│
├── preprocess/
│   ├── Preprocess_cicids2017.py
│   └── preprocess_unsw_nb15.py
│
├── data/
│   ├── raw/
│   │   ├── UNSW-NB15/
│   │   └── CIC-IDS2017/
│   │
│   └── processed/
│       ├── UNSW-NB15/
│       │   ├── clients_10/
│       │   └── clients_20/
│       │
│       └── CIC-IDS2017/
│           ├── clients_10/
│           └── clients_20/
│
├── results/
└── README.md
```

---

## File Description

| File / Path | Description |
|---|---|
| `main.py` | Main entry point for running PADAE experiments |
| `args.py` | Argument parser and experiment configuration |
| `server.py` | Server-side federated learning, MQV, MPDD, CMA, and aggregation |
| `client.py` | Local client training and attack execution |
| `model.py` | DNN model definition |
| `data_process.py` | Client data loading and server-side validation set construction |
| `attacks.py` | PDT, label-flipping, and random weight attack implementation |
| `preprocess/Preprocess_cicids2017.py` | Preprocessing script for CIC-IDS2017 |
| `preprocess/preprocess_unsw_nb15.py` | Preprocessing script for UNSW-NB15 |
| `data/raw/` | Stores the original downloaded datasets |
| `data/processed/` | Stores the processed datasets and federated client splits |
| `data/processed/<dataset>/clients_10/` | Stores client data for the 10-client setting |
| `data/processed/<dataset>/clients_20/` | Stores client data for the 20-client setting |
| `results/` | Stores experiment logs, summaries, and client-level results |

---

## Data Preparation

Place the original downloaded datasets under:

```text
data/raw/UNSW-NB15/
data/raw/CIC-IDS2017/
```

Then run the corresponding preprocessing script:

```bash
python preprocess/preprocess_unsw_nb15.py
```

```bash
python preprocess/Preprocess_cicids2017.py
```

After preprocessing, the processed federated client datasets will be stored under:

```text
data/processed/UNSW-NB15/clients_10/
data/processed/UNSW-NB15/clients_20/
data/processed/CIC-IDS2017/clients_10/
data/processed/CIC-IDS2017/clients_20/
```

---

## Basic Usage

### No Attack

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type none
```

### PDT Attack

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type pdt --malicious_clients 0 --tamper_ratio 0.4 --alpha 0.0 --pdt_mode swap

python main.py --dataset UNSW-NB15 --K 20 --attack_type pdt --malicious_clients 0,1,2,3 --tamper_ratio 0.4 --alpha 0.0 --pdt_mode swap
```

### Label-flipping Attack

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type label_flip --malicious_clients 0 --flip_ratio 0.1 --label_flip_mode targeted_pair

python main.py --dataset UNSW-NB15 --K 20 --attack_type label_flip --malicious_clients 0,1,2,3 --flip_ratio 0.1 --label_flip_mode targeted_pair
```

### Random Weight Attack

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type random_weight --malicious_clients 0 --weight_attack_mode random --weight_noise_scale 1.0

python main.py --dataset UNSW-NB15 --K 20 --attack_type random_weight --malicious_clients 0,1,2,3 --weight_attack_mode random --weight_noise_scale 1.0
```

### Use CMA Aggregation

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type pdt --malicious_clients 0 --tamper_ratio 0.4 --alpha 0.0 --pdt_mode swap --aggregation_method cma --cma_beta 0.10 --cma_lambda 0.80
```

### Use FedAvg Aggregation

```bash
python main.py --dataset UNSW-NB15 --K 10 --attack_type pdt --malicious_clients 0,1,2,3 --tamper_ratio 1.0 --alpha 0.9 --pdt_mode mean_shift --aggregation_method fedavg
```

---

## Important Arguments

| Argument | Description |
|---|---|
| `--dataset` | Dataset name, such as `UNSW-NB15` or `CIC-IDS2017` |
| `--K` | Number of federated clients |
| `--E` | Number of local training epochs |
| `--r` | Number of global training rounds |
| `--B` | Batch size |
| `--lr` | Learning rate |
| `--optimizer` | Optimizer used for local training |
| `--attack_type` | Attack type: `none`, `pdt`, `label_flip`, or `random_weight` |
| `--malicious_clients` | Comma-separated malicious client indices, e.g., `0,1,2,3` |
| `--tamper_ratio` | Ratio of selected samples to tamper in PDT |
| `--alpha` | Mean-shift strength in PDT |
| `--pdt_mode` | PDT mode: `mean_shift` or `swap` |
| `--flip_ratio` | Ratio of labels to flip |
| `--label_flip_mode` | Label flipping mode: `random` or `targeted_pair` |
| `--weight_attack_mode` | Random weight attack mode: `random` or `noise` |
| `--weight_noise_scale` | Noise strength for random weight attack |
| `--ks_threshold` | MQV K-S statistic threshold |
| `--pvalue_threshold` | MPDD p-value threshold |
| `--abnormal_round_threshold` | Number of consecutive abnormal rounds before permanent client removal |
| `--aggregation_method` | Aggregation method, such as `fedavg` or `cma` |
| `--cma_beta` | Proportion of high-contribution clients in CMA |
| `--cma_lambda` | Total aggregation weight assigned to high-contribution clients in CMA |
| `--seed` | Random seed |

---

## Output

Each experiment creates a result folder under:

```text
results/
```

Common output files:

| File | Description |
|---|---|
| `console_log.txt` | Full console output of the experiment |
| `experiment_settings.txt` | Experiment settings and attack parameters |
| `client_status_summary.csv` | Client-level status, validation accuracy, KS value, PVA mean, final accuracy, and aggregation status |
| `summary_results.csv` | Summary of the current experiment |
| `error_log.txt` | Error traceback if the experiment fails |

A global summary is saved as:

```text
results/all_experiments_summary.csv
```

---

## Notes on Reproducibility

The experiment pipeline supports random seed control through:

```bash
--seed 42
```

For better reproducibility, the code fixes Python, NumPy, and TensorFlow random seeds. However, small numerical differences may still occur depending on GPU, CUDA, cuDNN, and TensorFlow versions.

---

## Citation

If you use this code, please cite the related manuscript:

```text
PADAE: Poisoning Attack Defense and Accuracy Enhancement Model Based on the Kolmogorov-Smirnov Test for Federated Learning in Intrusion Detection.
```

---

## License

This repository is provided for academic and research purposes.

---

## Contact

For questions, please contact the corresponding author.
