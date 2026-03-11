# A Multitask Transformer for Offensive Language Detection and Target Identification in HateBR

This repository contains the official code for the paper **"A Multitask Transformer for Offensive Language Detection and Target Identification in HateBR"**.

The project proposes a multitask learning (MTL) model based on a shared **BERTimbau** encoder for hierarchical hate speech analysis in Brazilian Portuguese.

---

## 1) Project Overview

The model jointly solves three tasks on the HateBR dataset:

1. **Binary offensiveness detection** (Offensive vs. Non-offensive)
2. **Offensiveness severity classification** (None, Slightly, Moderately, Highly offensive)
3. **Hate target identification** (9-label multilabel classification)

The main idea is to replace independent pipelines with a unified architecture, improving consistency between predictions across hierarchy levels.

---

## 2) What this repository does

The experimental pipeline:

- Loads and normalizes HateBR from Hugging Face (`ruanchaves/hatebr`)
- Tokenizes text with **BERTimbau Base** (`neuralmind/bert-base-portuguese-cased`)
- Trains:
  - three single-task baselines (binary, level, target)
  - one unified multitask model
- Evaluates all models with task-specific metrics
- Generates plots and per-run result files
- Aggregates results across two seeds

---

## 3) Installation

### 3.1) Requirements

- Python 3.10+
- (Recommended) CUDA GPU for faster training

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4) How to run

From the repository root:

```bash
python -m src.run_experiments --seeds 88,89 --use_fp16
```

### Useful optional arguments

- `--epochs 20`
- `--batch_size 16`
- `--lr 2e-5`
- `--max_length 128`
- `--patience 2`
- `--target_threshold 0.5`
- `--mask_urls_users` / `--no-mask_urls_users`

If you do not provide `--seeds`, the script uses two seeds automatically: `seed` and `seed + 1`.

---

## 5) Results (paper summary)

Experiments were run with two seeds (88 and 89). Mean ± standard deviation:

### 5.1) Binary offensiveness

- **Accuracy**: STL `0.8996 ± 0.0096` vs MTL `0.9111 ± 0.0045`
- **F1 (positive class)**: STL `0.8965 ± 0.0128` vs MTL `0.9088 ± 0.0053`
- **MCC**: STL `0.8012 ± 0.0167` vs MTL `0.8232 ± 0.0086`

### 5.2) Severity level

- **Accuracy**: STL `0.6443 ± 0.0121` vs MTL `0.6454 ± 0.0076`
- **Macro-F1**: STL `0.4634 ± 0.0031` vs MTL `0.4717 ± 0.0069`
- **Balanced Accuracy**: STL `0.4641 ± 0.0007` vs MTL `0.4720 ± 0.0056`

### 5.3) Hate target multilabel

- **Subset Accuracy**: STL `0.9421 ± 0.0071` vs MTL `0.8993 ± 0.0263`
- **Micro-F1**: STL `0.5923 ± 0.1011` vs MTL `0.4197 ± 0.0639`

### 5.4) Consistency findings

- **Target inconsistency rate (MTL)**: `0.0000 ± 0.0000` (0%)
- **Level inconsistency rate (MTL)**: `0.0014 ± 0.0010` (~0.14%)

In short: MTL improves the primary moderation signal and enforces hierarchical consistency, but suffers negative transfer on fine-grained multilabel targets under severe class imbalance.

---

## 6) Where results are saved

All artifacts are saved under `results/`.

Typical structure:

```text
results/
  experiment_<seedA>_<seedB>/
    final_results.txt
    seed_<seed>/
      singletask_offensive/
        checkpoints/
        metrics.json
        final_results.txt
      singletask_level/
        checkpoints/
        metrics.json
        final_results.txt
      singletask_target/
        checkpoints/
        metrics.json
        final_results.txt
      multitask/
        checkpoints/
        metrics.json
        final_results.txt
        plots/
          comparison_bar.png
          confusion_level.png
          pr_curve_bin.png
          inconsistency_bar.png
          target_f1_bar.png
```

The script also creates local Hugging Face cache directories in `.hf_cache/` and temporary files in `results/tmp/`.

---

## 7) Submission information

This work was **submitted to PROPOR**.

**Article title:** *A Multitask Transformer for Offensive Language Detection and Target Identification in HateBR*.

---

## 8) Citation

If you use this code, please cite the paper above (BibTeX entry to be added after publication).
