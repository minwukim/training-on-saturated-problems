
# 🧠 Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning

This repository contains the **official code** for the paper
***Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning***.

The code supports **evaluation**, **failure-prefix dataset construction**, and **RLVR training via GRPO**, using **TRL**, **vLLM**, **DeepSpeed**, and **Accelerate**.

---

## 📁 Repository Structure

```text
.
├── environment.yml          # Conda environment definition
├── zero3.yaml               # Accelerate + DeepSpeed (ZeRO-3) config
├── README.md
├── data/                    # Failure-prefix-conditioned datasets
│   ├── iteration1_target_acc_25.csv
│   ├── iteration1_target_acc_50.csv
│   ├── iteration1_target_acc_75.csv
│   └── iteration2_target_acc_50.csv
├── eval/
│   ├── eval_config.yaml     # Evaluation configuration
│   └── evaluation.py        # Evaluation runner
└── train/
    ├── GRPO_config.yaml     # GRPO / RLVR training config
    └── GRPO_trainer.py      # Training entry point
```

---

## ⚙️ Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate failure-prefix-conditioning
```

### Requirements

* CUDA-enabled GPUs
* [DeepSpeed](https://www.deepspeed.ai/)
* 🤗 `accelerate` (configured for your cluster / node setup)
* vLLM (for fast multi-process inference)

---

## 🚀 How to Run

### 🔍 Evaluation

Run evaluation using the provided YAML config:

```bash
python eval/evaluation.py --config eval_config.yaml
```

This script measures rollout accuracy and recovery behavior under prefix conditioning.

---

### 🎯 RLVR Training (GRPO via TRL + vLLM)

We perform **Reinforcement Learning with Verifiable Rewards (RLVR)** using **Group Relative Policy Optimization (GRPO)**.

* **TRL** handles policy optimization
* **vLLM** is used for fast, parallel rollout generation during reward evaluation
* **DeepSpeed ZeRO-3** enables efficient large-model training

Launch training with:

```bash
accelerate launch \
  --config_file zero3.yaml \
  --num_processes <NUM_PROCESSES> \
  train/GRPO_trainer.py \
  --config GRPO_config.yaml
```

Replace `<NUM_PROCESSES>` with the number of GPU processes available.

---

## 🧪 Failure-Prefix-Conditioned Datasets

The `data/` directory contains curated datasets used to study learning on **saturated problems**.
File names directly encode the construction iteration and target rollout accuracy threshold ( \tau ).

| Dataset file                   | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `iteration1_target_acc_25.csv` | Iteration 1, target accuracy ( \tau = 0.25 )                        |
| `iteration1_target_acc_50.csv` | Iteration 1, target accuracy ( \tau = 0.50 ) **(main setting)**     |
| `iteration1_target_acc_75.csv` | Iteration 1, target accuracy ( \tau = 0.75 )                        |
| `iteration2_target_acc_50.csv` | Iteration 2, ( \tau = 0.50 ), iterative failure-prefix conditioning |

These datasets differ only in the saturation threshold and conditioning iteration, enabling controlled comparisons.

---
