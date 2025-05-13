# Qwen Business Strategy Fine-Tuning & Evaluation

## Overview

This repository provides an overview for my final project for LLM course.

1. **Preprocess** raw dialogue data into JSONL format
2. **Fine‑tune** the Qwen model on business‑scenario strategy prompts
3. **Run inference** you can run some inferences to play with the question-answering tasks
4. **Compare** model outputs using Token‑F1, ROUGE‑L, and semantic similarity metrics

---

## Repository Structure

```
.
├── preprocess.ipynb         # Data cleaning & JSONL preparation notebook
├── finetune.py              # Fine‑tuning script (SFTTrainer configuration)
├── run_finetune_slurm.sh    # SLURM wrapper for cluster training
├── test_inference.py        # Quick inference on a few sample prompts
├── compare.py               # Full evaluation & comparison script
└── test_data.jsonl          # Generated test split for evaluation (not necessarily to look at I think)
```

---

## Requirements

* Python 3.10+
* PyTorch with HIP/CUDA support (for AMD/NVIDIA GPUs)
* 🤗 Transformers (`pip install transformers`)
* 🤗 Datasets (`pip install datasets`)
* TRL (`pip install trl`)
* (Optional) Jupyter for `preprocess.ipynb`

---

