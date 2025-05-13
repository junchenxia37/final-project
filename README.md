# Qwen Business Strategy Fine-Tuning & Evaluation

## Overview

This repository provides all the code and configuration necessary to:

1. **Preprocess** raw dialogue data into JSONL format
2. **Fine‑tune** the Qwen 2.5B model on business‑scenario strategy prompts
3. **Run inference** with both base and fine‑tuned models
4. **Compare** model outputs using EM, Token‑F1, ROUGE‑L, and semantic similarity metrics

---

## Repository Structure

```
.
├── preprocess.ipynb         # Data cleaning & JSONL preparation notebook
├── finetune.py              # Fine‑tuning script (SFTTrainer configuration)
├── run_finetune_slurm.sh    # SLURM wrapper for cluster training
├── test_inference.py        # Quick inference on a few sample prompts
├── compare.py               # Full evaluation & comparison script
└── test_data.jsonl          # Generated test split for evaluation
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

## Setup

1. **Clone & enter the repo**

   ```bash
   git clone https://your.repo.url.git
   cd your.repo
   ```
2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install torch transformers datasets trl
   ```

---

## 1. Data Preprocessing

Open and run `preprocess.ipynb` to:

* Load raw chat logs or business‑scenario transcripts
* Clean, normalize, and export to `processed_data.jsonl`

Result:

```
processed_data.jsonl  # ready for fine‑tuning
```

---

## 2. Fine‑Tuning

### Local (single‑GPU)

```bash
python finetune.py \
  --data-path processed_data.jsonl \
  --base-model ./.cache/hf/Qwen/Qwen2.5-0.5B \
  --output-dir qwen-business-sft
```

### HPC via SLURM

1. Adjust resources in `run_finetune_slurm.sh`
2. Submit:

   ```bash
   sbatch run_finetune_slurm.sh
   ```

Checkpoint saved to `qwen-business-sft/`.

---

## 3. Quick Inference

Run a short inference test:

```bash
python test_inference.py \
  --model-path qwen-business-sft \
  --n-samples 5
```

Prints generated responses vs. gold references.

---

## 4. Full Evaluation & Comparison

With `test_data.jsonl` ready, execute:

```bash
python compare.py \
  --test-file test_data.jsonl \
  --base-model ./.cache/hf/Qwen/Qwen2.5-0.5B \
  --ft-model qwen-business-sft
```

This will:

1. Load test examples
2. Generate outputs from both base and fine‑tuned models
3. Compute EM, Token‑F1, ROUGE‑L, and Semantic‑Cosine
4. Print a side‑by‑side performance summary

---

## Notes & Tips

* **Cache location**: If you encounter "disk quota exceeded," set `cache_dir` in `load_dataset` to a scratch path.
* **BF16 vs. FP16**: For AMD GPUs, enable BF16 (`bf16=True`); NVIDIA users can use FP16.
* **Memory tuning**: Adjust `gradient_accumulation_steps` in `finetune.py` to fit GPU memory.

---

## License & Citation

*Released under the MIT License.*

If you use this code in your research, please cite:

> Xia, J., … (2025). "Human‑Capital‑Based AI Strategy: Fine‑Tuning LLMs for Business Scenarios." *Under Review*.
