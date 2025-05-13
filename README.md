# Qwen Business Strategy Fine‑Tuning & Evaluation

## Overview
This repository contains all the code and data you need to preprocess conversations, fine‑tune **Qwen** on business‑scenario strategy prompts, and evaluate the resulting model.

1. **Preprocess** raw dialogue data into JSONL format  
2. **Fine‑tune** the base Qwen model with the processed data  
3. **Run inference** to test the model on sample strategy questions  
4. **Compare** baseline and fine‑tuned outputs using Token‑F1, ROUGE‑L, and semantic similarity  

---

## Repository Structure
```text
.
├── preprocess.ipynb         ── Data cleaning & JSONL preparation
├── finetune.py              ── Fine‑tuning script (SFTTrainer configuration)
├── run_finetune_slurm.sh    ── SLURM wrapper for cluster training
├── test_inference.py        ── Quick inference on sample prompts
├── compare.py               ── Full evaluation & comparison
└── requirements.txt         ── Python package list (optional)
```

---

## Requirements
* Python ≥ 3.10  
* PyTorch + CUDA **or** HIP (for NVIDIA / AMD GPUs)  
* [🤗 Transformers](https://github.com/huggingface/transformers)  
* [🤗 Datasets](https://github.com/huggingface/datasets)  
* [TRL](https://github.com/huggingface/trl)  
* (Optional) Jupyter for running `preprocess.ipynb`  

> **Installation tip:** Make sure your GPU driver & CUDA/HIP toolkit match the PyTorch wheel you install.

---

## How to Run

### 1  Clone the repo & create a fresh environment
```bash
git clone <your‑fork‑url> qwen‑biz‑strategy
cd qwen‑biz‑strategy

# Create an env (name it however you like)
conda create -n qwen‑biz python=3.10 -y
conda activate qwen‑biz

# Install packages
pip install -r requirements.txt          # preferred
# ── or, manual ──
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121             transformers datasets trl jupyter
```

### 2  Pre‑process raw dialogue
Run the notebook interactively **or** headless; it writes `data/processed_data.jsonl` by default.
```bash
jupyter notebook preprocess.ipynb                  # interactive
# OR
jupyter nbconvert --execute --to notebook --inplace preprocess.ipynb
```
Adjust `OUTPUT_PATH` inside the notebook if you need a different location.

### 3  Single‑GPU fine‑tuning (local)
```bash
python finetune.py   --model_name_or_path <BASE_MODEL_DIR>   --train_file data/processed_data.jsonl   --output_dir checkpoints/qwen‑biz‑sft   --per_device_train_batch_size 4   --gradient_accumulation_steps 8   --num_train_epochs 3
```
*If you hit OOM errors, lower `--per_device_train_batch_size` or add `--fp16`.*

### 4  Multi‑GPU / cluster fine‑tuning (SLURM)
Make sure `run_finetune_slurm.sh` points to the same JSONL file and model path, then submit:
```bash
sbatch run_finetune_slurm.sh
```
The script contains typical SLURM directives:
```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
```
Adapt partition/queue, container image, and resource specs to your cluster.

### 5  Quick inference sanity check
```bash
python test_inference.py   --model_dir checkpoints/qwen‑biz‑sft   --prompt "As the COO of a retail chain, how would you…?"
```

### 6  Full evaluation & model comparison
1. **Baseline** model answers (cached once):
   ```bash
   python compare.py --model_dir <BASE_MODEL_DIR> --save_preds baseline.json
   ```
2. **Fine‑tuned** model answers:
   ```bash
   python compare.py --model_dir checkpoints/qwen‑biz‑sft --save_preds finetune.json
   ```
3. **Metrics** comparison:
   ```bash
   python compare.py --evaluate      --baseline baseline.json      --fine_tuned finetune.json
   ```
   Results (Token‑F1, ROUGE‑L, cosine similarity) print to console and save to `results/`.

### 7  Resume training after interruption
```bash
python finetune.py   --resume_from_checkpoint checkpoints/qwen‑biz‑sft   --continue_training
```

---

## Gotchas & Tips
* For very long dialogues, raise `--model_max_length` and `--block_size` inside `finetune.py` (e.g., 4096 tokens).  
* On AMD ROCm, avoid fragmentation by exporting:  
  ```bash
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  ```  
