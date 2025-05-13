#!/usr/bin/env python3
import os

# ──────────────────────────────────────────────────────────────────────────────
# 0) Configure HIP allocator to reduce fragmentation (AMD GPU)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

# 1) Pin your GPU
# ──────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load tokenizer & base model
# ──────────────────────────────────────────────────────────────────────────────
base_model_path = "./.cache/hf/Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    model_max_length=99999,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer._tokenizer.enable_truncation(
    max_length=99999,
    strategy="longest_first",
    direction="left"
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda:0",
)
model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load, split (80/10/10), rename & shuffle your dataset
# ──────────────────────────────────────────────────────────────────────────────
dataset_path = "processed_data.jsonl"

# Load entire JSONL
full_ds = load_dataset(
    "json",
    data_files=dataset_path,
    split="all",
    cache_dir="/gpfs/wolf2/olcf/trn040/scratch/jxia2/finetuning-hw"
)

# 3.2) Shuffle and take half for everything
full_ds = full_ds.shuffle(seed=42)
half_size = len(full_ds) // 2
half_ds   = full_ds.select(range(half_size))
print(f"▸ Using {half_size} examples total (50% of data)")

# 3.3) Split that half into 80% train / 20% temp (val+test)
split1  = half_ds.train_test_split(test_size=0.2, seed=42)
train_ds, temp_ds = split1["train"], split1["test"]

# 3.4) Split temp into 50% val / 50% test → each is 10% of original half
split2  = temp_ds.train_test_split(test_size=0.5, seed=42)
val_ds, test_ds = split2["train"], split2["test"]

print(f"▸ Train: {len(train_ds)} examples")
print(f"▸ Val  : {len(val_ds)} examples")
print(f"▸ Test : {len(test_ds)} examples")

# 3.5) Rename for SFTTrainer
train_ds = train_ds.rename_column("response", "completion")
val_ds   = val_ds.rename_column(  "response", "completion")
test_ds  = test_ds.rename_column( "response", "completion")

# Shuffle splits
train_ds = train_ds.shuffle(seed=42)
val_ds   = val_ds.shuffle(seed=42)
test_ds  = test_ds.shuffle(seed=42)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Configure fine-tuning hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
training_args = SFTConfig(
    max_length=2048,
    num_train_epochs=3,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",     # run validation every eval_steps
    fp16=False,                # keep FP16 off for AMD
    bf16=True,                 # enable BF16 if supported
    output_dir="qwen-business-sft",
)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Instantiate & launch SFTTrainer
# ──────────────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# 6) Save your fine-tuned model
# ──────────────────────────────────────────────────────────────────────────────
finetuned_model_path = "Qwen2.5-0.5B-SFT-business"
trainer.save_model(finetuned_model_path)
print(f"✅ Model saved to '{finetuned_model_path}'")

# ──────────────────────────────────────────────────────────────────────────────
# 7) (Optional) Save test split for later evaluation
# ──────────────────────────────────────────────────────────────────────────────
test_ds.to_json("test_data.jsonl")
print("✅ Test split saved to 'test_data.jsonl'")
