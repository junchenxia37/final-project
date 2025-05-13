#!/usr/bin/env python3
import re
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── normalization & tokenization ────────────────────────────────────────────

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def tokenize(text):
    return normalize(text).split()

# ─── EM, token-F1, ROUGE-L ───────────────────────────────────────────────────

def exact_match_score(pred, ref):
    return int(normalize(pred) == normalize(ref))

def f1_score(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    c = sum(common.values())
    if c == 0:
        return 0.0
    prec, rec = c/len(p), c/len(r)
    return 2 * prec * rec / (prec + rec)

def _lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a,1):
        for j, y in enumerate(b,1):
            dp[i][j] = dp[i-1][j-1] + 1 if x==y else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    prec, rec = lcs/len(p), lcs/len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# ─── semantic sim via hidden states ──────────────────────────────────────────

class SemanticScorer:
    def __init__(self, lm, tokenizer):
        self.lm = lm.eval()
        self.lm.config.output_hidden_states = True
        self.tok = tokenizer
        self.device = next(lm.parameters()).device

    @torch.no_grad()
    def embed(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        out = self.lm(**enc)
        last_h = out.hidden_states[-1]              # [B, T, H]
        mask   = enc["attention_mask"].unsqueeze(-1)# [B, T, 1]
        summed = (last_h * mask).sum(dim=1)         # [B, H]
        lengths= mask.sum(dim=1)                    # [B, 1]
        return summed / lengths                     # [B, H]

    def score(self, preds, refs):
        sims = []
        for p, r in zip(preds, refs):
            ep = self.embed(p)
            er = self.embed(r)
            sims.append(F.cosine_similarity(ep, er).item())
        return sims

# ─── main evaluation ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) load test split
    test_ds = load_dataset(
        "json",
        data_files="test_data.jsonl",
        split="train",   # or "all" if it really is JSONL lines
        cache_dir="/gpfs/wolf2/olcf/trn040/scratch/jxia2/finetuning-hw"
    )
    
    test_ds = test_ds.shuffle(seed=42).select(range(10))

    # 2) load both models & tokenizers
    base_path = "./.cache/hf/Qwen/Qwen2.5-0.5B"
    ft_path   = "Qwen2.5-0.5B-SFT-business"

    tok_base = AutoTokenizer.from_pretrained(base_path, padding_side="right")
    lm_base  = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto")
    tok_ft   = AutoTokenizer.from_pretrained(ft_path,   padding_side="right")
    lm_ft    = AutoModelForCausalLM.from_pretrained(ft_path,   device_map="auto")

    lm_base.config.use_cache = False
    lm_ft.config.use_cache   = False

    # 3) iterate and generate
    preds_base, preds_ft, refs = [], [], []
    for example in test_ds:
        prompt = example["prompt"]
        gold   = example["completion"]

        # base model
        enc_b = tok_base(prompt, return_tensors="pt", truncation=True).to(lm_base.device)
        out_b = lm_base.generate(
            **enc_b,
            max_new_tokens=256,
            num_beams=4,
            eos_token_id=tok_base.eos_token_id,
            pad_token_id=tok_base.eos_token_id,
            early_stopping=True,
        )
        gen_b = out_b[0, enc_b["input_ids"].size(-1):]
        reply_b = tok_base.decode(gen_b, skip_special_tokens=True).strip()

        # finetuned model
        enc_f = tok_ft(prompt, return_tensors="pt", truncation=True).to(lm_ft.device)
        out_f = lm_ft.generate(
            **enc_f,
            max_new_tokens=256,
            num_beams=4,
            eos_token_id=tok_ft.eos_token_id,
            pad_token_id=tok_ft.eos_token_id,
            early_stopping=True,
        )
        gen_f = out_f[0, enc_f["input_ids"].size(-1):]
        reply_f = tok_ft.decode(gen_f, skip_special_tokens=True).strip()

        preds_base.append(reply_b)
        preds_ft.append(reply_f)
        refs.append(gold)

    # 4) compute metrics
    for name, preds in [("BASE", preds_base), ("FINETUNED", preds_ft)]:
        ems    = [exact_match_score(p, r) for p, r in zip(preds, refs)]
        f1s    = [f1_score(p, r)          for p, r in zip(preds, refs)]
        rouges = [rouge_l(p, r)           for p, r in zip(preds, refs)]
        sem    = SemanticScorer(lm_ft if name=="FINETUNED" else lm_base,
                                tok_ft if name=="FINETUNED" else tok_base)
        sims   = sem.score(preds, refs)

        print(f"\n--- {name} MODEL ---")
        print(f"Exact Match   : {sum(ems)/len(ems):.3f}")
        print(f"Token-F1      : {sum(f1s)/len(f1s):.3f}")
        print(f"ROUGE-L       : {sum(rouges)/len(rouges):.3f}")
        print(f"Semantic-Cos  : {sum(sims)/len(sims):.3f}")
