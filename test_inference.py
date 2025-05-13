import re
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── normalization & tokenization ────────────────────────────────────────────

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def tokenize(text):
    return normalize(text).split()

# ─── EM, token-F1, ROUGE-L (same as before) ─────────────────────────────────

def exact_match_score(pred, ref):
    return int(normalize(pred) == normalize(ref))

def f1_score(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r: return 0.0
    common = Counter(p) & Counter(r)
    c = sum(common.values())
    if c == 0: return 0.0
    prec, rec = c/len(p), c/len(r)
    return 2*prec*rec/(prec+rec)

def _lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a,1):
        for j, y in enumerate(b,1):
            dp[i][j] = dp[i-1][j-1]+1 if x==y else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r: return 0.0
    lcs = _lcs(p, r)
    prec, rec = lcs/len(p), lcs/len(r)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

# ─── semantic sim via your LM’s hidden states ────────────────────────────────

class SemanticScorer:
    def __init__(self, lm, tokenizer):
        """
        lm: your AutoModelForCausalLM
        tokenizer: its AutoTokenizer
        """
        self.lm = lm.eval()
        self.tok = tokenizer
        # turn on hidden states
        self.lm.config.output_hidden_states = True
        self.device = next(lm.parameters()).device

    @torch.no_grad()
    def embed(self, text):
        enc = self.tok(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True).to(self.device)
        out = self.lm(**enc)  # returns logits + hidden_states
        # last layer hidden states: tuple of (layer0…layerN), we want the last
        last_h = out.hidden_states[-1]              # [B, T, H]
        mask   = enc["attention_mask"].unsqueeze(-1)# [B, T, 1]
        summed = (last_h * mask).sum(dim=1)         # [B, H]
        lengths= mask.sum(dim=1)                    # [B, 1]
        return summed / lengths                     # mean-pooled [B, H]

    def score(self, preds, refs):
        sims = []
        for p, r in zip(preds, refs):
            ep = self.embed(p)
            er = self.embed(r)
            sims.append(F.cosine_similarity(ep, er).item())
        return sims

# ─── putting it all together ─────────────────────────────────────────────────

if __name__ == "__main__":
    finetuned = "Qwen2.5-0.5B-SFT-business"
    tok = AutoTokenizer.from_pretrained(finetuned, padding_side="right")
    lm  = AutoModelForCausalLM.from_pretrained(finetuned, device_map="auto")

    data = [
        {
            "title": "Turnaround in healthcare",
            "prompt": (
                "<|im_start|>system\nYou are an experienced business strategist.<|im_end|>\n"
                "<|im_start|>user\nOur small private-equity-backed healthcare firm is losing money and "
                "has negative revenue growth. How would you recalibrate operations, leverage our network effects, "
                "and drive a path to profitability?\n<|im_start|>assistant\n"
            ),
            "gold": (
                "First, conduct a process audit to identify cost leaks in your supply chain and staffing. "
                "Then renegotiate payer contracts, redeploy underutilized network partners for shared savings, "
                "and implement a lean staffing model; finally, introduce telehealth services to boost utilization..."
            ),
        },
        # …add your other 4 items…
    ]

    preds, refs = [], []
    for item in data:
        enc = tok(item["prompt"], return_tensors="pt", truncation=True).to(lm.device)
        out = lm.generate(
            **enc,
            max_new_tokens=256,
            num_beams=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            early_stopping=True,
        )
        gen = out[0, enc["input_ids"].size(-1):]
        reply = tok.decode(gen, skip_special_tokens=True).strip()
        preds.append(reply); refs.append(item["gold"])
        print(f"\n=== {item['title']} ===\nPRED: {reply}\nGOLD: {item['gold']}\n")

    # compute EM/F1/ROUGE
    ems    = [exact_match_score(p,r) for p,r in zip(preds, refs)]
    f1s    = [f1_score(p,r)          for p,r in zip(preds, refs)]
    rouges = [rouge_l(p,r)           for p,r in zip(preds, refs)]

    print(f"Exact Match   : {sum(ems)/len(ems):.3f}")
    print(f"Token-F1      : {sum(f1s)/len(f1s):.3f}")
    print(f"ROUGE-L       : {sum(rouges)/len(rouges):.3f}")

    # semantic sim
    sem = SemanticScorer(lm, tok)
    sims = sem.score(preds, refs)
    print(f"Semantic-Cos  : {sum(sims)/len(sims):.3f}")

