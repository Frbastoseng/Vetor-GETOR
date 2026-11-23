"""v0.3: pre-calcula os top-K logits do professor (SmolLM2-1.7B-Instruct) nos tokens de resposta.

Usado so no treino (nunca na inferencia). Tokenizador identico ao do aluno (verificado).
Saida: data/v3_teacher_top{K}_{split}.pt  — por exemplo: indices [n,K] int32 e log p_T [n,K] float16,
nas posicoes p que preveem um token de resposta (mask[p+1] = 1).
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from vetor.model_adapter import ROOT, load_config, pick_device
from vetor.v3.train import batches, collate, load_jsonl

TEACHER = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    args = ap.parse_args()
    cfg = load_config()
    device = pick_device(cfg["device"])
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"])
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16).to(device).eval()
    rev = getattr(teacher.config, "_commit_hash", None)
    for split in args.splits:
        ex = load_jsonl(ROOT / "data" / f"v3_{split}.jsonl")
        enc, bl = batches(ex, tok, max_tokens=8192, shuffle=False)
        store = [None] * len(enc)
        t0 = time.time()
        with torch.inference_mode():
            for idx in tqdm(bl, desc=f"professor {split}"):
                ids, att, lab = collate(enc, idx, tok.pad_token_id or 0, device)
                lg = teacher(input_ids=ids, attention_mask=att).logits.float()
                lp = F.log_softmax(lg, -1)
                top = lp.topk(args.k, dim=-1)
                pred_mask = torch.zeros_like(lab)
                pred_mask[:, :-1] = lab[:, 1:]  # posicao p preve token p+1
                for r, j in enumerate(idx):
                    m = pred_mask[r]
                    store[j] = (top.indices[r][m].to(torch.int32).cpu(), top.values[r][m].half().cpu())
        torch.save({"teacher": TEACHER, "teacher_revision": rev, "k": args.k, "examples": store},
                   ROOT / "data" / f"v3_teacher_top{args.k}_{split}.pt")
        mass = torch.cat([s[1].float().exp().sum(-1) for s in store if s is not None and len(s[1])])
        print(f"{split}: {len(store)} exemplos, {sum(len(s[0]) for s in store)} posicoes, "
              f"massa media no top-{args.k} = {mass.mean():.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
