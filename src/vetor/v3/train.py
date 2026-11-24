"""Treino dos modulos da v0.3 (LLM congelado).

L = L_task + alpha L_distill + beta L_latency + gamma L_stability
- L_task: CE nos tokens de resposta (assistente).
- L_distill: KL(professor || aluno) sobre os top-K logits do professor (pre-calculados, cache).
- L_latency: beta * sum_l c_l * media(a_l)  (c_l em us medidos) — so para arquiteturas com portao.
- L_stability: gamma * KL(p_FULL || p_theta) em texto generico de referencia.
"""
from __future__ import annotations

import json
import math
import random

import torch
import torch.nn.functional as F

from ..model_adapter import ROOT
from .modules import V3MLP


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def encode_chat(tok, messages, max_len=512):
    """ids e mascara de rotulo (1 nos tokens do assistente)."""
    ids, mask = [], []
    for i in range(len(messages)):
        prefix = tok.apply_chat_template(messages[:i + 1], tokenize=False)
        cur = tok(prefix, add_special_tokens=False).input_ids
        new = cur[len(ids):]
        mask += [1 if messages[i]["role"] == "assistant" else 0] * len(new)
        ids = cur
    return ids[:max_len], mask[:max_len]


def batches(examples, tok, max_tokens=8192, seed=0, shuffle=True):
    enc = [encode_chat(tok, e["messages"]) for e in examples]
    order = list(range(len(enc)))
    if shuffle:
        random.Random(seed).shuffle(order)
    # agrupa por tamanho dentro de blocos para reduzir padding
    blocks = [sorted(order[i:i + 256], key=lambda j: len(enc[j][0])) for i in range(0, len(order), 256)]
    out = []
    for blk in blocks:
        cur = []
        for j in blk:
            L = max([len(enc[k][0]) for k in cur + [j]])
            if cur and L * (len(cur) + 1) > max_tokens:
                out.append(cur)
                cur = []
            cur.append(j)
        if cur:
            out.append(cur)
    if shuffle:
        random.Random(seed + 1).shuffle(out)
    return enc, out


def collate(enc, idx, pad_id, device):
    L = max(len(enc[j][0]) for j in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    att = torch.zeros(len(idx), L, dtype=torch.long)
    lab = torch.zeros(len(idx), L, dtype=torch.bool)
    for r, j in enumerate(idx):
        x, m = enc[j]
        ids[r, :len(x)] = torch.tensor(x)
        att[r, :len(x)] = 1
        lab[r, :len(m)] = torch.tensor(m, dtype=torch.bool)
    return ids.to(device), att.to(device), lab.to(device)


def set_modules_enabled(model, on: bool):
    """Liga/desliga correcoes e substituicoes (para obter p_FULL dentro do mesmo grafo)."""
    for layer in model.model.layers:
        if isinstance(layer.mlp, V3MLP):
            layer.mlp._disabled = not on


def gate_means(model) -> dict[int, torch.Tensor]:
    out = {}
    for layer in model.model.layers:
        if isinstance(layer.mlp, V3MLP) and layer.mlp.gate_stats is not None:
            out[layer.mlp.layer_idx] = layer.mlp.gate_stats.mean()
    return out


def distill_loss(logits, top_idx, top_lp, mask):
    """KL(T||S) aproximado nos top-K do professor (top_lp = log p_T, renormalizado no top-K).
    logits [B,T,V], top_idx/top_lp [B,T,K] alinhados a posicao que PREVE o token t+1, mask [B,T]."""
    lsm = F.log_softmax(logits.float(), -1)
    s_lp = lsm.gather(-1, top_idx)
    pt = top_lp.exp()
    kl = (pt * (top_lp - s_lp)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1)
