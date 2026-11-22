"""Ablacao controlada em um LLM real; precisa de transformers + rede.

Este smoke-test NAO implementa ainda geracao guiada nem benchmark de latencia.
Foco: reproduzir baseline e calcular KL de ablacao isolada do ultimo token.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rodar como python scripts/02_ablate_smoke.py a partir da raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vetor.core import patch_llama_like_model  # noqa: E402


EXAMPLES = [
    "Once upon a time there was a small bird in a green forest.",
    "The solar panels generate electricity during the day.",
    "A computer program consists of instructions and data.",
    "The mathematician wrote a theorem on a white board.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = p.parse_args()

    available = torch.backends.mps.is_available()
    device = "mps" if (args.device == "auto" and available) else ("cpu" if args.device == "auto" else args.device)
    if device == "mps" and not available:
        raise SystemExit("MPS indisponivel; rode com --device cpu")
    print(f"Device={device}; torch={torch.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # dtype=torch.float32 evita divergencia que poderia mascarar o smoke-test.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    wraps = patch_llama_like_model(model)
    print(f"Model={args.model}; layers={len(wraps)}; hidden={model.config.hidden_size}")

    all_rows = []
    limit = min(max(0, args.max_layers), len(wraps))
    with torch.inference_mode():
        for ex_idx, text in enumerate(EXAMPLES):
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 3:
                raise RuntimeError("Example tokenization too short")
            prefix = ids[:, :-1]
            true_next = int(ids[0, -1].item())
            for wrap in wraps:
                wrap.capture = True
                wrap.skip_decode = False
                wrap.ablate_last = False
            full = model(input_ids=prefix, use_cache=False).logits[:, -1, :].float()
            full_logp = torch.log_softmax(full, dim=-1)
            full_p = full_logp.exp()
            captured = [w.last_input.clone() for w in wraps]
            assert all(x is not None and x.shape[-1] == model.config.hidden_size for x in captured)
            for layer_idx in range(limit):
                wraps[layer_idx].ablate_last = True
                try:
                    alt = model(input_ids=prefix, use_cache=False).logits[:, -1, :].float()
                finally:
                    wraps[layer_idx].ablate_last = False
                alt_logp = torch.log_softmax(alt, dim=-1)
                kl = float((full_p * (full_logp - alt_logp)).sum().item())
                delta_nll = float((full_logp[0, true_next] - alt_logp[0, true_next]).item())
                row = {
                    "example": ex_idx,
                    "layer": layer_idx,
                    "kl_nats": kl,
                    "delta_nll_nats": delta_nll,
                    "safe_initial": kl <= 0.02 and delta_nll <= 0.05,
                    "prefix_tokens": int(prefix.shape[1]),
                }
                all_rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    os.makedirs("results", exist_ok=True)
    path = Path("results/ablation_smoke.json")
    path.write_text(json.dumps({"model": args.model, "device": device, "results": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo: {path}")


if __name__ == "__main__":
    main()
