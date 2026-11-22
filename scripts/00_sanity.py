"""Etapa 0: ambiente, dispositivo, carregamento do modelo, config e caminho real do MLP."""
from __future__ import annotations

import argparse
import json

import torch

from vetor.model_adapter import (ROOT, env_info, load_config, load_model, mlp_flops_per_token,
                                 model_info, pick_device, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config()
    device = pick_device(args.device or cfg["device"])
    model, tok = load_model(cfg, device)
    info = {"env": env_info(device), "model_id": cfg["model"]["id"],
            "revision": cfg["model"]["revision"], "model": model_info(model),
            "mlp_flops_per_token_per_layer": mlp_flops_per_token(model),
            "mlp_flops_per_token_all_layers": mlp_flops_per_token(model) * model.config.num_hidden_layers}
    layer0 = model.model.layers[0]
    info["decoder_layer_children"] = [n for n, _ in layer0.named_children()]
    info["mlp_children"] = {n: (tuple(m.weight.shape) if hasattr(m, "weight") else type(m).__name__) for n, m in layer0.mlp.named_children()}
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    info["smoke_generation"] = tok.decode(out[0])
    print(json.dumps(info, indent=2, ensure_ascii=False))
    write_json(ROOT / "results" / "00_sanity.json", info)


if __name__ == "__main__":
    main()
