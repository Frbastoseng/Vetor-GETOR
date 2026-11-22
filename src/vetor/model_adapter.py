"""Carregamento do modelo, ambiente e utilidades de reprodutibilidade."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def pick_device(pref: str = "auto") -> str:
    mps = torch.backends.mps.is_available()
    if pref == "auto":
        if not mps:
            print("[vetor] AVISO: MPS indisponivel, usando CPU")
        return "mps" if mps else "cpu"
    if pref == "mps" and not mps:
        raise RuntimeError("MPS solicitado mas indisponivel")
    return pref


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device: str) -> None:
    """Sincroniza o dispositivo. Usar SO nos limites de trechos cronometrados."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class Throttle:
    """Limita a CPU a `lag` forwards a frente da GPU via eventos MPS.

    Achado medido (Etapa 1): sem limite, a CPU enfileira o decode inteiro e o MPS fica
    ~2x mais lento por token (6.6 vs 3.3 ms). Esperar o evento de `lag` passos atras nao
    esvazia o pipeline da GPU (nao e um synchronize global). Aplicado a TODAS as politicas.
    """

    def __init__(self, device: str, lag: int = 2):
        self.on = device == "mps" and lag > 0
        self.lag = lag
        self.events: list = []

    def step(self) -> None:
        if not self.on:
            return
        e = torch.mps.Event()
        e.record()
        self.events.append(e)
        if len(self.events) > self.lag:
            self.events.pop(0).synchronize()


def load_model(cfg: dict, device: str, dtype: torch.dtype | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mc = cfg["model"]
    dtype = dtype or getattr(torch, mc.get("dtype", "float32"))
    tok = AutoTokenizer.from_pretrained(mc["id"], revision=mc.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(mc["id"], revision=mc.get("revision"), dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_info(model) -> dict:
    c = model.config
    mlp = model.model.layers[0].mlp
    inner = getattr(mlp, "inner", mlp)
    return {
        "architectures": c.architectures,
        "num_hidden_layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "hidden_act": c.hidden_act,
        "vocab_size": c.vocab_size,
        "attn_implementation": getattr(c, "_attn_implementation", None),
        "mlp_path": "model.model.layers[i].mlp",
        "mlp_class": f"{type(inner).__module__}.{type(inner).__name__}",
        "dtype": str(next(model.parameters()).dtype),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def mlp_flops_per_token(model) -> int:
    c = model.config
    return 6 * c.hidden_size * c.intermediate_size


def weights_fingerprint(model) -> str:
    """Hash SHA-256 de todos os pesos (para provar congelamento)."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def env_info(device: str) -> dict:
    import transformers

    def _cmd(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "cpu": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mem_bytes": int(_cmd(["sysctl", "-n", "hw.memsize"]) or 0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
    }


def new_run_id(tag: str) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"_{tag}"


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p
