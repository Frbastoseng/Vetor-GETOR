"""Bateria de tarefas da v0.4 sobre o lm-evaluation-harness, em particoes NUNCA avaliadas antes no projeto.

Cada tarefa base do lm-eval e reconstruida pelo registro com overrides (test_split -> particao nao usada),
renomeada para v4_<nome>. DEV/FINAL por SHA-256(f"{tarefa}:{indice}:vetor-v0.4"), com limite por metade.
IF-v0.4: prompts novos (smol-smoltalk, nunca usados) + restricoes verificaveis geradas e checadas pelo codigo
oficial do IFEval (lm_eval.tasks.ifeval); dados em configs/v4_ifeval.jsonl.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..model_adapter import ROOT

MANIFEST = ROOT / "configs" / "v4_task_manifest.json"
IF_DATA = ROOT / "configs" / "v4_ifeval.jsonl"
SALT = "vetor-v0.4"

# nome base -> especificacao. dev/final = limites de exemplos por metade.
SUITE: dict[str, dict] = {
    "hellaswag": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none", "chance": 0.25,
                  "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "piqa": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none", "chance": 0.5,
             "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "winogrande": {"cat": "raciocinio", "lang": "en", "metric": "acc,none", "chance": 0.5,
                   "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "arc_challenge": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none", "chance": 0.25,
                      "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "arc_easy": {"cat": "conhecimento", "lang": "en", "metric": "acc_norm,none", "chance": 0.25,
                 "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "sciq": {"cat": "conhecimento", "lang": "en", "metric": "acc_norm,none", "chance": 0.25,
             "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "boolq": {"cat": "compreensao", "lang": "en", "metric": "acc,none", "chance": 0.5,
              "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "gsm8k": {"cat": "matematica", "lang": "en", "metric": "exact_match,flexible-extract", "chance": 0.0, "generative": True,
              "chat": True,  # modelo Instruct: no formato bruto responde <|im_end|> de imediato (registrado no plano, sec. 9)
              "ov": {"test_split": "train", "fewshot_split": "test", "training_split": None}, "dev": 100, "final": 250,
              "max_gen_toks": 256},
    "ifeval": {"cat": "instrucoes", "lang": "en", "metric": "prompt_level_loose_acc,none", "chance": 0.0, "generative": True,
               "chat": True, "ov": "IF", "max_gen_toks": 384},
    "arc_pt": {"cat": "portugues", "lang": "pt", "metric": "acc_norm,none", "chance": 0.25,
               "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
    "assin2_rte": {"cat": "portugues", "lang": "pt", "metric": "f1,none", "chance": 0.5, "corpus_f1": True,
                   "ov": {"test_split": "train", "validation_split": None}, "dev": 200, "final": 400},
}

# restricoes do IFEval usadas no IF-v0.4 (parametros explicitos, sorteados com semente fixa)
_IF_KW = {
    "change_case:english_lowercase": lambda r: {},
    "change_case:english_capital": lambda r: {},
    "punctuation:no_comma": lambda r: {},
    "startend:quotation": lambda r: {},
    "detectable_format:title": lambda r: {},
    "detectable_format:json_format": lambda r: {},
    "detectable_format:number_bullet_lists": lambda r: {"num_bullets": int(r.integers(2, 6))},
    "detectable_format:number_highlighted_sections": lambda r: {"num_highlights": int(r.integers(1, 4))},
    "length_constraints:number_words": lambda r: {"relation": str(r.choice(["less than", "at least"])),
                                                  "num_words": int(r.choice([40, 60, 80, 120, 150]))},
    "length_constraints:number_sentences": lambda r: {"relation": str(r.choice(["less than", "at least"])),
                                                      "num_sentences": int(r.integers(2, 7))},
    "startend:end_checker": lambda r: {"end_phrase": str(r.choice(["Is there anything else I can help with?",
                                                                    "Let me know if you have additional questions.",
                                                                    "Any other questions?"]))},
    "detectable_content:postscript": lambda r: {"postscript_marker": str(r.choice(["P.S.", "P.P.S"]))},
    "keywords:existence": lambda r: {"keywords": [str(w) for w in r.choice(
        ["example", "important", "simple", "result", "reason", "detail", "summary", "process"], 2, replace=False)]},
    "keywords:forbidden_words": lambda r: {"forbidden_words": [str(w) for w in r.choice(
        ["very", "really", "thing", "basically", "also", "just"], 2, replace=False)]},
}


def _h(task: str, i: int) -> int:
    return int(hashlib.sha256(f"{task}:{i}:{SALT}".encode()).hexdigest(), 16)


def build_ifeval_data(prompts_dev: list[dict], prompts_final: list[dict], seed: int = 20261124) -> list[dict]:
    from lm_eval.tasks.ifeval import instructions_registry as reg
    rng = np.random.default_rng(seed)
    ids = sorted(_IF_KW)
    rows = []
    for split, ps in (("dev", prompts_dev), ("final", prompts_final)):
        for p in ps:
            n = 1 + int(rng.random() < 0.5)
            chosen, descs, kws = [], [], []
            for iid in rng.permutation(ids):
                if len(chosen) == n:
                    break
                if any(iid in reg.INSTRUCTION_CONFLICTS[c] or c in reg.INSTRUCTION_CONFLICTS[iid] for c in chosen):
                    continue
                inst = reg.INSTRUCTION_DICT[iid](iid)
                kw = _IF_KW[iid](rng)
                descs.append(inst.build_description(**kw))
                args = inst.get_instruction_args() or {}
                chosen.append(str(iid))
                kws.append({k: v for k, v in args.items() if v is not None})
            rows.append({"key": len(rows), "split": split, "source_id": p["id"],
                         "prompt": p["prompt"].strip() + " " + " ".join(descs),
                         "instruction_id_list": chosen, "kwargs": kws})
    return rows


def _ifeval_kwargs_unified(rows):
    keys = sorted({k for r in rows for kw in r["kwargs"] for k in kw})
    for r in rows:
        r["kwargs"] = [{k: kw.get(k) for k in keys} for kw in r["kwargs"]]
    return rows


def build_tasks(names: list[str]):
    """Tarefas v4_<nome> (objetos Task do lm-eval) a partir do registro + overrides."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    out = {}
    for n in names:
        ov = SUITE[n]["ov"]
        if ov == "IF":
            ov = {"dataset_path": "json", "dataset_name": None, "dataset_kwargs": {"data_files": {"train": str(IF_DATA)}},
                  "test_split": "train"}
        ov = {"task": f"v4_{n}", **ov}
        if "max_gen_toks" in SUITE[n]:
            base = tm._factory._load_full_config(tm._entry(n), None).get("generation_kwargs", {}) or {}
            ov["generation_kwargs"] = {**base, "max_gen_toks": SUITE[n]["max_gen_toks"]}
        out[n] = tm._factory.build(tm._entry(n), overrides=ov, registry=tm._index)
    return out


def build_manifest() -> dict:
    if MANIFEST.exists():
        raise FileExistsError("Manifesto de tarefas da v0.4 ja existe (pre-registro); nao recriar.")
    from .data4 import load_prompts
    rows = _ifeval_kwargs_unified(build_ifeval_data(load_prompts("chat_en:if_dev"),
                                                    load_prompts("chat_en:if_final", allow_final=True)))
    with open(IF_DATA, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tasks = build_tasks(list(SUITE))
    man = {"salt": SALT, "rule": "DEV/FINAL por paridade de sha256(f'{tarefa}:{i}:{salt}'); dentro de cada metade, "
                                 "os primeiros N pela ordem do hash", "tasks": {}}
    for n, t in tasks.items():
        size = len(list(t.eval_docs))
        if n == "ifeval":
            dev = [r["key"] for r in rows if r["split"] == "dev"]
            fin = [r["key"] for r in rows if r["split"] == "final"]
        else:
            order = sorted(range(size), key=lambda i: _h(n, i))
            dev = sorted([i for i in order if _h(n, i) % 2 == 0][:SUITE[n]["dev"]])
            fin = sorted([i for i in order if _h(n, i) % 2 == 1][:SUITE[n]["final"]])
        man["tasks"][n] = {"split_used": SUITE[n]["ov"] if n != "ifeval" else "configs/v4_ifeval.jsonl",
                           "n_split": size, "dev": dev, "final": fin}
    MANIFEST.write_text(json.dumps(man, indent=0), encoding="utf-8")
    return man


def split_indices(split: str, names: list[str], *, allow_final: bool = False) -> dict[str, list[int]]:
    if split == "final" and not allow_final:
        raise PermissionError("Metade FINAL da v0.4 so com allow_final=True.")
    man = json.loads(MANIFEST.read_text())["tasks"]
    return {n: man[n][split] for n in names}


def run(model, tok, names: list[str], split: str, *, batch_size: int = 16, allow_final: bool = False,
        device: str = "mps", policy=None) -> dict:
    """policy: vetor.v4.attach.Policy (opcional). O regime e ajustado por grupo (loglik / gen)."""
    """Roda as tarefas na metade pedida; metricas por tarefa e escore POR EXEMPLO (comparacao pareada)."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from ..v3.bench import _val

    idx = split_indices(split, names, allow_final=allow_final)
    tasks = build_tasks(names)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device, max_gen_toks=384)
    out = {"split": split, "tasks": {}}
    groups: dict = {}
    for n in names:
        groups.setdefault((bool(SUITE[n].get("chat")), bool(SUITE[n].get("generative"))), []).append(n)
    for (chat, gen), ns in groups.items():
        if policy is not None:
            policy.regime = "gen" if gen else "loglik"
            policy.skips, policy.tokens = 0.0, 0
        r = simple_evaluate(model=lm, tasks=[tasks[n] for n in ns], samples={f"v4_{n}": idx[n] for n in ns},
                            log_samples=True, apply_chat_template=chat, fewshot_as_multiturn=chat, bootstrap_iters=0, random_seed=0,
                            numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
        for n in ns:
            met = SUITE[n]["metric"]
            mkey, mfilter = (met.split(",") + ["none"])[:2]
            per = [(s["doc_id"], _val(s.get(mkey, float("nan"))))
                   for s in r["samples"].get(f"v4_{n}", []) if s.get("filter", "none") == mfilter]
            out["tasks"][n] = {"metric": met, "value": r["results"].get(f"v4_{n}", {}).get(met), "n": len(per),
                               "per_example": per}
            if policy is not None:
                out["tasks"][n]["group_skips_per_token"] = policy.rate()
    return out
