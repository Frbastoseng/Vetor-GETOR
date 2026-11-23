"""Bateria de qualidade da v0.3 sobre o lm-evaluation-harness (implementacoes padrao da comunidade).

Cada tarefa tem seus exemplos de avaliacao divididos em DEV (desenvolvimento) e FINAL (intocado) por
hash SHA-256 de (tarefa, indice, sal) — manifesto versionado em configs/v3_eval_manifest.json.
Resultados sao guardados POR EXEMPLO (acerto 0/1) para comparacoes pareadas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..model_adapter import ROOT

MANIFEST = ROOT / "configs" / "v3_eval_manifest.json"
SALT = "vetor-v0.3"

# tarefa -> (categoria, idioma, metrica principal, opcoes de execucao)
SUITE: dict[str, dict] = {
    # raciocinio / senso comum
    "arc_challenge": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none"},
    "winogrande": {"cat": "raciocinio", "lang": "en", "metric": "acc,none"},
    "hellaswag": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none"},
    "piqa": {"cat": "raciocinio", "lang": "en", "metric": "acc_norm,none"},
    # conhecimento factual
    "arc_easy": {"cat": "conhecimento", "lang": "en", "metric": "acc_norm,none"},
    "openbookqa": {"cat": "conhecimento", "lang": "en", "metric": "acc_norm,none"},
    "sciq": {"cat": "conhecimento", "lang": "en", "metric": "acc_norm,none"},
    # interpretacao de texto
    "boolq": {"cat": "interpretacao", "lang": "en", "metric": "acc,none"},
    "belebele_eng_Latn": {"cat": "interpretacao", "lang": "en", "metric": "acc_norm,none"},
    # alucinacao / incerteza
    "truthfulqa_mc1": {"cat": "veracidade", "lang": "en", "metric": "acc,none"},
    "truthfulqa_mc2": {"cat": "veracidade", "lang": "en", "metric": "acc,none"},
    # matematica verificavel
    "arithmetic_2da": {"cat": "matematica", "lang": "en", "metric": "acc,none"},
    "arithmetic_3da": {"cat": "matematica", "lang": "en", "metric": "acc,none"},
    "arithmetic_2dm": {"cat": "matematica", "lang": "en", "metric": "acc,none"},
    "gsm8k": {"cat": "matematica", "lang": "en", "metric": "exact_match,flexible-extract", "generative": True},
    # seguimento de instrucoes
    "ifeval": {"cat": "instrucoes", "lang": "en", "metric": "prompt_level_loose_acc,none", "generative": True,
               "chat": True},
    # programacao com testes executaveis
    "mbpp": {"cat": "programacao", "lang": "en", "metric": "pass_at_1,none", "generative": True, "code": True},
    "humaneval": {"cat": "programacao", "lang": "en", "metric": "pass@1,create_test", "generative": True, "code": True},
    # portugues brasileiro
    "arc_pt": {"cat": "portugues", "lang": "pt", "metric": "acc_norm,none"},
    "hellaswag_pt": {"cat": "portugues", "lang": "pt", "metric": "acc_norm,none"},
    "belebele_por_Latn": {"cat": "portugues", "lang": "pt", "metric": "acc_norm,none"},
    "global_piqa_parallel_cloze_por_latn_braz": {"cat": "portugues", "lang": "pt", "metric": "acc_norm,none"},
    "assin2_rte": {"cat": "portugues", "lang": "pt", "metric": "f1,none", "corpus_f1": True},
}


def _is_final(task: str, idx: int) -> bool:
    return int(hashlib.sha256(f"{task}:{idx}:{SALT}".encode()).hexdigest(), 16) % 2 == 1


def build_manifest(task_sizes: dict[str, int]) -> dict:
    if MANIFEST.exists():
        raise FileExistsError("Manifesto v0.3 ja existe (pre-registro); nao recriar.")
    man = {"salt": SALT, "rule": "final se sha256(f'{task}:{idx}:{salt}') impar; senao dev",
           "tasks": {}}
    for t, n in task_sizes.items():
        dev = [i for i in range(n) if not _is_final(t, i)]
        fin = [i for i in range(n) if _is_final(t, i)]
        man["tasks"][t] = {"n": n, "dev": dev, "final": fin}
    MANIFEST.write_text(json.dumps(man), encoding="utf-8")
    return man


def split_indices(split: str, tasks: list[str], *, allow_final: bool = False) -> dict[str, list[int]]:
    if split == "final" and not allow_final:
        raise PermissionError("Metade FINAL da v0.3 so com allow_final=True (avaliacao final).")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    return {t: man[t][split] for t in tasks}


def leaf_tasks(task_dict) -> dict:
    out = {}
    for k, v in task_dict.items():
        if isinstance(v, dict):
            out.update(leaf_tasks(v))
        else:
            out[getattr(k, "group_name", k) if not isinstance(k, str) else k] = v
    return out


def task_sizes(names: list[str]) -> dict[str, int]:
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    sizes = {}
    for t in names:
        for leaf, obj in leaf_tasks(get_task_dict([t], tm)).items():
            sizes[str(leaf)] = len(list(obj.eval_docs))
    return sizes


def run(model, tok, tasks: list[str], split: str, *, batch_size: int = 32, allow_final: bool = False,
        max_gen_toks: int | None = None, device: str = "mps") -> dict:
    """Roda as tarefas (folhas) na metade pedida. Retorna metricas por tarefa e acertos por exemplo."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    idx = split_indices(split, tasks, allow_final=allow_final)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device,
              max_gen_toks=max_gen_toks or 256)
    out = {"split": split, "tasks": {}}
    groups = {}
    for t in tasks:
        spec = SUITE.get(t) or SUITE.get(_parent(t)) or {}
        groups.setdefault((spec.get("chat", False), spec.get("code", False)), []).append(t)
    for (chat, code), ts in groups.items():
        r = simple_evaluate(model=lm, tasks=ts, samples={t: idx[t] for t in ts}, log_samples=True,
                            apply_chat_template=chat, confirm_run_unsafe_code=code, bootstrap_iters=0,
                            random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
        for t in ts:
            spec = SUITE.get(t) or SUITE.get(_parent(t)) or {}
            met = spec.get("metric", "acc,none")
            mkey, mfilter = (met.split(",") + ["none"])[:2]
            per = [(s["doc_id"], _val(s.get(mkey, s.get(met, float("nan")))))
                   for s in r["samples"].get(t, []) if s.get("filter", "none") == mfilter]
            out["tasks"][t] = {"metric": met, "value": r["results"].get(t, {}).get(met),
                               "n": len(per), "per_example": per}
    return out


def _val(v):
    """Escore por exemplo: float, ou par (gabarito, predicao) para metricas de corpus (F1 macro)."""
    if isinstance(v, (list, tuple)):
        return [int(x) if not isinstance(x, str) else x for x in v]
    return float(v)


def _parent(leaf: str) -> str:
    for g in ("global_mmlu_pt",):
        if leaf.startswith(g):
            return g
    return leaf
