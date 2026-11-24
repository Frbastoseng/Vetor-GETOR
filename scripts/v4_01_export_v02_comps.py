"""v0.4 / Fase 0: exporta os compensadores da v0.2 em formato compacto (float32, tensores contiguos).

O arquivo original (results/v0.2/compensators.pt, 248 MB) guardava VISOES de matrizes d x d float64, e por
isso o torch.save gravava o armazenamento inteiro; ficou fora do Git e os roteadores publicados da v0.2 nao
eram reproduziveis publicamente. O arquivo compacto tem os MESMOS valores (convertidos para float32, que e a
precisao em que os compensadores sao usados) e e verificado elemento a elemento.
"""
import torch

from vetor.model_adapter import ROOT

src = torch.load(ROOT / "results" / "v0.2" / "compensators.pt", weights_only=False)
out = {"alpha": src["alpha"], "mean": [{"ybar": f["ybar"].float().clone()} for f in src["mean"]],
       "lr": {r: [{k: v.float().contiguous().clone() for k, v in f.items()} for f in fits] for r, fits in src["lr"].items()},
       "note": "copia float32 compacta de results/v0.2/compensators.pt (scripts/v4_01_export_v02_comps.py)"}
for r, fits in src["lr"].items():
    for a, b in zip(fits, out["lr"][r]):
        for k in a:
            assert torch.equal(a[k].float(), b[k])
dst = ROOT / "results" / "v0.2" / "compensators_f32.pt"
torch.save(out, dst)
print(dst, dst.stat().st_size / 1e6, "MB")
