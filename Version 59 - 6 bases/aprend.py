import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
T = {"es": dict(ep="época", perd="pérdida", tr="entrenamiento", va="validación", rest="pesos conservados",
                a=lambda n: f"Primer nivel, {n} materiales", b=lambda n: f"Tubo, {n} materiales"),
     "en": dict(ep="epoch", perd="loss", tr="training", va="validation", rest="weights kept",
                a=lambda n: f"First level, {n} materials", b=lambda n: f"Tube, {n} materials")}
for L, t in T.items():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for i, n in enumerate((6, 5)):
        for k, (f, tit) in enumerate(((f"{S}/v58_{n}/historial_entrenamiento_v58_{n}_cadena.csv", t["a"](n)),
                                      (f"{S}/v59_{n}/historial_entrenamiento_cadena_tubo_v59_{n}.csv", t["b"](n)))):
            a = ax[k, i]; H = pd.read_csv(f); e = np.arange(1, len(H) + 1)
            a.plot(e, H.loss, color="#1f77b4", label=t["tr"]); a.plot(e, H.val_loss, color="#ff7f0e", label=t["va"])
            b = int(H.val_loss.idxmin()) + 1 if k == 1 else len(H)
            a.axvline(b, color="gray", ls="--"); a.text(b, a.get_ylim()[1], t["rest"], rotation=90, va="top", ha="right", color="gray", fontsize=8)
            a.set_yscale("log"); a.set_title(tit, fontsize=12); a.set_xlabel(t["ep"]); a.set_ylabel(t["perd"]); a.grid(alpha=.3, which="both"); a.legend()
    fig.tight_layout(); fig.savefig(f"{S}/v59/fig/curvas_aprendizaje_v59_{L}.png", dpi=150); plt.close(fig)
print("ok")
