# Datos para figuras y Anexo de entrenamiento de V59: ejecuta celdas 0..9 del cuaderno y guarda lo necesario.
import json, sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
N = int(sys.argv[1]); S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"; os.chdir(f"{S}/v59_{N}")
cs = [''.join(c['source']) for c in json.load(open(f"curva_j_v59_{N}.ipynb"))['cells'] if c['cell_type'] == 'code']
g = {}
for k in range(10): exec(compile(cs[k], f"c{k}", "exec"), g)
RED, Z, DELTA, E = g["RED"], g["Z"], g["DELTA"], g["ESCZ"]
sp = lambda x: np.logaddexp(0, x)
res = {}
for nom, X, idx in (("entr", g["X_train"], g["idx_train"]), ("val", g["X_val"], g["idx_val"]), ("prueba", g["X_test"], g["idx_test"])):
    P = RED.predict([X, (Z[idx] / E).astype("float32")], batch_size=2048, verbose=0)
    lv = lambda z: (z[:, None] > np.arange(P[0].shape[1])[None, :]).astype(float)
    ce = sum((lv(Z[idx, j]) * sp(-P[j]) + (1 - lv(Z[idx, j])) * sp(P[j])).sum(1).mean() for j in range(N))
    mse = float(np.mean((P[-1][:, 0] - (DELTA[idx] - g["D_MED"]) / g["D_SD"]) ** 2))
    res[nom] = [float(ce), mse, float(ce + g["PESO_DELTA"] * mse)]
H = pd.read_csv(g["ARCHIVO_HISTORIAL"])
res["epocas"] = len(H); res["mejor"] = int(H.val_loss.idxmin()) + 1; res["reducc"] = int((H.learning_rate.diff().fillna(0) < 0).sum())
res["n"] = (len(g["idx_train"]), len(g["idx_val"]), len(g["idx_test"])); res["param"] = int(RED.count_params())
print(N, res); json.dump(res, open(f"{S}/v59/entren_{N}.json", "w"))
# curvas de los tres sujetos de ejemplo
SUJ = pd.read_csv(f"sujetos_v59_{N}.csv")
ids = list(g["POB"].id)
out = {"P": g["P_GRID"]}
for s in ("NT04", "HT20", "HT09"):
    i = ids.index(s); y = g["POB_L"][i]; out[f"{s}_y"] = y
    for m in ("minimax", "red ±1"):
        r = SUJ[(SUJ.sujeto == s) & (SUJ["método"] == m)].iloc[0]
        n = np.array([r[f"n{j+1}"] for j in range(N)]); out[f"{s}_{m}"] = g["curva_injerto"](g["theta"](n, r["delta"]))
np.savez(f"{S}/v59/curvas_ej_{N}.npz", **out)
