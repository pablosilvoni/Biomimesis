# V58: validación externa (las 98 curvas de V54) con la red en cadena y el minimax, ambos con tope de 20 láminas.
#   Uso: python3 ext_v58.py N    (después de ejecutar curva_j_v58_N.ipynb, que deja pesos y caché del minimax)
import json, sys, os, time
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
N = int(sys.argv[1]); S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
os.chdir(f"{S}/v58_{N}")
nb = json.load(open(f"curva_j_v58_{N}.ipynb"))
cs = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
g = {}
for k in range(13):                         # celdas 0–12: datos, minimax, herramientas y red en cadena (se cargan los pesos)
    exec(compile(cs[k], f"c{k}", "exec"), g)
A = g["A"]; ZM = g["Z_MAX"]
d = np.load(f"{S}/v54_{N}/v54_arreglos_{N}.npz"); T = pd.read_csv(f"{S}/v54_{N}/v54_curvas_externas_{N}.csv")
Y = d["Y_EXT"]; assert np.allclose(d["X"], g["X"]) and np.allclose(d["A"], A), "otra grilla o base"
t0 = time.time(); Zm = np.array([g["redondeo_minimax"](y) for y in Y]); ms_mm = (time.time() - t0) / len(Y) * 1e3
t0 = time.time(); Zc = g["predecir_cadena"](Y); Zb = g["mejorar_vecinos"](Y, Zc); ms_red = (time.time() - t0) / len(Y) * 1e3
em, ec, eb = g["emax"](Y, Zm), g["emax"](Y, Zc), g["emax"](Y, Zb)
T["mm20_emax_%"], T["cad_emax_%"], T["cadpm1_emax_%"] = em, ec, eb
T["mm_sin_tope_emax_%"] = T["ent_emax_%"]
for j in range(N):
    T[f"mm20_z{j + 1}"], T[f"cad_z{j + 1}"], T[f"cadpm1_z{j + 1}"] = Zm[:, j], Zc[:, j], Zb[:, j]
T["igual_mm"] = (Zc == Zm).all(1); T["igual_mm_pm1"] = (Zb == Zm).all(1)
T.to_csv(f"v58_externas_{N}.csv", index=False)
ok = T[T.J & T.acepta & T.en_banda]                 # V58: solo las curvas dentro de y1–y2 (las que la red puede reproducir)
filas = []
for (f, gr), D in ok.groupby(["fuente", "grupo"], sort=False):
    filas.append({"fuente": f, "grupo": gr, "n": len(D),
                  "minimax ≤3": int((D["mm20_emax_%"] <= 3).sum()), "minimax peor": D["mm20_emax_%"].max(),
                  "cadena ≤3": int((D["cad_emax_%"] <= 3).sum()), "cadena peor": D["cad_emax_%"].max(),
                  "cadena±1 ≤3": int((D["cadpm1_emax_%"] <= 3).sum()), "cadena±1 peor": D["cadpm1_emax_%"].max(),
                  "minimax sin tope ≤3": int((D["mm_sin_tope_emax_%"] <= 3).sum())})
R = pd.DataFrame(filas); R.to_csv(f"v58_externas_resumen_{N}.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(2).to_string(index=False))
P, M = ok[ok.protocolo == "P"], ok[ok.protocolo == "M"]
res = {"N": N, "aceptadas": len(ok), "ms_minimax": ms_mm, "ms_red_pm1": ms_red,
       "max_laminas_mm20": int(Zm.max()), "curvas_en_tope_mm20": int((Zm == ZM).any(1).sum()),
       "max_laminas_sin_tope": int(T[[c for c in T.columns if c.startswith("z") and c[1:].isdigit()]].values.max())}
for nom, D in (("P", P), ("M", M), ("todas", ok)):
    for k in ("mm20", "cad", "cadpm1", "mm_sin_tope"):
        res[f"{nom}_{k}_cumple"] = int((D[f"{k}_emax_%"] <= 3).sum()); res[f"{nom}_{k}_peor"] = float(D[f"{k}_emax_%"].max())
        res[f"{nom}_{k}_med"] = float(D[f"{k}_emax_%"].median())
    res[f"{nom}_n"] = len(D)
json.dump(res, open(f"v58_externas_{N}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
