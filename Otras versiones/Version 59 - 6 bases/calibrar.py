import sys, numpy as np, base, time
N = int(sys.argv[1]); esc = None if len(sys.argv) < 3 else [float(v) for v in sys.argv[2].split(",")]
tope = None if len(sys.argv) < 4 else int(sys.argv[3])
g = base.cargar(N, hasta=5, E_ESCALA=esc, sin_ajuste_sujetos=True)
if tope is not None:
    g["HI"] = np.r_[np.full(N, tope * g["ESP_LAMINA_MM"]), g["DELTA_MAX"]]
g = base.agregar_procedimiento(g, tope)
t = time.time(); R = []
for nm, y in zip(g["POB"].id, g["POB_L"]):
    th, n, _ = g["procedimiento"](y); R.append((nm, g["e_max_th"](y, th), n))
E = np.array([r[1] for r in R]); Nn = np.array([r[2] for r in R])
print("E_MAT", g["E_MAT"].round(0), "| t/sujeto", round((time.time() - t) / 41, 2), "s")
print("cumplen", (E <= 3).sum(), "mediana", round(np.median(E), 2), "peor", round(E.max(), 2), "| test:", np.round(E[g["MASK_TEST"]].max(), 2))
print("max laminas por material", Nn.max(0), "| total mediana", np.median(Nn.sum(1)), "| sujetos >20:", (Nn.max(1) > 20).sum())
print("f sugerido", np.round(np.maximum(1, Nn.max(0) / 18), 3).tolist())
