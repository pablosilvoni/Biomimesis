# Carga el modelo de pared gruesa de V47 (celdas 0–4) con N materiales. Uso: g = cargar(N)
import json, os, time, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None; plt.savefig = lambda *a, **k: None
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
NB = "/mnt/user-data/uploads/Biomimesis - GitHub/Versiones viejas/Version 47 - 6 bases/curva_j_v47_6.ipynb"
def celdas():
    nb = json.load(open(NB)); return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
def cargar(N, hasta=5, E_ESCALA=None, sin_ajuste_sujetos=False):
    os.chdir(f"{S}/v59_{N}")
    cs = celdas(); g = {}
    c0 = cs[0].replace("N_BASES = 6 ", f"N_BASES = {N} ").replace('VERSION = f"v47_{N_BASES}"', 'VERSION = f"v59_{N_BASES}"')
    assert f"N_BASES = {N}" in c0
    exec(compile(c0, "c0", "exec"), g)
    for k in range(1, hasta):
        src = cs[k]
        if k == 3 and E_ESCALA is not None:
            src = src.replace("E_MAT = np.maximum(_C_SUJ.max(0), 1.0) * R_REF / H_CALIBRA",
                              "E_MAT = np.maximum(_C_SUJ.max(0), 1.0) * R_REF / H_CALIBRA * np.asarray(E_ESCALA)")
            g["E_ESCALA"] = E_ESCALA
        if k == 4 and sin_ajuste_sujetos:
            src = src[:src.index("t0 = time.perf_counter()")]
        exec(compile(src, f"c{k}", "exec"), g)
    return g

def agregar_procedimiento(g, tope=None):
    """Agrega minimax_continuo, e_max_th, a_enteros (de la Celda 8 de V47) con un tope opcional de láminas por material."""
    c8 = celdas()[8]
    a = c8.index("P_NORMAS = "); b = c8.index("def red(y):")
    src = "import itertools\nfrom scipy.optimize import least_squares, minimize_scalar\n" + c8[a:b]
    if tope is not None:
        src = src.replace("opciones = [sorted({int(np.floor(v)), int(np.ceil(v))}) for v in n0]",
                          f"opciones = [sorted({{min(int(np.floor(v)), {tope}), min(int(np.ceil(v)), {tope})}}) for v in n0]")
        src = src.replace("vecinos = [v for v in vecinos if v.min() >= 0]", f"vecinos = [v for v in vecinos if v.min() >= 0 and v.max() <= {tope}]")
        assert src.count(str(tope)) >= 2
    exec(compile(src, "c8", "exec"), g)
    def procedimiento(y, th0=None):
        th_mc = g["ajuste_exacto"](y, th0=th0)
        th, n = g["a_enteros"](g["curva_injerto"](th_mc), th_mc)
        return th, n, th_mc
    g["procedimiento"] = procedimiento
    return g
