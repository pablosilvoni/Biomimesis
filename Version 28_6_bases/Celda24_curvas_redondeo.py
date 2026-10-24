# CELDA 24: y2, Mean HT e y1 — curva original, MCO, red y los tres redondeos a láminas enteras (requiere las Celdas 0 a 11; no hace falta correr de la 12 a la 23)
#   Tabla: láminas de cada material, total y error punto a punto máximo de cada caso; se guarda en CSV y, si está el CSV
#   del otro número de bases, se imprime la comparación de errores 5 vs 6 bases.
#   Izquierda: las curvas. Derecha: el residuo de cada aproximación (aproximación − original), en % del rango de la curva.
#   Escala de láminas: C_LAM de la Celda 0 (o CLAM_FIG si se quiere otra, sin reentrenar).
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds


def redondeo_conjunto(c, B):
    """argmin_{z entero ≥ 0} ||B(z − c)||²: la curva con láminas enteras más cercana a la de láminas reales."""
    M = B.T @ B
    R = np.linalg.cholesky(M).T
    z0 = np.maximum(np.round(c), 0)
    mejor = [float((z0 - c) @ M @ (z0 - c)), z0.copy()]
    z = np.zeros(len(c))

    def nivel(k, parcial):
        s = c[k] - (R[k, k + 1:] @ (z[k + 1:] - c[k + 1:])) / R[k, k]
        radio = np.sqrt(max(mejor[0] - parcial, 0.0)) / abs(R[k, k])
        for v in sorted(range(max(0, int(np.ceil(s - radio))), int(np.floor(s + radio)) + 1), key=lambda v: abs(v - s)):
            z[k] = v
            p = parcial + (R[k, k] * (v - s)) ** 2
            if p < mejor[0]:
                if k == 0:
                    mejor[0], mejor[1] = p, z.copy()
                else:
                    nivel(k - 1, p)
    nivel(len(c) - 1, 0.0)
    return mejor[1]


def redondeo_minimax(y, B):
    """argmin_{z entero ≥ 0} max|y − Bz| (en 500 puntos de la grilla)."""
    Bs, ys = B[::2], y[::2]
    m, n = Bs.shape
    r = milp(np.r_[np.zeros(n), 1.0], integrality=np.r_[np.ones(n), 0], bounds=Bounds(np.zeros(n + 1), np.inf),
             constraints=LinearConstraint(np.vstack([np.c_[Bs, -np.ones(m)], np.c_[-Bs, -np.ones(m)]]), -np.inf, np.r_[ys, -ys]),
             options=dict(time_limit=60))
    return None if r.x is None else np.round(r.x[:n])


_err = lambda y, B, z: np.abs(y - B @ z).max() / np.ptp(y) * 100



CLAM_FIG = C_LAM                                             # escala de láminas de la figura (un número o una lista)
_cl = np.resize(np.asarray(CLAM_FIG, float), N_BASES)
_B = G * (PEND_MAX / _cl)                                     # bases con esa escala
_CURVAS = {"y2 (límite superior)": Y2, "Mean HT": MEDIAS["Mean HT"], "y1 (límite inferior)": Y1}
_EST = {"Original": ("k", "-", 2.6), "MCO": ("tab:blue", ":", 2.0), "Red": ("tab:red", "--", 1.6),
        "Redondeo individual": ("tab:orange", "-", 1.2), "Redondeo conjunto": ("tab:green", "-.", 1.4),
        "Redondeo minimax": ("tab:purple", "-", 1.6)}

_filas = []
fig, axs = plt.subplots(3, 2, figsize=(14, 13), gridspec_kw=dict(width_ratios=[1.25, 1]))
for (nom, y), (ax, axr) in zip(_CURVAS.items(), axs):
    c_mco = mco(y, _B)
    c_red = predecir(y) / ESC * _cl / C_LAM_RED               # láminas reales de la red en la escala de la figura
    c_min = redondeo_minimax(y, _B)
    Z = {"MCO": c_mco, "Red": c_red, "Redondeo individual": np.maximum(np.round(c_red), 0),
         "Redondeo conjunto": redondeo_conjunto(c_red, _B), "Redondeo minimax": redondeo_conjunto(c_red, _B) if c_min is None else c_min}
    ax.plot(X, y, color="k", ls="-", lw=_EST["Original"][2], label="Original")
    for k, z in Z.items():
        col, ls, lw = _EST[k]
        e = _err(y, _B, z)
        ax.plot(X, _B @ z, color=col, ls=ls, lw=lw, label=f"{k} ({e:.2f} %)")
        axr.plot(X, (_B @ z - y) / np.ptp(y) * 100, color=col, ls=ls, lw=lw, label=k)
        _filas.append({"curva": nom, "método": k, **{f"mat. {j + 1}": v for j, v in enumerate(z)}, "total": z.sum(), "error máx. %": e})
    for a in (ax, axr):
        a.grid(alpha=0.3)
    ax.set(title=f"{nom}", xlabel=ETQ_X, ylabel=ETQ_Y)
    ax.legend(fontsize=8, loc="upper left")
    axr.axhspan(-ERROR_MAX_PCT, ERROR_MAX_PCT, color="tab:green", alpha=0.07)
    axr.axhline(0, color="k", lw=1)
    axr.set(title=f"{nom}: residuo (banda verde: ±{ERROR_MAX_PCT:g} %)", xlabel=ETQ_X, ylabel="aproximación − original [% del rango]")
    axr.legend(fontsize=8, loc="best")
fig.suptitle(f"{N_BASES} bases, C_LAM = {np.unique(_cl).tolist()}: original, MCO, red y láminas enteras (entre paréntesis: error punto a punto máximo)",
             fontsize=12)
fig.tight_layout()
plt.savefig(f"curvas_redondeo_{VERSION}.png", dpi=300, bbox_inches="tight")
plt.show()

TABLA_CURVAS = pd.DataFrame(_filas).set_index(["curva", "método"])
pd.set_option("display.width", 250)
print(f"Láminas por material (C_LAM = {np.unique(_cl).tolist()}), total y error punto a punto máximo [% del rango]:")
print(TABLA_CURVAS.round(2).to_string())
TABLA_CURVAS.to_csv(f"tabla_curvas_redondeo_{VERSION}.csv")
_otros = {v: f"tabla_curvas_redondeo_v28_{v}.csv" for v in (5, 6)}
if all(os.path.exists(f) for f in _otros.values()):
    _cmp = pd.concat({f"{v} bases": pd.read_csv(f, index_col=[0, 1])["error máx. %"] for v, f in _otros.items()}, axis=1)
    print("\nError punto a punto máximo [%], 5 vs 6 bases:")
    print(_cmp.round(2).to_string())
