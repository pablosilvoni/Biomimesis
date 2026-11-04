# V50 · ¿Cuánto le falta al paso 3 (A3, minimax entero heurístico) para llegar al óptimo entero?
#   Para cada sujeto (curva limpia, camino A): paso 2 → θ*; paso 3 (A3) → diseño entero.
#   Búsqueda exhaustiva: todas las combinaciones n_j ∈ round(h*_j/e) ± 2 (n ≥ 0), con δ en una grilla de 13 valores (δ* ± 0,03)
#   y Brent sobre δ para las 200 mejores (el diseño del A3 también es candidato). Mismo objetivo que el A3: e_max contra la curva ajustada del paso 2; además, contra la original.
import matplotlib; matplotlib.use('Agg')
import io, contextlib, time, itertools, sys
_src = open('run.py').read(); _c = _src.split('# CELDA ')
_CEL = {x.split(':')[0]: '# CELDA ' + x for x in _c[1:]}
with contextlib.redirect_stdout(io.StringIO()):
    exec(_c[0]); exec(_CEL['0']); exec(_CEL['1'].split('# 1) Escala absoluta')[0]); exec(_CEL['3'])
    exec(_CEL['4'].split('t0 = time.perf_counter()')[0])
    exec(_CEL['8'].split('def red(y)')[0])
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
CAJA, NDG, NBEST = 2, 13, 200
esp = ESP_LAMINA_MM
fil = []
t00 = time.time()
for nm, y in zip(POB.id, POB_L):
    t0 = time.time()
    th_mc = ajuste_exacto(y, th0=arranque_delgada(y))
    yt = curva_injerto(th_mc)
    th_a3, n_a3 = a_enteros(yt, th_mc)
    e_a3 = e_max_th(yt, th_a3); e_a3_o = e_max_th(y, th_a3)
    base = np.round(np.maximum(th_mc[:-1], 0) / esp).astype(int)
    rangos = [range(max(0, b - CAJA), b + CAJA + 1) for b in base]
    dg = np.clip(th_mc[-1] + np.linspace(-0.03, 0.03, NDG), -DELTA_MAX, DELTA_MAX)
    res = []
    for n in itertools.product(*rangos):
        h = np.asarray(n, float) * esp
        ev = [e_max_th(yt, np.r_[h, d]) for d in dg]
        k = int(np.argmin(ev)); res.append((ev[k], dg[k], n))
    res.sort(key=lambda r: r[0])
    best = (e_a3, th_a3[-1], n_a3.astype(float))          # el diseño del A3 es un candidato más (la exhaustiva nunca queda peor)
    for e0, d0, n in res[:NBEST]:
        h = np.asarray(n, float) * esp
        r = minimize_scalar(lambda d: e_max_th(yt, np.r_[h, d]), bounds=(max(d0 - 0.006, -DELTA_MAX), min(d0 + 0.006, DELTA_MAX)),
                            method="bounded", options=dict(xatol=1e-5))
        e = min(r.fun, e0); d = r.x if r.fun <= e0 else d0
        if e < best[0]: best = (e, d, np.asarray(n))
    e_ex, d_ex, n_ex = best
    th_ex = np.r_[n_ex * esp, d_ex]
    dentro = bool(np.all(np.abs(n_a3 - base) <= CAJA))
    fil.append(dict(sujeto=nm, combinaciones=len(res), e_A3_ajustada=e_a3, e_exh_ajustada=e_ex, mejora=e_a3 - e_ex,
                    e_A3_original=e_a3_o, e_exh_original=e_max_th(y, th_ex), igual=bool(np.array_equal(n_a3, n_ex)), A3_dentro_caja=dentro,
                    n_A3=" ".join(map(str, n_a3)), n_exh=" ".join(map(str, n_ex.astype(int))), delta_A3=th_a3[-1], delta_exh=d_ex, seg=time.time() - t0))
    print(f"{nm}: A3 {e_a3:.3f} | exhaustiva {e_ex:.3f} | igual {fil[-1]['igual']} | {time.time()-t0:.0f} s", flush=True)
D = pd.DataFrame(fil); D.to_csv(f"v50_gap_{N_BASES}.csv", index=False)
print(f"\n{N_BASES} materiales, caja ±{CAJA} láminas, {(time.time()-t00)/60:.1f} min")
print(f"mismo diseño entero: {D.igual.sum()}/41 | la exhaustiva mejora el peor punto (contra la ajustada) en {(D.mejora>1e-3).sum()} sujetos; "
      f"mejora mediana {D.mejora.median():.3f}, máxima {D.mejora.max():.3f} puntos")
print(f"contra la original: A3 mediana {D.e_A3_original.median():.3f} peor {D.e_A3_original.max():.3f} cumple {(D.e_A3_original<=3).sum()}/41 | "
      f"exhaustiva mediana {D.e_exh_original.median():.3f} peor {D.e_exh_original.max():.3f} cumple {(D.e_exh_original<=3).sum()}/41")
