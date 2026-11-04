# Redondeo conjunto (Schnorr–Euchner) "desde la red": entero más cercano a c_red en la métrica AᵀA, comparado con el de c_MCO
import matplotlib; matplotlib.use('Agg')
import io, contextlib, sys, time
src = open('run.py').read(); c = src.split('# CELDA ')
CEL = {x.split(':')[0]: '# CELDA ' + x for x in c[1:]}
with contextlib.redirect_stdout(io.StringIO()):
    exec(c[0])
    for n in [str(i) for i in range(0, 12) if i not in (4, 5)]:
        exec(CEL[n])
    exec(CEL['24'].split('# --- y2, Mean HT, y1')[0])
import numpy as np, pandas as pd
idx = np.random.default_rng(SEED).choice(len(W_ESP), min(N_MUESTRA_ESPACIO, len(W_ESP)), replace=False)
fil = []
t0 = time.time()
for grupo, Ys, noms in (("41 sujetos", POB_Y, list(POB.id)), ("2000 curvas", W_ESP[idx], [f"W{i}" for i in idx]),
                        ("tres", np.array([Y2, MEDIAS["Mean HT"], Y1]), ["y2", "Mean HT", "y1"])):
    CM, CR = mco(Ys), predecir(Ys)
    for y, nm, cm, cr in zip(Ys, noms, CM, CR):
        zr, zm = redondeo_mce(cr), redondeo_mce(cm)
        er, em = errores_curva(y, zr), errores_curva(y, zm)
        fil.append(dict(grupo=grupo, curva=nm, igual=bool((zr == zm).all()), Erms_red=er[0], emax_red=er[1], Erms_mco=em[0], emax_mco=em[1],
                        **{f"zr{j+1}": zr[j] for j in range(N_BASES)}))
D = pd.DataFrame(fil); D.to_csv(f"se_desde_red_{N_BASES}.csv", index=False)
for g, d in D.groupby("grupo", sort=False):
    print(f"{N_BASES} mat | {g}: igual al conjunto desde MCO {d.igual.sum()}/{len(d)} | desde red: Erms med {d.Erms_red.median():.3f} e_max med {d.emax_red.median():.3f} peor {d.emax_red.max():.3f} "
          f"cumple3 {(d.emax_red<=3).sum()} cumple4 {(d.emax_red<=4).sum()} | desde MCO: Erms med {d.Erms_mco.median():.3f} e_max med {d.emax_mco.median():.3f} peor {d.emax_mco.max():.3f} cumple3 {(d.emax_mco<=3).sum()} cumple4 {(d.emax_mco<=4).sum()}")
print(D[D.grupo=="tres"].round(3).to_string())
print(f"{time.time()-t0:.0f} s")
