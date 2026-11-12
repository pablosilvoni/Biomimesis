import json, numpy as np, pandas as pd
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
for L in ("es", "en"):
    f = (lambda x, d=2: f"{x:.{d}f}".replace(".", ",")) if L == "es" else (lambda x, d=2: f"{x:.{d}f}")
    b = lambda x: (r"\textbf{" + f(x) + "}") if x > 3 else f(x)
    for N in (6, 5):
        D = pd.read_csv(f"{S}/v59_{N}/sujetos_v59_{N}.csv")
        H = (r"Sujeto & Conj. & \multicolumn{4}{c}{Red en cadena + $\pm1$} & \multicolumn{4}{c}{Minimax (tres pasos)}\\" if L == "es"
             else r"Subject & Set & \multicolumn{4}{c}{Chained network + $\pm1$} & \multicolumn{4}{c}{Minimax (three steps)}\\")
        lam = "láminas" if L == "es" else "laminae"
        H2 = rf" & & {lam} & $\delta$ & $E_{{\mathrm{{rms}}}}$ & $e_{{\max}}$ & {lam} & $\delta$ & $E_{{\mathrm{{rms}}}}$ & $e_{{\max}}$\\"
        cmr = r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}"
        cap = (f"Nivel del tubo, curva presión--diámetro. Los 41 sujetos de Masson, {N} materiales: láminas enteras de 0,025\\,mm de cada capa ($n_1\\ldots n_{N}$, capa 1 junto a la sangre), preestiramiento $\\delta$ y errores contra la curva $\\lambda(P)$ del sujeto, $E_{{\\mathrm{{rms}}}}$ y $e_{{\\max}}$ [\\% del rango]. Conj.: E, entrenamiento; P, prueba. En negrita, $e_{{\\max}}>3$\\,\\%." if L == "es" else
               f"Tube level, pressure--diameter curve. The 41 Masson subjects, {N} materials: integer 0.025\\,mm laminae of each layer ($n_1\\ldots n_{N}$, layer 1 next to the blood), prestretch $\\delta$ and errors against the subject's $\\lambda(P)$ curve, $E_{{\\mathrm{{rms}}}}$ and $e_{{\\max}}$ [\\% of range]. Set: T, training; E, test. In bold, $e_{{\\max}}>3$\\,\\%.")
        lines = [r"{\scriptsize\setlength{\tabcolsep}{2.4pt}", r"\begin{longtable}{llcccccccc}", rf"\caption{{{cap}}}\label{{tab:tubo{N}}}\\\toprule", H, cmr, H2 + r"\midrule\endfirsthead",
                 r"\toprule " + H + cmr + H2 + r"\midrule\endhead", r"\bottomrule\endfoot"]
        for s, G in D.groupby("sujeto", sort=False):
            c = {"entrenamiento": "E" if L == "es" else "T", "prueba": "P" if L == "es" else "E"}[G.conjunto.iloc[0]]
            cel = []
            for m in ("red ±1", "minimax"):
                r = G[G["método"] == m].iloc[0]
                z = r"\enspace ".join(str(int(r[f"n{j+1}"])) for j in range(N))
                d = f(r.delta, 3); d = ("$+$" + d) if r.delta >= 0 else ("$-$" + d[1:])
                cel.append(f"{z} & {d} & {f(r['Erms %'])} & {b(r['e_max %'])}")
            lines.append(f"{s} & {c} & {cel[0]} & {cel[1]}\\\\")
        lines.append(r"\end{longtable}}")
        open(f"{S}/{'artfin2' if L == 'es' else 'arten'}/tab_tubo{N}_{L}.tex", "w").write("\n".join(lines) + "\n")
print("ok")
