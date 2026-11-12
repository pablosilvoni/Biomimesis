import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
NS = [int(a) for a in sys.argv[1:]] or [6, 5]
T = {"es": dict(mm="minimax (tres pasos)", red="red en cadena", pm="red + $\\pm1$", ylab="peor error $e_{\\max}$ [% del rango]",
                tit=lambda n: f"{n} materiales", meta="meta 3 %", lam="estiramiento $\\lambda$ del borde media–adventicia",
                pk="presión que soporta la pared [kPa]", ph="[mmHg]", art="arteria",
                nom={"NT04": "NT04 (el más blando)", "HT20": "HT20 (intermedio)", "HT09": "HT09 (el más rígido)"}),
     "en": dict(mm="minimax (three steps)", red="chained network", pm="network + $\\pm1$", ylab="worst error $e_{\\max}$ [% of range]",
                tit=lambda n: f"{n} materials", meta="3 % target", lam="media–adventitia stretch $\\lambda$",
                pk="pressure carried by the wall [kPa]", ph="[mmHg]", art="artery",
                nom={"NT04": "NT04 (softest)", "HT20": "HT20 (intermediate)", "HT09": "HT09 (stiffest)"})}
for L, t in T.items():
    fig, ax = plt.subplots(len(NS), 1, figsize=(12, 3.3 * len(NS)), squeeze=False)
    for a, N in zip(ax[:, 0], NS):
        D = pd.read_csv(f"{S}/v59_{N}/sujetos_v59_{N}.csv")
        P = D.pivot(index="sujeto", columns="método", values="e_max %"); conj = D.groupby("sujeto").conjunto.first()
        P = P.sort_values("minimax"); x = np.arange(len(P))
        a.plot(x, P["minimax"], "o", mfc="none", color="#1f4e8c", label=t["mm"], ms=7)
        a.plot(x, P["red"], "x", color="#c0392b", label=t["red"], ms=6)
        a.plot(x, P["red ±1"], "s", mfc="none", color="#27864a", label=t["pm"], ms=6)
        a.axhline(3, color="r", lw=1); a.text(len(P) - 0.5, 3.05, t["meta"], color="r", ha="right", va="bottom", fontsize=9)
        a.set_xticks(x); a.set_xticklabels(P.index, rotation=90, fontsize=8)
        for lab, s in zip(a.get_xticklabels(), P.index):
            if conj[s] == "prueba": lab.set_fontweight("bold")
        a.set_yscale("log"); a.set_ylabel(t["ylab"]); a.set_title(t["tit"](N)); a.grid(alpha=.3, which="both"); a.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{S}/v59/fig/v59_emax_{L}.png", dpi=170); plt.close(fig)
    # ejemplos, 6 materiales
    C = np.load(f"{S}/v59/curvas_ej_6.npz"); Pk = C["P"]; MM = 0.1333224
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    D = pd.read_csv(f"{S}/v59_6/sujetos_v59_6.csv")
    for a, s in zip(ax, ("NT04", "HT20", "HT09")):
        a.plot(C[f"{s}_y"], Pk, "k", lw=2.4, label=t["art"])
        for m, c, ls, lab in (("minimax", "#1f4e8c", "--", t["mm"]), ("red ±1", "#27864a", ":", t["pm"])):
            r = D[(D.sujeto == s) & (D["método"] == m)].iloc[0]
            a.plot(C[f"{s}_{m}"], Pk, color=c, ls=ls, lw=2, label=f"{lab}: {r['láminas']}, $e_{{\\max}}$ {r['e_max %']:.2f} %".replace(".", "," if L == "es" else "."))
        a.set_title(t["nom"][s]); a.set_xlabel(t["lam"]); a.grid(alpha=.3); a.legend(fontsize=7.5, loc="upper left")
        sec = a.secondary_yaxis("right", functions=(lambda p: p / MM, lambda h: h * MM))
        if s == "HT09": sec.set_ylabel(t["ph"])
    ax[0].set_ylabel(t["pk"])
    fig.tight_layout(); fig.savefig(f"{S}/v59/fig/v59_disenos_{L}.png", dpi=170); plt.close(fig)
print("ok")
