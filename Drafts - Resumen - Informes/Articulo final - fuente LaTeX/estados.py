import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc, FancyArrowPatch
T0 = 128.7; RI, RM = 3.50, 4.261; P2 = (2.34, 3.05); P3 = (2.72, 3.35)
AZ, VI, VE, GR, RO = "#1f77b4", "#9467bd", "#2ca02c", "#9e9e9e", "#d9534f"
T = {"es": dict(t1="① Sector abierto, sin tensiones\n(radios R)", t2="② Arteria cerrada, sin presión\n(radios ρ)", t3="③ Arteria cerrada, con presión (≈ 68 mmHg)\n(radios r)",
                eje="eje de simetría", hueco="hueco = 2(180° − Θ₀) = 102,6°", th="Θ₀ = 128,7°",
                n1="Referencia sin tensiones: se obtiene\ncortando la arteria a lo largo. Cada punto\nde radio R pasa a ρ en ② y a r en ③.",
                n2="pared = 0,71 mm", n3="pared = 0,63 mm",
                pie="De ② a ③ la presión empuja hacia afuera: cara interna 2,34 → 2,72 mm (+16,5 %);  borde media–adventicia 3,05 → 3,35 mm (+10 %)",
                ley="azul: cara interna · violeta: borde media–adventicia · los tres dibujos a la misma escala · sujeto NT01", esc="1 mm", dec=","),
     "en": dict(t1="① Open sector, stress-free\n(radii R)", t2="② Closed artery, no pressure\n(radii ρ)", t3="③ Closed artery, with pressure (≈ 68 mmHg)\n(radii r)",
                eje="symmetry axis", hueco="gap = 2(180° − Θ₀) = 102.6°", th="Θ₀ = 128.7°",
                n1="Stress-free reference: obtained by cutting\nthe artery lengthwise. Each point at\nradius R goes to ρ in ② and to r in ③.",
                n2="wall = 0.71 mm", n3="wall = 0.63 mm",
                pie="From ② to ③ pressure pushes outward: inner face 2.34 → 2.72 mm (+16.5 %);  media–adventitia boundary 3.05 → 3.35 mm (+10 %)",
                ley="blue: inner face · violet: media–adventitia boundary · all three drawings at the same scale · subject NT01", esc="1 mm", dec=".")}
def pol(r, a): return r * np.cos(np.radians(a)), r * np.sin(np.radians(a))
def flecha(ax, a, b, c, lw=1.8): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=c, lw=lw))
for L, t in T.items():
    f = lambda v: f"{v:.2f}".replace(".", t["dec"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 7.0))
    for ax in axs: ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-5.4, 5.4); ax.set_ylim(-6.4, 5.4)
    # --- estado 1
    ax = axs[0]; g = 180 - T0; a1, a2 = 90 + g, 90 - g + 360
    ax.add_patch(Wedge((0, 0), RM, a1, a2, width=RM - RI, fc=GR, ec="none", alpha=.6))
    ax.add_patch(Arc((0, 0), 2 * RM, 2 * RM, theta1=a1, theta2=a2, color=VI, lw=2.2)); ax.add_patch(Arc((0, 0), 2 * RI, 2 * RI, theta1=a1, theta2=a2, color=AZ, lw=2.2))
    for a in (a1, a2):
        ax.plot(*zip((0, 0), pol(RM + .25, a)), ls="--", color=VE, lw=1.2)
    ax.plot(*zip((0, 0), pol(RM + .3, 270)), ls=":", color="k", lw=1.2); ax.text(.15, -RM - .3, t["eje"], fontsize=9, ha="left", va="top")
    ax.add_patch(Arc((0, 0), 2.8, 2.8, theta1=a1, theta2=270, color=VE, lw=2.4))
    ax.text(-1.6, .35, t["th"], color=VE, fontsize=11.5, weight="bold", ha="right", va="center")
    ax.add_patch(Arc((0, 0), 1.4, 1.4, theta1=90 - g, theta2=90 + g, color="k", lw=1, ls="--")); ax.text(0, 3.0, t["hueco"], ha="center", fontsize=10)
    flecha(ax, (0, 0), pol(RI, 300), AZ); ax.text(*pol(1.9, 318), f"$R_i$ = {f(RI)}", color=AZ, fontsize=11, weight="bold", ha="center")
    flecha(ax, (0, 0), pol(RM, 345), VI); ax.text(*pol(2.6, 2), f"$R_m$ = {f(RM)}", color=VI, fontsize=11, weight="bold", ha="center")
    ax.plot(0, 0, "k+"); ax.set_title(t["t1"], fontsize=12); ax.text(0, -6.3, t["n1"], ha="center", va="bottom", fontsize=9.5, color="#444")
    # --- estados 2 y 3
    for ax, (ri, rm), tt, nota, rel in ((axs[1], P2, t["t2"], t["n2"], False), (axs[2], P3, t["t3"], t["n3"], True)):
        ax.add_patch(Wedge((0, 0), rm, 0, 360, width=rm - ri, fc=(RO if rel else GR), ec="none", alpha=(.8 if rel else .6)))
        ax.add_patch(plt.Circle((0, 0), rm, fill=False, color=VI, lw=2.2)); ax.add_patch(plt.Circle((0, 0), ri, fill=False, color=AZ, lw=2.2))
        if rel:
            for a in range(0, 360, 45):
                flecha(ax, pol(.9, a), pol(ri - .08, a), "#8b1a1a", lw=1.2)
        sim = "r" if rel else r"\rho"
        flecha(ax, (0, 0), pol(ri, 215), AZ); ax.text(*pol(1.25, 160), f"${sim}_i$ = {f(ri)}", color=AZ, fontsize=10.5, weight="bold", ha="center")
        flecha(ax, (0, 0), pol(rm, 330), VI); ax.text(*pol(rm + .9, 330), f"${sim}_m$ = {f(rm)}", color=VI, fontsize=10.5, weight="bold", ha="center")
        ax.plot(0, 0, "k+"); ax.set_title(tt, fontsize=12); ax.text(0, -4.7, nota, ha="center", fontsize=10.5)
    # barra de escala
    ax = axs[2]; ax.plot([3.4, 4.4], [-5.9, -5.9], "k", lw=3); ax.text(3.9, -5.7, t["esc"], ha="center", fontsize=9.5)
    fig.text(.5, .07, t["pie"], ha="center", fontsize=12, weight="bold"); fig.text(.5, .025, t["ley"], ha="center", fontsize=9.5, style="italic")
    fig.subplots_adjust(left=.01, right=.99, top=.9, bottom=.1, wspace=.05)
    fig.savefig(f"d10_tres_estados_{L}.png", dpi=170); plt.close(fig)
print("ok")
