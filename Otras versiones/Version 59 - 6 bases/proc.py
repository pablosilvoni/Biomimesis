import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
T = {"es": dict(cur="Curva del paciente\n$\\lambda$ en 141 presiones\n(limpia o medida)",
               p1="Paso 1\narranque de\npared delgada\n(lineal)", p2="Paso 2\nmínimos cuadrados\nen pared gruesa\n(espesores reales)",
               p3="Paso 3\nláminas enteras\npor minimax\ncontra la curva\ndel paso 2",
               red="Red en cadena\nláminas $n_1\\ldots n_N$ y $\\delta$\nen una evaluación",
               opc="Opcional\nafinar $\\delta$ (una variable)\ny revisar $\\pm1$ lámina",
               fin="Diseño que se fabrica\n$n_1\\ldots n_N$ (0 a 20 láminas\nde 0,025 mm) y $\\delta$",
               A="Referencia: minimax (tres pasos)", B="Red neuronal"),
     "en": dict(cur="Patient curve\n$\\lambda$ at 141 pressures\n(clean or measured)",
               p1="Step 1\nthin-wall\nstart\n(linear)", p2="Step 2\nthick-wall\nleast squares\n(real thicknesses)",
               p3="Step 3\ninteger laminae\nby minimax\nagainst the\nstep-2 curve",
               red="Chained network\nlaminae $n_1\\ldots n_N$ and $\\delta$\nin one evaluation",
               opc="Optional\nrefine $\\delta$ (one variable)\nand check $\\pm1$ lamina",
               fin="Design to manufacture\n$n_1\\ldots n_N$ (0 to 20 laminae\nof 0.025 mm) and $\\delta$",
               A="Reference: minimax (three steps)", B="Neural network")}
def caja(ax, x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="#333", lw=1.2))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=10.5)
def flecha(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#222"))
for L, t in T.items():
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    caja(ax, 0.1, 1.7, 2.0, 1.2, t["cur"], "#f2f3f5")
    caja(ax, 2.8, 2.75, 2.0, 1.55, t["p1"], "#d7e8f6"); caja(ax, 5.3, 2.75, 2.2, 1.55, t["p2"], "#fdf6e3"); caja(ax, 8.0, 2.75, 2.2, 1.55, t["p3"], "#e6f4ea")
    caja(ax, 2.8, 0.45, 3.4, 1.15, t["red"], "#f8d9d6"); caja(ax, 6.8, 0.45, 3.4, 1.15, t["opc"], "#fbe9e7")
    caja(ax, 10.8, 1.65, 2.1, 1.3, t["fin"], "#eeeeee")
    ax.text(2.8, 4.42, t["A"], fontsize=11, weight="bold", color="#2b5d8a"); ax.text(2.8, 1.75, t["B"], fontsize=11, weight="bold", color="#a33")
    flecha(ax, (2.1, 2.6), (2.8, 3.4)); flecha(ax, (2.1, 2.0), (2.8, 1.1))
    flecha(ax, (4.8, 3.52), (5.3, 3.52)); flecha(ax, (7.5, 3.52), (8.0, 3.52)); flecha(ax, (6.2, 1.02), (6.8, 1.02))
    flecha(ax, (10.2, 3.4), (10.8, 2.7)); flecha(ax, (10.2, 1.1), (10.8, 1.9))
    fig.savefig(f"f_procedimiento_v59_{L}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
