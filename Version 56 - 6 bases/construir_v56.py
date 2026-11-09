# V56: la red en cadena (tope 20 láminas) se entrena SOLO con curvas dentro de la banda y1–y2 de Masson.
#   Parte del notebook V55 de N materiales; las etiquetas del minimax (50 000 curvas) se reutilizan y se filtran en la Celda 8.
import json, sys, shutil, os
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
N = int(sys.argv[1]); d5, d6 = f"{S}/v55_{N}", f"{S}/v56_{N}"
os.makedirs(d6, exist_ok=True)
nb = json.load(open(f"{d5}/curva_j_v55_{N}.ipynb"))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
def src(c): return ''.join(c['source'])
def put(c, s): c['source'] = s; c['outputs'] = []; c['execution_count'] = None
c0 = src(code[0])
c0 = c0.replace('VERSION = f"v55_{N_BASES}"', 'VERSION = f"v56_{N_BASES}"')
c0 = c0.replace('ARCHIVO_ETIQUETAS_ENTERAS = f"etiquetas_enteras_v55_{N_BASES}.npz"', 'ARCHIVO_ETIQUETAS_ENTERAS = f"etiquetas_enteras_v56_{N_BASES}.npz"   # las 50 000 curvas generadas (la Celda 8 deja las de la banda)')
c0 = c0.replace("# V55: una sola red, la red en cadena", "# una sola red: la red en cadena, entrenada dentro de y1–y2")
assert 'v56_' in c0 and 'etiquetas_enteras_v56' in c0
put(code[0], c0)
c8 = src(code[8])
FILTRO = '''# V56: LA RED SE ENTRENA SOLO DENTRO DE LO QUE PUEDE REPRODUCIR — curvas (sin ruido) dentro de la banda y1–y2 de Masson.
#   Fuera de la banda quedan casi todas las "puras" (un solo material) y la mitad de las "combinadas"; las fisiológicas quedan todas
#   salvo algunas exponenciales. Las etiquetas (minimax, tope Z_MAX) se calcularon para las 50 000 y aquí se filtran.
EN_BANDA = dentro_espacio(Y_limpio)
Y_raw, Y_limpio, familia, Z_ENT, C_RED, C_RED_MCO, MASCARA, X_log = (a[EN_BANDA] for a in
                                                                    (Y_raw, Y_limpio, familia, Z_ENT, C_RED, C_RED_MCO, MASCARA, X_log))
N_GENERADAS, N_SAMPLES = len(EN_BANDA), int(EN_BANDA.sum())
print(f"✅ Dentro de y1–y2: {N_SAMPLES} de {N_GENERADAS} curvas | " +
      ", ".join(f"{FAMILIAS[f]} {int((familia == f).sum())}" for f in FAMILIAS))
'''
i = c8.index("idx_trval, idx_test")
c8 = c8[:i] + FILTRO + c8[i:]
put(code[8], c8)
for c in code[9:]:
    s = src(c)
    if "v55" in s: print("aviso: v55 en celda", s[:60])
    put(c, s)
json.dump(nb, open(f"{d6}/curva_j_v56_{N}.ipynb", "w"), ensure_ascii=False, indent=1)
for f in os.listdir(d5):
    if f.endswith(".csv") and ("masson" in f or f.startswith("bases_v40") or f.startswith("plantilla")):
        shutil.copy(f"{d5}/{f}", d6)
shutil.copy(f"{d5}/etiquetas_enteras_v55_{N}.npz", f"{d6}/etiquetas_enteras_v56_{N}.npz")
print("ok", os.listdir(d6))
