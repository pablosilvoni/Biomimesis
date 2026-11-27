# V59: la RED EN CADENA ORDINAL llevada al tubo de pared gruesa. Arma curva_j_v59_N.ipynb a partir del código de V47.
#   Uso: python3 construir_v59.py N
import json, sys, os
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
N = int(sys.argv[1])
NB47 = "/mnt/user-data/uploads/Biomimesis - GitHub/Versiones viejas/Version 47 - 6 bases/curva_j_v47_6.ipynb"
cs = [''.join(c['source']) for c in json.load(open(NB47))['cells'] if c['cell_type'] == 'code']

INTRO = """# V59 — Una sola red también para el tubo: la red en cadena ordinal da las láminas enteras de cada capa

**Qué cambia respecto de V47.** En V47 la red de pared gruesa era un perceptrón común que daba espesores reales y servía solo como
punto de partida. En V59 se usa **la misma red del primer nivel (Figura 3 del artículo): la red en cadena con salida ordinal**.
Recibe la curva presión–diámetro λ(P) del paciente y da **directamente las láminas enteras de cada capa** (0 a 20) y el
preestiramiento δ. Después se puede afinar δ (una sola variable) y revisar ±1 lámina por material.

**Referencia.** El procedimiento de 3 pasos de V47 (arranque de pared delgada → mínimos cuadrados no lineales de pared gruesa →
minimax entero), que se llama aquí *minimax* para abreviar.

**Tope de 20 láminas por material** (criterio de fabricación). Para respetarlo con láminas de 0,025 mm, la rigidez de los
materiales más rígidos se recalibró: con 6 materiales, ×1,111, ×1,278 y ×1,5 para los materiales 4, 5 y 6; con 5, ×1,5 el 5.
"""

c0 = cs[0].replace("N_BASES = 6 ", f"N_BASES = {N} ").replace('VERSION = f"v47_{N_BASES}"', 'VERSION = f"v59_{N_BASES}"')
assert f"N_BASES = {N}" in c0
c0 = c0.replace("# CELDA 0: CONFIGURACIÓN (ejecutar siempre primero)\n",
                "# CELDA 0: CONFIGURACIÓN (ejecutar siempre primero)\n"
                "#   V59 = la red en cadena ORDINAL del primer nivel, llevada al tubo de pared gruesa: da directamente las láminas enteras\n"
                "#   de cada capa (0 a 20) y el preestiramiento δ. Se compara contra el procedimiento de 3 pasos de V47 (el «minimax»).\n")
c0 += '''
# --- V59 ---------------------------------------------------------------------------------------------------------------
TOPE_LAMINAS = 20                # a lo sumo 20 láminas por material (criterio de fabricación)
E_ESCALA = {6: [1, 1, 1, 1.111, 1.278, 1.5], 5: [1, 1, 1, 1, 1.5]}[N_BASES]   # recalibración de la rigidez para respetar el tope
ARCHIVO_MODELO = f"modelo_rn_cadena_tubo_{VERSION}.weights.h5"
ARCHIVO_SCALERS = f"scalers_cadena_tubo_{VERSION}.pkl"
ARCHIVO_HISTORIAL = f"historial_entrenamiento_cadena_tubo_{VERSION}.csv"
ARCHIVO_ETIQUETAS = f"etiquetas_enteras_tubo_{VERSION}.npz"
ENTRENAR = not all(os.path.exists(f) for f in (ARCHIVO_MODELO, ARCHIVO_SCALERS, ARCHIVO_HISTORIAL))
print(f"   V59: red en cadena ordinal para el tubo | tope {TOPE_LAMINAS} láminas por material | rigidez ×{E_ESCALA} | "
      f"red: {'se entrena' if ENTRENAR else 'se carga'}")
'''
c3 = cs[3].replace("E_MAT = np.maximum(_C_SUJ.max(0), 1.0) * R_REF / H_CALIBRA",
                   "E_MAT = np.maximum(_C_SUJ.max(0), 1.0) * R_REF / H_CALIBRA * np.asarray(E_ESCALA)   # V59: recalibrada (tope de 20 láminas)")
assert "E_ESCALA" in c3
c4 = cs[4].replace("HI = np.r_[np.full(N_BASES, H_CAPA_MAX), DELTA_MAX]",
                   "HI = np.r_[np.full(N_BASES, min(H_CAPA_MAX, TOPE_LAMINAS * ESP_LAMINA_MM)), DELTA_MAX]   # V59: tope de láminas")
assert "TOPE_LAMINAS" in c4

c8 = cs[8]; a = c8.index("P_NORMAS = "); b = c8.index("def red(y):")
proc = c8[a:b]
proc = proc.replace("opciones = [sorted({int(np.floor(v)), int(np.ceil(v))}) for v in n0]",
                    "opciones = [sorted({min(int(np.floor(v)), TOPE_LAMINAS), min(int(np.ceil(v)), TOPE_LAMINAS)}) for v in n0]")
proc = proc.replace("vecinos = [v for v in vecinos if v.min() >= 0]", "vecinos = [v for v in vecinos if v.min() >= 0 and v.max() <= TOPE_LAMINAS]")
assert proc.count("TOPE_LAMINAS") == 3
C5 = '''# CELDA 5: LA REFERENCIA — el procedimiento de 3 pasos de V47 (el «minimax» del nivel tubo), con el tope de 20 láminas
#   Paso 1: arranque de pared delgada · Paso 2: mínimos cuadrados no lineales de pared gruesa (reales) ·
#   Paso 3: láminas enteras con minimax no lineal contra la curva del paso 2 (piso/techo, δ reajustado y mejora local).
import time
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

''' + proc + '''

def errores_l(y, l):
    """(Erms %, e_max %) de la curva l contra y, en % del rango de y."""
    r = 100.0 * (np.asarray(l) - y) / np.ptp(y)
    return float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def minimax_tubo(y):
    """Diseño entero de referencia: (θ = (h_1…h_N, δ), n láminas)."""
    th_mc = ajuste_exacto(y)
    return a_enteros(curva_injerto(th_mc), th_mc)


t0 = time.perf_counter()
REF_SUJ = [minimax_tubo(y) for y in POB_L]
T_MINIMAX = (time.perf_counter() - t0) / len(POB_L)
E_REF = np.array([e_max_th(y, th) for y, (th, n) in zip(POB_L, REF_SUJ)])
N_REF = np.array([n for th, n in REF_SUJ])
print(f"✅ Referencia (minimax del tubo) en los 41 sujetos: {(E_REF <= ERROR_MAX_PCT).sum()}/41 cumplen | mediana {np.median(E_REF):.2f} % | "
      f"peor {E_REF.max():.2f} % | láminas por material: máx. {N_REF.max()} | {T_MINIMAX:.2f} s por sujeto")
'''

C6 = '''# CELDA 6: DATASET CON ETIQUETAS ENTERAS (láminas n_1…n_N de 0 a 20 y preestiramiento δ)
#   Familia 1 (injertos): se sortean las láminas y δ, se calcula su curva → la etiqueta es exactamente ese diseño.
#   Familias 2, 3, 5 y 6 (arterias): la etiqueta es el diseño entero de referencia (Celda 5). Es lento (≈ 0,4 s por curva):
#   se calcula en paralelo, se guarda en ARCHIVO_ETIQUETAS y las veces siguientes se carga.
import os
import time
import multiprocessing as mp
import numpy as np

N_INJERTOS = 6000
rng_b = np.random.default_rng(2028)
N_GEN_MAX = np.minimum(np.ceil(1.3 * N_REF[MASK_TRAIN].max(0)), TOPE_LAMINAS).astype(int)
D_GEN = (min(th[-1] for th, n in REF_SUJ) - 0.03, max(th[-1] for th, n in REF_SUJ) + 0.03)


def _sortear_injerto():
    n = np.zeros(N_BASES, int)
    act = rng_b.choice(N_BASES, size=rng_b.integers(2, N_BASES + 1), replace=False)
    n[act] = rng_b.integers(0, N_GEN_MAX[act] + 1)
    return n, rng_b.uniform(*D_GEN)


def _etiqueta(y):
    th, n = minimax_tubo(y)
    return np.r_[n, th[-1]]


if os.path.exists(ARCHIVO_ETIQUETAS):
    _d = np.load(ARCHIVO_ETIQUETAS)
    N_INJ, D_INJ, Y_INJ, ET_ART = _d["N_INJ"], _d["D_INJ"], _d["Y_INJ"], _d["ET_ART"]
    print(f"📦 Etiquetas cargadas de '{ARCHIVO_ETIQUETAS}'")
else:
    t0 = time.time()
    N_INJ, D_INJ, Y_INJ = [], [], []
    while len(N_INJ) < N_INJERTOS:                                     # injertos que cubren la ventana con forma razonable
        n, d = _sortear_injerto()
        l = curva_injerto(np.r_[n * ESP_LAMINA_MM, d])
        if X_INJ[0] + 0.01 < l[0] and l[-1] < X_INJ[-1] - 0.01 and np.ptp(l) > 0.02:
            N_INJ.append(n); D_INJ.append(d); Y_INJ.append(l)
    N_INJ, D_INJ, Y_INJ = np.array(N_INJ), np.array(D_INJ), np.array(Y_INJ)
    _Y = np.vstack([LIMPIAS[f] for f in (2, 3, 5)])
    print(f"⏳ Diseño entero de referencia de {len(_Y)} arterias sintéticas en paralelo...", flush=True)
    with mp.get_context("fork").Pool(max(1, os.cpu_count())) as pool:
        ET_ART = np.array(pool.map(_etiqueta, list(_Y), chunksize=50))
    ET_ART = np.vstack([ET_ART, np.array([np.r_[n, th[-1]] for th, n in np.array(REF_SUJ, dtype=object)[MASK_TRAIN]])])
    np.savez_compressed(ARCHIVO_ETIQUETAS, N_INJ=N_INJ, D_INJ=D_INJ, Y_INJ=Y_INJ, ET_ART=ET_ART)
    print(f"✅ Etiquetas en {(time.time() - t0) / 60:.1f} min → {ARCHIVO_ETIQUETAS}")

_n6 = MASK_TRAIN.sum()
_rep = np.resize(np.arange(_n6), N_FAM[6])
Y_ALL = np.vstack([Y_INJ] + [LIMPIAS[f] for f in (2, 3, 5)] + [POB_L[MASK_TRAIN][_rep]])
ET_ALL = np.vstack([np.c_[N_INJ, D_INJ], ET_ART[:-_n6], ET_ART[-_n6:][_rep]])
Z = ET_ALL[:, :N_BASES].astype(int)                                     # láminas enteras (etiqueta)
DELTA = ET_ALL[:, -1]                                                     # preestiramiento (etiqueta)
familia = np.concatenate([np.full(len(Y_INJ), 1)] + [np.full(N_FAM[f], f) for f in (2, 3, 5, 6)]).astype(np.int8)
assert Z.max() <= TOPE_LAMINAS and Z.min() >= 0
print(f"✅ Dataset: {len(Y_ALL)} curvas ({len(Y_INJ)} injertos + {len(Y_ALL) - len(Y_INJ)} arterias) | láminas por material ≤ {Z.max()} | "
      f"δ de {DELTA.min():.3f} a {DELTA.max():.3f}")
'''

C7 = '''# CELDA 7: PARTICIÓN TRAIN / VAL / TEST Y NORMALIZACIÓN
#   Entrada: la curva λθ(P) en los 141 valores de presión, estandarizada. Salidas: láminas enteras (ordinal) y δ estandarizado.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

idx_trval, idx_test = train_test_split(np.arange(len(Y_ALL)), test_size=0.15, random_state=SEED, stratify=familia)
idx_train, idx_val = train_test_split(idx_trval, test_size=0.12, random_state=SEED, stratify=familia[idx_trval])
sx = StandardScaler().fit(Y_ALL[idx_train])
D_MED, D_SD = float(DELTA[idx_train].mean()), float(DELTA[idx_train].std())
X_train, X_val, X_test = (sx.transform(Y_ALL[i]).astype(np.float32) for i in (idx_train, idx_val, idx_test))
K = TOPE_LAMINAS + 1
print(f"✅ Train {len(idx_train)} | Val {len(idx_val)} | Test {len(idx_test)} | {K} cantidades posibles por material (0 a {TOPE_LAMINAS})")
'''

C8 = '''# CELDA 8: LA RED EN CADENA ORDINAL PARA EL TUBO (la misma de la Figura 3 del artículo)
#   Tronco común 141 → 256 → 128 → 64 (ReLU + LayerNormalization) → h. Cabeza j: (h, m_1…m_{j−1}) → 128 → 64 → un puntaje s_j;
#   20 preguntas «¿más de k láminas?» con P(n_j > k) = σ(s_j + b_jk) (CORAL). m = vector de láminas decididas (n/20).
#   Una cabeza más para δ: (h, m_1…m_N) → 64 → δ (estandarizado). Pérdida: entropía cruzada binaria de las preguntas
#   (sumada sobre preguntas y materiales) + error cuadrático de δ. Teacher forcing: al entrenar, m son las láminas de la etiqueta.
#   En cada época se suma a la entrada un ruido nuevo (0 a 1 % del rango de λ de cada curva).
import os, time, pickle
import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(SEED)
EPOCHS, BATCH_SIZE, RUIDO_MAX, PESO_DELTA = 800, 256, 0.01, 5.0
ESCZ = float(TOPE_LAMINAS)
_sxm, _sxs = tf.constant(sx.mean_, tf.float32), tf.constant(sx.scale_, tf.float32)


class Umbrales(tf.keras.layers.Layer):
    """CORAL: logit_k = s + b_k (k = 0…K−2), el MISMO puntaje s y un umbral b_k por pregunta «¿más de k láminas?»."""
    def __init__(self, n, **kw):
        super().__init__(**kw); self.n = n
    def build(self, shape):
        self.b = self.add_weight(name="umbrales", shape=(self.n,), initializer=tf.keras.initializers.Constant(np.linspace(2.0, -2.0, self.n)))
    def call(self, s):
        return s + self.b


def red_cadena_tubo():
    x = tf.keras.Input(shape=(N_P,), name="curva")
    m = tf.keras.Input(shape=(N_BASES,), name="laminas_decididas")
    h = x
    for u in (256, 128, 64):
        h = tf.keras.layers.Dense(u, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(h)
        h = tf.keras.layers.LayerNormalization()(h)
    sal = []
    for j in range(N_BASES):
        e = h if j == 0 else tf.keras.layers.Concatenate()([h, tf.keras.layers.Lambda(lambda t, j=j: t[:, :j])(m)])
        e = tf.keras.layers.Dense(128, activation="relu")(e)
        e = tf.keras.layers.Dense(64, activation="relu")(e)
        sal.append(Umbrales(K - 1, name=f"m{j + 1}")(tf.keras.layers.Dense(1, use_bias=False)(e)))
    e = tf.keras.layers.Concatenate()([h, m])
    e = tf.keras.layers.Dense(64, activation="relu")(e)
    sal.append(tf.keras.layers.Dense(1, name="delta")(e))
    return tf.keras.Model([x, m], sal)


def perdida_ordinal(y, logits):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits), axis=1)


def niveles(z):
    return (np.asarray(z)[:, None] > np.arange(K - 1)[None, :]).astype("float32")


def _objetivos(idx):
    return tuple(niveles(Z[idx, j]) for j in range(N_BASES)) + (((DELTA[idx] - D_MED) / D_SD).astype("float32")[:, None],)


def _con_ruido(entr, y):
    x, m = entr
    l = x * _sxs + _sxm
    rango = tf.reduce_max(l, axis=1, keepdims=True) - tf.reduce_min(l, axis=1, keepdims=True)
    u = tf.random.uniform([tf.shape(x)[0], 1], 0.0, RUIDO_MAX)
    return (((l + tf.random.normal(tf.shape(l)) * u * rango) - _sxm) / _sxs, m), y


RED = red_cadena_tubo()
RED.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=[perdida_ordinal] * N_BASES + ["mse"],
            loss_weights=[1.0] * N_BASES + [PESO_DELTA])
if ENTRENAR:
    t0 = time.time()
    M_va = (Z[idx_val] / ESCZ).astype("float32")
    ds = (tf.data.Dataset.from_tensor_slices(((X_train, (Z[idx_train] / ESCZ).astype("float32")), _objetivos(idx_train)))
          .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)
          .map(_con_ruido, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    H = RED.fit(ds, validation_data=((X_val, M_va), _objetivos(idx_val)), epochs=EPOCHS, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6)])
    hist = pd.DataFrame(H.history); hist.index = pd.RangeIndex(1, len(hist) + 1, name="epoca")
    hist.to_csv(ARCHIVO_HISTORIAL)
    RED.save_weights(ARCHIVO_MODELO)
    with open(ARCHIVO_SCALERS, "wb") as f:
        pickle.dump(dict(scaler_x=sx, D_MED=D_MED, D_SD=D_SD, E_MAT=E_MAT, NUDOS=NUDOS, R_REF=R_REF), f)
    print(f"✅ Red entrenada en {(time.time() - t0) / 60:.1f} min ({len(hist)} épocas; pérdida val. {hist.val_loss.min():.3f})")
else:
    RED.load_weights(ARCHIVO_MODELO)
    hist = pd.read_csv(ARCHIVO_HISTORIAL, index_col="epoca")
    with open(ARCHIVO_SCALERS, "rb") as f:
        _m = pickle.load(f)
    assert np.allclose(_m["E_MAT"], E_MAT), "la red se entrenó con otros materiales"
    sx, D_MED, D_SD = _m["scaler_x"], _m["D_MED"], _m["D_SD"]
    print(f"📦 Red cargada de '{ARCHIVO_MODELO}' ({len(hist)} épocas)")
print(f"   {RED.count_params():,} parámetros")
'''

C9 = '''# CELDA 9: CÓMO SE USA LA RED — paso a paso, más dos revisiones opcionales
#   (1) Cadena: la red se evalúa N veces; en la evaluación j, m = (n_1…n_{j−1}, 0…0)/20 y se lee la cabeza j
#       (n_j = cantidad de preguntas con P > 1/2). Al final, la cabeza de δ con todas las láminas.
#   (2) δ afinado: con las láminas fijas, δ se ajusta en ±0,03 (una sola variable, Brent acotado) contra el peor punto.
#   (3) Búsqueda ±1: se prueban las 3^N combinaciones a ±1 lámina por material (sin salir de 0…20), con ese δ; en las 8 de
#       menor peor punto se vuelve a afinar δ y se elige la mejor.
#   Antes de todo, la curva se suaviza con un polinomio de grado 6 en P (no cambia una curva limpia; a una medida le quita el ruido,
#   como hace el paso 2 del minimax).
import itertools
import numpy as np
from scipy.optimize import minimize_scalar

VECINOS = np.array(list(itertools.product((-1, 0, 1), repeat=N_BASES)))


def predecir_cadena(Y):
    Y = np.atleast_2d(Y)
    Xs = sx.transform(Y).astype("float32")
    zp = np.zeros((len(Y), N_BASES), "float32")
    for j in range(N_BASES):
        zp[:, j] = (RED.predict([Xs, zp / ESCZ], batch_size=1024, verbose=0)[j] > 0).sum(1)
    d = RED.predict([Xs, zp / ESCZ], batch_size=1024, verbose=0)[-1][:, 0] * D_SD + D_MED
    return zp.astype(int), np.clip(d, -DELTA_MAX, DELTA_MAX)


def theta(n, d):
    return np.r_[np.asarray(n, float) * ESP_LAMINA_MM, d]


def afinar_delta(y, n, d0):
    r = minimize_scalar(lambda d: e_max_th(y, theta(n, d)), bounds=(max(d0 - 0.03, -DELTA_MAX), min(d0 + 0.03, DELTA_MAX)),
                        method="bounded", options=dict(xatol=1e-5))
    return (r.x, r.fun) if r.fun <= e_max_th(y, theta(n, d0)) else (d0, e_max_th(y, theta(n, d0)))


def busqueda_pm1(y, n, d):
    cand = np.clip(n[None, :] + VECINOS, 0, TOPE_LAMINAS)
    cand = np.unique(cand, axis=0)
    e = np.array([e_max_th(y, theta(c, d)) for c in cand])
    mejores = [afinar_delta(y, cand[i], d) + (cand[i],) for i in np.argsort(e)[:8]]   # δ reajustado en las 8 mejores
    d2, e2, nb = min(mejores, key=lambda t: t[1])
    return nb, d2, e2


GRADO_SUAVE = 6


def suavizar(y, g=GRADO_SUAVE):
    """Polinomio de grado g en P ajustado por mínimos cuadrados: quita el ruido de una curva medida y no cambia una limpia."""
    x = (P_GRID - P_GRID.mean()) / np.ptp(P_GRID)
    return np.polyval(np.polyfit(x, y, g), x)


def disenar_con_red(Y):
    """Para cada curva: dict con (n, δ) de la red sola, con δ afinado y con la búsqueda ±1 (todo sobre la curva suavizada)."""
    Y = np.array([suavizar(y) for y in np.atleast_2d(Y)])
    Nn, D = predecir_cadena(Y)
    out = []
    for y, n, d in zip(Y, Nn, D):
        da, _ = afinar_delta(y, n, d)
        nb, db, _ = busqueda_pm1(y, n, da)
        out.append({"red": (n, d), "red + δ": (n, da), "red ±1": (nb, db)})
    return out
'''

C10 = '''# CELDA 10: RESULTADOS — 41 sujetos (entrenamiento y prueba) y curvas sintéticas de prueba, contra la curva original
import time
import numpy as np
import pandas as pd

METODOS = ["minimax", "red", "red + δ", "red ±1"]
RED(([sx.transform(POB_L[:2]).astype("float32"), np.zeros((2, N_BASES), "float32")]))
t0 = time.perf_counter(); DIS_RED = disenar_con_red(POB_L); T_RED = (time.perf_counter() - t0) / len(POB_L)
_fil = []
for i, (nm, y) in enumerate(zip(POB.id, POB_L)):
    th_r, n_r = REF_SUJ[i]
    dis = {"minimax": (n_r, th_r[-1]), **DIS_RED[i]}
    for m in METODOS:
        n, d = dis[m]
        er, em = errores_l(y, curva_injerto(theta(n, d)))
        _fil.append({"sujeto": nm, "conjunto": "prueba" if MASK_TEST[i] else "entrenamiento", "método": m, "Erms %": er, "e_max %": em,
                     "delta": d, "láminas": str(np.asarray(n, int).tolist()), "pared mm": np.sum(n) * ESP_LAMINA_MM,
                     **{f"n{j + 1}": int(n[j]) for j in range(N_BASES)}})
SUJ59 = pd.DataFrame(_fil)
SUJ59.to_csv(f"sujetos_{VERSION}.csv", index=False)

# curvas sintéticas de prueba (las arterias del conjunto de prueba, a lo sumo 300)
_it = idx_test[familia[idx_test] != 1][:300]
DIS_T = disenar_con_red(Y_ALL[_it])
_ft = []
for k, i in enumerate(_it):
    y = Y_ALL[i]
    dis = {"minimax": (Z[i], DELTA[i]), **DIS_T[k]}
    for m in METODOS:
        n, d = dis[m]
        _ft.append({"método": m, "e_max %": errores_l(y, curva_injerto(theta(n, d)))[1], "igual": bool(np.all(np.asarray(n) == Z[i]))})
TEST59 = pd.DataFrame(_ft)


def _res(D, grupo):
    return D.groupby("método", sort=False)["e_max %"].agg(
        mediana="median", peor="max", cumplen=lambda v: f"{(v <= ERROR_MAX_PCT).sum()}/{len(v)}").assign(grupo=grupo)


RES59 = pd.concat([_res(SUJ59[SUJ59.conjunto == "entrenamiento"], "41 sujetos: entrenamiento (31)"),
                   _res(SUJ59[SUJ59.conjunto == "prueba"], "41 sujetos: prueba (10)"),
                   _res(TEST59, f"arterias sintéticas de prueba ({len(_it)})")])
RES59.round(2).to_csv(f"resumen_{VERSION}.csv")
print(f"Diseño ENTERO (láminas de {ESP_LAMINA_MM:g} mm, a lo sumo {TOPE_LAMINAS} por material) — peor error en λ [% del rango]")
print(RES59.round(2).to_string())
print(f"\\nTiempo por sujeto: minimax {T_MINIMAX:.2f} s | red (cadena + δ afinado + ±1) {T_RED:.2f} s")
print("Láminas iguales al minimax (sintéticas de prueba):", TEST59.groupby("método", sort=False)["igual"].mean().round(3).to_dict())
'''

C11 = '''# CELDA 11: CURVAS MEDIDAS — ruido en el diámetro o pocas presiones; el diseño se compara con la curva VERDADERA
#   El minimax recibe la curva medida (su paso 2 la suaviza y el paso 3 elige las láminas contra la curva suave).
#   La red recibe la curva medida; el δ afinado y la búsqueda ±1 se hacen contra la curva medida.
import time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ESC_ROB = {"ruido 0,3 %": (0.003, None), "10 presiones": (0.0, 10), "10 presiones + ruido 0,2 %": (0.002, 10)}
_rng = np.random.default_rng(SEED + 5)
_fil = []
t00 = time.time()
for esc, (r, npts) in ESC_ROB.items():
    YM = []
    for y in POB_L:
        if npts:
            pm = np.linspace(P_GRID[0], P_GRID[-1], npts)
            ym = np.interp(pm, P_GRID, y)
            ym = PchipInterpolator(pm, ym + _rng.normal(0.0, r, size=npts) * ym)(P_GRID)
        else:
            ym = y + _rng.normal(0.0, r, size=y.shape) * y
        YM.append(ym)
    YM = np.array(YM)
    DR = disenar_con_red(YM)
    for i, (nm, y, ym) in enumerate(zip(POB.id, POB_L, YM)):
        th, n = minimax_tubo(ym)
        dis = {"minimax": (n, th[-1]), **DR[i]}
        for m in METODOS:
            nn, d = dis[m]
            _fil.append({"escenario": esc, "método": m, "sujeto": nm, "e_max %": e_max_th(y, theta(nn, d))})
ROB59 = pd.DataFrame(_fil)
ROB59.to_csv(f"robustez_{VERSION}.csv", index=False)
print(f"Curvas MEDIDAS → diseño entero comparado con la curva VERDADERA | 41 sujetos | {(time.time() - t00) / 60:.1f} min")
print(ROB59.groupby(["escenario", "método"], sort=False)["e_max %"].agg(
    mediana="median", peor="max", cumplen=lambda v: f"{(v <= ERROR_MAX_PCT).sum()}/41").round(2).to_string())
'''

celdas = [("markdown", INTRO), ("code", c0), ("markdown", "## Celda 1 — Los 41 sujetos"), ("code", cs[1]),
          ("markdown", "## Celda 2 — Arterias sintéticas"), ("code", cs[2]),
          ("markdown", "## Celda 3 — El injerto de pared gruesa (rigidez recalibrada)"), ("code", c3),
          ("markdown", "## Celda 4 — Mínimos cuadrados no lineales (con el tope de láminas)"), ("code", c4),
          ("markdown", "## Celda 5 — Referencia: el minimax del tubo"), ("code", C5),
          ("markdown", "## Celda 6 — Dataset con etiquetas enteras"), ("code", C6),
          ("markdown", "## Celda 7 — Partición y normalización"), ("code", C7),
          ("markdown", "## Celda 8 — La red en cadena ordinal para el tubo"), ("code", C8),
          ("markdown", "## Celda 9 — Cómo se usa la red"), ("code", C9),
          ("markdown", "## Celda 10 — Resultados con curvas limpias"), ("code", C10),
          ("markdown", "## Celda 11 — Curvas medidas"), ("code", C11)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in celdas],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{S}/v59_{N}/curva_j_v59_{N}.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok", f"{S}/v59_{N}/curva_j_v59_{N}.ipynb")
