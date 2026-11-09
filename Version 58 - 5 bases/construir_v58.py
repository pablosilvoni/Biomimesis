# V58: la misma red en cadena, pero cada cabeza es ORDINAL (CORAL, Cao et al. 2020): en vez de 21 clases sin orden,
#   responde 20 preguntas "¿más de k láminas?" (k = 0…19) con un mismo vector de pesos y 20 umbrales; n = cantidad de "sí".
import json, sys, shutil, os
S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"
N = int(sys.argv[1]); d0, d1 = f"{S}/v56_{N}", f"{S}/v58_{N}"
os.makedirs(d1, exist_ok=True)
nb = json.load(open(f"{d0}/curva_j_v56_{N}.ipynb"))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
def src(c): return ''.join(c['source'])
def put(c, s): c['source'] = s; c['outputs'] = []; c['execution_count'] = None
c0 = src(code[0]).replace('VERSION = f"v56_{N_BASES}"', 'VERSION = f"v58_{N_BASES}"')
c0 = c0.replace('ARCHIVO_ETIQUETAS_ENTERAS = f"etiquetas_enteras_v56_{N_BASES}.npz"', 'ARCHIVO_ETIQUETAS_ENTERAS = f"etiquetas_enteras_v58_{N_BASES}.npz"')
c0 = c0.replace("# una sola red: la red en cadena, entrenada dentro de y1–y2", "# una sola red: la red en cadena con cabezas ordinales, entrenada dentro de y1–y2")
assert 'v58_' in c0; put(code[0], c0)
c11 = src(code[11])
a = c11.index("def entrenar")
c11 = c11[:a] + '''def perdida_ordinal(y, logits):
    """CORAL: suma, sobre k = 0…K−2, de la entropía cruzada binaria de la pregunta "¿más de k láminas?"."""
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits), axis=1)


def niveles(z):
    """Láminas enteras → respuestas a "¿más de k láminas?" (k = 0…K−2), como 0/1."""
    return (np.asarray(z)[:, None] > np.arange(K - 1)[None, :]).astype("float32")


def entrenar(model, entradas_tr, entradas_va, nombre):
    tf.random.set_seed(SEED); np.random.seed(SEED)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=[perdida_ordinal] * N_BASES)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6)]
    t0 = time.time()
    H = model.fit(entradas_tr, [niveles(Z[idx_train, j]) for j in range(N_BASES)],
                  validation_data=(entradas_va, [niveles(Z[idx_val, j]) for j in range(N_BASES)]),
                  epochs=800, batch_size=256, callbacks=cb, verbose=0)
    pd.DataFrame(H.history).to_csv(f"historial_entrenamiento_{VERSION}_{nombre}.csv", index=False)
    print(f"   {nombre}: {len(H.history['loss'])} épocas, {(time.time() - t0) / 60:.1f} min | "
          f"pérdida val. {min(H.history['val_loss']):.3f}")
'''
put(code[11], "# CELDA 11: TRONCO COMÚN Y ENTRENAMIENTO (red en cadena con cabezas ordinales, CORAL)\n" + c11.split("\n", 1)[1])
c12 = src(code[12])
old_head = 'salidas.append(tf.keras.layers.Dense(K, activation="softmax", name=f"m{j + 1}")(e))'
assert old_head in c12
c12 = c12.replace(old_head, 'salidas.append(Umbrales(K - 1, name=f"m{j + 1}")(tf.keras.layers.Dense(1, use_bias=False)(e)))   # un peso, K−1 umbrales')
c12 = c12.replace("def red_cadena():", '''class Umbrales(tf.keras.layers.Layer):
    """CORAL: logit_k = s + b_k (k = 0…K−2) con el MISMO puntaje s y un umbral b_k por pregunta "¿más de k láminas?"."""
    def __init__(self, n, **kw):
        super().__init__(**kw); self.n = n
    def build(self, shape):
        self.b = self.add_weight(name="umbrales", shape=(self.n,), initializer=tf.keras.initializers.Constant(
            np.linspace(2.0, -2.0, self.n)), trainable=True)
    def call(self, s):
        return s + self.b


def red_cadena():''')
old_pred = "zp[:, j] = RED_CAD.predict([Xs, zp / ESCZ], batch_size=1024, verbose=0)[j].argmax(1)"
assert old_pred in c12
c12 = c12.replace(old_pred, "zp[:, j] = (RED_CAD.predict([Xs, zp / ESCZ], batch_size=1024, verbose=0)[j] > 0).sum(1)   # cantidad de \"sí\" (P > 0,5)")
c12 = c12.replace("# CELDA 12: RED EN CADENA — elige los materiales de a uno, sabiendo los anteriores",
                  "# CELDA 12: RED EN CADENA CON CABEZAS ORDINALES (CORAL) — elige los materiales de a uno, sabiendo los anteriores\n"
                  "#   Cada cabeza responde K−1 preguntas \"¿más de k láminas?\" con un mismo puntaje y K−1 umbrales (Cao et al., 2020):\n"
                  "#   las láminas son la cantidad de \"sí\"; equivocarse por una lámina cuesta poco y por diez, mucho.")
put(code[12], c12)
for c in code[13:]: put(c, src(c))
for c in code[1:11]: put(c, src(c))
json.dump(nb, open(f"{d1}/curva_j_v58_{N}.ipynb", "w"), ensure_ascii=False, indent=1)
for f in os.listdir(d0):
    if f.endswith(".csv") and ("masson" in f or f.startswith("bases_v40") or f.startswith("plantilla") or f.startswith("tabla2") or f.startswith("tabla4")):
        shutil.copy(f"{d0}/{f}", d1)
shutil.copy(f"{d0}/etiquetas_enteras_v56_{N}.npz", f"{d1}/etiquetas_enteras_v58_{N}.npz")
for g in ("41_sujetos", "2000_curvas"):
    shutil.copy(f"{d0}/minimax_{g}_v56_{N}.npz", f"{d1}/minimax_{g}_v58_{N}.npz")
print("ok", sorted(os.listdir(d1)))
