# Biomimesis

Diseño de injertos vasculares biomiméticos multicapa: una red neuronal y mínimos cuadrados (MCO) calculan, para la curva tensión–estiramiento circunferencial σθθ(λθ) de una carótida, cuántas láminas de cada material hay que apilar.

## Cuadernos en Google Colab

Dos cuadernos por número de materiales: **V40** para el primer nivel (curva tensión–estiramiento σθθ(λθ), redondeos a láminas enteras) y **V47** para el segundo nivel (curva presión–diámetro, injerto de pared gruesa). Son los que respaldan el artículo.

### Primer nivel — V40

Los botones abren los cuadernos desde este repositorio (como es privado, Colab pide autorizar el acceso a GitHub la primera vez: marcar *Incluir repositorios privados*).

1. Subir la carpeta `Colab` de este repositorio a Google Drive, dentro de `Mi unidad/Biomimesis/` (quedan `Mi unidad/Biomimesis/Colab/V40_5bases`, `…/V40_6bases`, `…/V47_5bases` y `…/V47_6bases`).
2. Abrir el cuaderno con el botón y ejecutar *Entorno de ejecución → Ejecutar todas*; aceptar la conexión con Drive.
3. La primera celda toma de Drive los datos y las redes ya entrenadas: no se entrena nada (≈ 15–25 min).

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V40_5bases/curva_j_v40_5_colab.ipynb) |
| 6 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V40_6bases/curva_j_v40_6_colab.ipynb) |

Cada carpeta `Colab/V40_Nbases` contiene el cuaderno, las tablas de Masson et al. (2011), las bases (`bases_v40_N.csv`), las etiquetas enteras y las dos redes que se comparan (`v40_N_enteras` y `v35_N`).

La Celda 28 compara el redondeo conjunto (Schnorr–Euchner) arrancando de la red con el que arranca de los mínimos cuadrados, contra la curva original.

### Segundo nivel, pared gruesa — V47

Mismo procedimiento que arriba. El cuaderno carga la red ya entrenada y las etiquetas: no entrena nada. Diseña los 41 sujetos por los dos caminos (arranque lineal o red → mínimos cuadrados no lineales → láminas enteras por minimax contra la curva ajustada) y repite con curvas medidas (ruido y pocas presiones). La Celda 11, apagada por defecto (`CORRER_EXHAUSTIVA = False`), mide la distancia al óptimo entero del paso 3 con una búsqueda exhaustiva de ±2 láminas por material (≈ 70 min con 6 materiales).

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V47_5bases/curva_j_v47_5_colab.ipynb) |
| 6 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V47_6bases/curva_j_v47_6_colab.ipynb) |

Cada carpeta `Colab/V47_Nbases` contiene el cuaderno, las tablas de Masson et al. (2011), la red entrenada (`modelo_rn_v47_N.keras`, `scalers_v47_N.pkl`, historial) y sus etiquetas (`etiquetas_pared_gruesa_v47_N.npz`).
