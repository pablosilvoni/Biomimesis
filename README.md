# Biomimesis

Diseño de injertos vasculares biomiméticos multicapa: una red neuronal y mínimos cuadrados (MCO) calculan, para la curva tensión–estiramiento circunferencial σθθ(λθ) de una carótida, cuántas láminas de cada material hay que apilar.

## Por dónde empezar

**[Ajuste de curvas en J - Santiago y Pablo.pdf](Ajuste%20de%20curvas%20en%20J%20-%20Santiago%20y%20Pablo.pdf)** (carpeta principal) resume el primer nivel: cómo se ajusta una curva en J con una base de rectas truncadas y coeficientes enteros, cómo se elige la base (precisión, cond(Ã) < 30, coeficientes no negativos), por qué se usa el minimax y cómo la red neuronal en cadena (V58) da los coeficientes sin resolver ninguna optimización. Su cuaderno es el de la **V58**, más abajo.

## Qué hay en cada carpeta

| Carpeta | Contenido |
|---|---|
| `Colab/` | Cuadernos listos para abrir en Google Colab (V58, V40 y V47), con sus datos y redes ya entrenadas |
| `Version 58 - 5 bases/`, `Version 58 - 6 bases/` | Versión de trabajo de la V58: cuaderno, scripts, pesos de la red y resultados |
| `Otras versiones/` | Versiones anteriores. En GitHub solo está la V59 (red en cadena para el injerto de pared gruesa); el resto queda en la computadora |
| `Drafts - Resumen - Informes/` | Artículo final (español e inglés, con su fuente LaTeX), fuente LaTeX del resumen e informes de cada etapa |
| `Artículos/` | Bibliografía y datos de Masson et al. (2011) |
| `Análisis Numérico/`, `Algebra lineal para Analisis de Datos/` | Material de consulta |
| `MATLAB/` | Scripts de MATLAB |

## Cuadernos en Google Colab

Un cuaderno por número de materiales (5 o 6) en cada versión: **V58** para el primer nivel con la red en cadena (la del resumen), **V40** para el primer nivel con los redondeos a láminas enteras y **V47** para el segundo nivel (curva presión–diámetro, injerto de pared gruesa).

**Cómo se usan.** Los botones abren los cuadernos desde este repositorio (como es privado, Colab pide autorizar el acceso a GitHub la primera vez: marcar *Incluir repositorios privados*).

1. Subir la carpeta `Colab` de este repositorio a Google Drive, dentro de `Mi unidad/Biomimesis/` (quedan `Mi unidad/Biomimesis/Colab/Version 58 - 5 bases`, `…/Version 58 - 6 bases`, `…/V40_5bases`, `…/V40_6bases`, `…/V47_5bases` y `…/V47_6bases`).
2. Abrir el cuaderno con el botón y ejecutar *Entorno de ejecución → Ejecutar todas*; aceptar la conexión con Drive.
3. La celda de preparación toma de Drive los datos y las redes ya entrenadas: no se entrena nada.

### Primer nivel, red en cadena — V58 (la del resumen)

La red en cadena con salida ordinal da los coeficientes enteros (láminas por material, de 0 a 20) de una curva σθθ(λθ) y se compara con el minimax (MILP, HiGHS). El cuaderno carga la red y las etiquetas del minimax, evalúa los 41 sujetos de Masson et al. (2011) y 2000 curvas sintéticas, con y sin la revisión de ±1, y dibuja la figura de e_máx.

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/Version%2058%20-%205%20bases/curva_j_v58_5.ipynb) |
| 6 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/Version%2058%20-%206%20bases/curva_j_v58_6.ipynb) |

Cada carpeta `Colab/Version 58 - N bases` contiene el cuaderno, las tablas de Masson et al. (2011), las bases (`bases_v40_N.csv`), las etiquetas enteras del minimax (`etiquetas_enteras_v58_N.npz`), la red en cadena (`modelo_rn_curva_j_v58_N_cadena.weights.h5`, `scalers_curva_j_v58_N_cadena.pkl`, historial) y los resultados (`tabla_41_RN_vs_minimax_v58_N.csv`, `comparacion_enteros_v58_N.csv`, validación externa `v58_externas_N.csv`). El `LEEME.txt` de cada carpeta describe la red.

### Primer nivel, redondeos — V40

Compara la red entrenada con láminas enteras con la entrenada con MCO real, con distintos redondeos (≈ 15–25 min).

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
