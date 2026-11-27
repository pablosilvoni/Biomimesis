# Biomimesis

Ajuste de curvas en J con una base de rectas truncadas, coeficientes enteros y una red neuronal. Aplicado a la curva tensión–estiramiento circunferencial σθθ(λθ) de la carótida: la red da cuántas láminas de cada material hay que apilar.

## Resumen

**[Ajuste de curvas en J - Santiago y Pablo.pdf](Ajuste%20de%20curvas%20en%20J%20-%20Santiago%20y%20Pablo.pdf)**: cómo se ajusta una curva en J con rectas truncadas y coeficientes enteros, cómo se elige la base (precisión, cond(Ã) < 30, coeficientes no negativos), por qué se usa el minimax y cómo la red neuronal en cadena da los coeficientes sin resolver ninguna optimización.

## Cuaderno en Google Colab — V58

La red en cadena con salida ordinal da los coeficientes enteros (láminas por material, de 0 a 20) de una curva σθθ(λθ) y se compara con el minimax (MILP, HiGHS) en los 41 sujetos de Masson et al. (2011) y en 2000 curvas sintéticas, con y sin la revisión de ±1.

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/Version%2058%20-%205%20bases/curva_j_v58_5.ipynb) |
| 6 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/Version%2058%20-%206%20bases/curva_j_v58_6.ipynb) |

**Cómo se usa.** El repositorio es privado: la primera vez, Colab pide autorizar el acceso a GitHub (marcar *Incluir repositorios privados*).

1. Subir la carpeta `Colab/Version 58 - 5 bases` o `Colab/Version 58 - 6 bases` de este repositorio a Google Drive, en `Mi unidad/Biomimesis/Colab/`.
2. Abrir el cuaderno con el botón y ejecutar *Entorno de ejecución → Ejecutar todas*; aceptar la conexión con Drive.
3. La celda de preparación toma de Drive las tablas de Masson, las bases, las etiquetas del minimax y la red ya entrenada, y el cuaderno muestra los resultados sin entrenar.

**Para ver cómo entrena la red.** En la Celda 0, cambiar `ENTRENAR = "auto"` por `ENTRENAR = True`: la red se entrena de nuevo (hasta 800 épocas, con parada temprana) a partir de las etiquetas del minimax ya calculadas. Conviene elegir un entorno con GPU (*Entorno de ejecución → Cambiar tipo de entorno*). Las curvas de aprendizaje del entrenamiento original están en `historial_entrenamiento_v58_N_cadena.csv`.

Cada carpeta contiene el cuaderno, las tablas de Masson et al. (2011), las bases (`bases_v40_N.csv`), las etiquetas enteras del minimax (`etiquetas_enteras_v58_N.npz`), la red en cadena (`modelo_rn_curva_j_v58_N_cadena.weights.h5`, `scalers_curva_j_v58_N_cadena.pkl`) y los resultados (`tabla_41_RN_vs_minimax_v58_N.csv`, `comparacion_enteros_v58_N.csv`, validación externa `v58_externas_N.csv`). El `LEEME.txt` describe la red.
