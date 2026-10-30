# Biomimesis

Diseño de injertos vasculares biomiméticos multicapa: una red neuronal y mínimos cuadrados (MCO) calculan, para la curva tensión–estiramiento circunferencial σθθ(λθ) de una carótida, cuántas láminas de cada material hay que apilar.

## Cuadernos en Google Colab (versión 40)

Los botones abren los cuadernos desde este repositorio (como es privado, Colab pide autorizar el acceso a GitHub la primera vez: marcar *Incluir repositorios privados*).

1. Subir la carpeta `Colab` de este repositorio a Google Drive, dentro de `Mi unidad/Biomimesis/` (queda `Mi unidad/Biomimesis/Colab/V40_5bases` y `…/V40_6bases`).
2. Abrir el cuaderno con el botón y ejecutar *Entorno de ejecución → Ejecutar todas*; aceptar la conexión con Drive.
3. La primera celda toma de Drive los datos y las redes ya entrenadas: no se entrena nada (≈ 15–25 min).

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V40_5bases/curva_j_v40_5_colab.ipynb) |
| 6 materiales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablosilvoni/Biomimesis/blob/main/Colab/V40_6bases/curva_j_v40_6_colab.ipynb) |

Cada carpeta `Colab/V40_Nbases` contiene el cuaderno, las tablas de Masson et al. (2011), las bases (`bases_v40_N.csv`), las etiquetas enteras y las dos redes que se comparan (`v40_N_enteras` y `v35_N`).
