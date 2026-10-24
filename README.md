# Biomimesis

Diseño de injertos vasculares biomiméticos multicapa: una red neuronal y mínimos cuadrados (MCO) calculan, para la curva presión–diámetro de una arteria, cuántas láminas (o qué espesor) de cada material hay que apilar.

## Cuadernos en Google Colab (versión 31)

Los botones abren los cuadernos guardados en Google Drive, con sus resultados.

| Cuaderno | Abrir en Colab |
|---|---|
| 5 materiales (red `v27_5`) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hhkwVlOs3vcQvEqMLnlVrl568qugMgD7) |
| 6 materiales (red `v28_6`) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-ZVko-t17tnDvwWzj_j4Oi3AvYTlzaNF) |

Cada cuaderno necesita, en su misma carpeta: `tabla2_masson_NT.csv`, `tabla4_masson_HT.csv`, `masson2011_curvas_medias.csv` y los tres archivos de su red (`modelo_rn_curva_j_….keras`, `scalers_curva_j_….pkl`, `historial_entrenamiento_….csv`). En Colab hace falta Keras 3.15 (`!pip install -q "keras==3.15.0"` y reiniciar la sesión).
