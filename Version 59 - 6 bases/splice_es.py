S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad/artfin2"
s = open(f"{S}/articulo_final.tex").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
# 1. cuerpo de la Sección 4
i = s.index(r"\subsection{Pared delgada y pared gruesa}"); j = s.index("%=====================================================================\n\\section{Discusión}")
s = s[:i] + open(f"{S}/sec4_cuerpo_es.tex").read() + "\n" + open(f"{S}/sec4_res_es.tex").read() + "\n" + s[j:]
# 2. resumen
rep(r"""Luego el método se aplica a la curva presión--diámetro, para la que el injerto es un tubo de pared gruesa: diseñado con la ley de Laplace ningún sujeto cumple al fabricarse, mientras que con pared gruesa cumplen 40--41 de 41. El diseño se vuelve un problema no lineal entero, que se resuelve en tres pasos: arranque, mínimos cuadrados no lineales y búsqueda entera minimax contra la curva ajustada. Allí una red usada como punto de partida evita los mínimos locales que con curvas medidas producen diseños inservibles (peor error de 468,7\,\% a 2,3\,\%).""",
r"""Luego el método se aplica a la curva presión--diámetro, para la que el injerto es un tubo de pared gruesa: diseñado con la ley de Laplace ningún sujeto cumple al fabricarse, mientras que con pared gruesa cumplen los 41. El diseño se vuelve un problema no lineal entero; como referencia se resuelve en tres pasos (arranque, mínimos cuadrados no lineales y minimax entero), que cumplen en los 41 sujetos. La misma red en cadena, con una cabeza más para el preestiramiento, da las láminas de cada capa en una evaluación: con una revisión de $\pm1$ lámina cumple en los 41 sujetos con 6 materiales y en 40 con 5, y con curvas medidas con ruido cumple en más sujetos que el minimax, con un peor caso menor.""")
rep(r"y (v) un procedimiento de tres pasos en el que una red cumple el papel de punto de partida.",
    r"y (v) la misma red en cadena aplicada al tubo de pared gruesa, comparada con un procedimiento de optimización de tres pasos.")
# 3. discusión
rep(r"""En pared gruesa el ajuste es iterativo, local y depende del punto de partida: el arranque clásico, heredado del caso lineal, ignora el gradiente de estiramiento en el espesor y, con curvas medidas, puede llevar a mínimos locales que dan diseños inservibles. Una red entrenada sobre miles de diseños resueltos deja al ajuste cerca del mínimo correcto. Es la idea de la optimización amortizada~\cite{amos}: la red no reemplaza al optimizador, lo inicia. Su ventaja es de robustez, clara con 6 materiales y menor con 5.""",
r"""En el tubo de pared gruesa la misma red, con la misma salida ordinal, da las láminas de cada capa y el preestiramiento; con la revisión de $\pm1$ lámina iguala al procedimiento de tres pasos con curvas limpias y lo supera con curvas con ruido, porque elige entre diseños vistos en el entrenamiento y no persigue el ruido. Es la idea de la optimización amortizada~\cite{amos}: un modelo aprendido sobre muchos casos resueltos da la solución de uno nuevo, y una búsqueda corta la termina. La red no reemplaza al minimax como referencia para comprobar un diseño.""")
rep(r"(3) La búsqueda entera en pared gruesa es heurística: queda a menos de medio punto del óptimo en una vecindad de $\pm2$ láminas, pero no se demostró el óptimo global; se usó una sola realización de ruido por sujeto y caso.",
    r"(3) La búsqueda entera en pared gruesa es heurística: queda a menos de medio punto del óptimo en una vecindad de $\pm2$ láminas, pero no se demostró el óptimo global; se usó una sola realización de ruido por sujeto y caso. Para respetar el tope de 20 láminas se aumentó la rigidez de los materiales más rígidos, lo que deberá verificarse con materiales reales.")
# 4. conclusiones
rep(r"""\item A nivel del tubo el diseño debe hacerse en pared gruesa: diseñado con la ley de Laplace, ningún sujeto cumple al fabricarse; en pared gruesa cumplen 40--41 de 41. El problema es no lineal entero.""",
    r"""\item A nivel del tubo el diseño debe hacerse en pared gruesa: diseñado con la ley de Laplace, ningún sujeto cumple al fabricarse; en pared gruesa cumplen los 41. El problema es no lineal entero.""")
i = s.index(r"\item El procedimiento de tres pasos ---arranque"); j = s.index(r"\end{enumerate}", i)
s = s[:i] + r"""\item Con láminas de 0,025\,mm y a lo sumo 20 por material, el procedimiento de tres pasos ---arranque, mínimos cuadrados no lineales y minimax entero--- cumple en los 41 sujetos, con 6 y con 5 materiales, en menos de un segundo por sujeto.
\item La misma red en cadena, entrenada para el tubo, da las láminas de cada capa en una evaluación; con $\delta$ afinado y una revisión de $\pm1$ lámina cumple en los 41 sujetos con 6 materiales y en 40 con 5. Con curvas medidas con ruido cumple en más sujetos que el minimax (29 contra 24 con 6 materiales) y con un peor caso menor. Una sola red sirve así para los dos niveles.
""" + s[j:]
# 5. anexos
rep("\\section{Entrenamiento de las redes neuronales}", open(f"{S}/anexo_calculo_es.tex").read() + "\n\\section{Entrenamiento de las redes neuronales}")
rep(r"y la de pared gruesa (bloque A4), cada una con 6 y con 5 materiales.", r"y la red en cadena del tubo, que va de la curva $\lambda(P)$ a las láminas de cada capa y al preestiramiento (Sección~\ref{sec:aplicacion}), cada una con 6 y con 5 materiales.")
rep("{curvas_aprendizaje_es.png}", "{curvas_aprendizaje_v59_es.png}")
rep(r"Arriba, la red en cadena del primer nivel (entropía cruzada binaria sumada sobre las 20 preguntas y los materiales); abajo, la red de pared gruesa (error cuadrático medio del diseño estandarizado).",
    r"Arriba, la red en cadena del primer nivel (entropía cruzada binaria sumada sobre las 20 preguntas y los materiales); abajo, la del tubo (la misma suma más 5 veces el error cuadrático medio de $\delta$ estandarizado). En la red del primer nivel se conservan los pesos finales; en la del tubo, los del mínimo de validación.")
rep(r"la red en cadena usa la entropía cruzada binaria sumada sobre las preguntas y los materiales (en nats) y la de pared gruesa el error cuadrático medio del diseño estandarizado, por lo que los valores de las dos redes no se comparan entre sí.",
    r"las dos redes usan la entropía cruzada binaria sumada sobre las preguntas y los materiales (en nats); la del tubo suma además 5 veces el error cuadrático medio de $\delta$ estandarizado, que es menor que 0,02.")
rep("Pared gruesa, 6 mat. & 11\\,220 / 1530 / 2250 & 425 / 305 & 5 & 0,342 & 0,462 & 0,484 & 1,42\\\\",
    "Red en cadena del tubo, 6 mat. & 11\\,220 / 1530 / 2250 & 288 / 228 & 5 & 14,91 & 21,72 & 22,57 & 1,51\\\\")
rep("Pared gruesa, 5 mat. & 11\\,220 / 1530 / 2250 & 629 / 509 & 4 & 0,153 & 0,227 & 0,250 & 1,64\\\\",
    "Red en cadena del tubo, 5 mat. & 11\\,220 / 1530 / 2250 & 409 / 349 & 6 & 8,88 & 14,61 & 14,69 & 1,65\\\\")
i = s.index(r"\paragraph{Red de pared gruesa.}"); j = s.index(r"\clearpage", i)
s = s[:i] + r"""\paragraph{Red en cadena del tubo.} La validación se separa del entrenamiento desde las primeras 30 épocas y se estabiliza, mientras el entrenamiento sigue bajando. La parada temprana conserva los pesos del mínimo de validación, y la pérdida de prueba queda cerca de la de validación (22,57 contra 21,72 y 14,69 contra 14,61): la red no se degrada con curvas nuevas. La pérdida es más alta que en el primer nivel, y la razón está en el problema y no en la red: en el tubo hay todavía más diseños enteros con casi la misma curva.

\emph{Condicionamiento.} Para 60 arterias de las familias fisiológicas (con los módulos sin recalibrar) se calculó la matriz jacobiana $\mathbf{D}$ de la curva del injerto respecto del diseño $\theta=(h_1\ldots h_N,\delta)$, por diferencias finitas, con cada parámetro en desvíos estándar y la curva en \% de su rango. Con 6 materiales, en la dirección más sensible la curva cambia 8,5 veces su rango por desvío; en la menos sensible, apenas un 1\,\%. El número de condición tiene una mediana de 900 con 6 materiales y 300 con 5, contra $\mathrm{cond}(A)=28$ en el primer nivel. La dirección menos sensible es, típicamente, más espesor de un material y menos de su vecino. Desde la curva, esos diseños son indistinguibles para cualquier método, y la red no puede adivinar cuál de ellos es la etiqueta.

\emph{La medida que importa.} La red da exactamente las láminas del minimax en el 7\,\% de las arterias sintéticas de prueba con 6 materiales y en el 26\,\% con 5; aun así, con la revisión de $\pm1$, su diseño cumple la meta en 294 y 297 de 300. Como en el primer nivel, la red se juzga por el error en la curva y no por coincidir con el minimax.

""" + s[j:]
rep(r"""\section{Láminas y errores de cada curva}\label{sec:tablas}
Láminas enteras de cada material y errores de la red en cadena, de la red en cadena con búsqueda de $\pm1$ y del minimax, en los 41 sujetos de Masson y en las 63 curvas externas dentro de $y_1$--$y_2$, con 6 y con 5 materiales. Las láminas se listan del material 1 al $N$.""",
r"""\section{Láminas y errores de cada curva}\label{sec:tablas}
\textbf{Primer nivel, curva tensión--estiramiento} (Tablas~\ref{tab:suj6} a \ref{tab:ext5}): láminas enteras de cada material y errores de la red en cadena, de la red en cadena con búsqueda de $\pm1$ y del minimax, en los 41 sujetos de Masson y en las 63 curvas externas dentro de $y_1$--$y_2$, con 6 y con 5 materiales. \textbf{Nivel del tubo, curva presión--diámetro} (Tablas~\ref{tab:tubo6} y \ref{tab:tubo5}): láminas de cada capa, preestiramiento y errores de la red en cadena con revisión de $\pm1$ y del minimax de tres pasos, en los 41 sujetos. Las láminas se listan del material 1 al $N$.""")
rep("\\input{tab_ext5_es.tex}", "\\input{tab_ext5_es.tex}\n\\input{tab_tubo6_es.tex}\n\\input{tab_tubo5_es.tex}")
rep(r"\date{9 de noviembre de 2026}", r"\date{11 de noviembre de 2026}")
open(f"{S}/articulo_final.tex", "w").write(s)
print("ok")
