# Esquema TikZ de la red en cadena (mismo estilo que el esquema de la red del primer nivel). Uso: python3 gen.py
T = {
 "es": dict(babel="spanish",
   titulo="Red en cadena ordinal para el tubo: una cabeza por material y una para $\delta$",
   sub=r"tronco $141\to256\to128\to64$; cabeza $j$: $(h,\,m_1\ldots m_{j-1})\to128\to64\to s_j$; 20 preguntas $P(n_j>k)=\sigma(s_j+b_{jk})$; cabeza de $\delta$: $(h,\,\mathbf{m})\to64\to\delta$",
   entrada=r"\textbf{Entrada}\\[-1pt]\footnotesize 141 presiones\\[-1pt]\footnotesize\itshape $\lambda(P)$",
   capa=lambda k, u: rf"\textbf{{\shortstack{{Capa\\oculta {k}}}}}\\[-1pt]\footnotesize {u} neuronas",
   reln=r"\footnotesize\itshape ReLU +\\[-1pt]\footnotesize\itshape norm. por capa",
   tronco="Tronco común (la curva)",
   cabeza=lambda j: (r"\textbf{Cabeza 1}\\ $h\to128\to64\to$ ordinal" if j == 1 else r"\textbf{Cabeza de $\delta$}\\ $(h,\,m_1\ldots m_6)\to64\to\delta$" if j == 7 else
                     rf"\textbf{{Cabeza {j}}}\\ $(h,\,m_1\ldots m_{{{j-1}}})\to128\to64\to$ ordinal"),
   prev=r"vector de láminas decididas $\mathbf{m}$",
   pre=r"\textbf{Preprocesamiento}\\ suavizado: polinomio\\ de grado 6 en $P$\\ estandarización",
   busq=r"\textbf{Revisión}\\ (opcional)\\ afina $\delta$\\ (una variable);\\ prueba $3^N$ vecinos\\ $n_j-1,\ n_j,\ n_j+1$\\ y elige el de\\ menor $e_{\max}$",
   sal=r"\textbf{Diseño del tubo}\\ láminas $n_1\ldots n_6$\\ ($h_j=n_j\times0{,}025$\,mm)\\ y preestiramiento $\delta$",
   loss=r"\textbf{Pérdida: entropía cruzada binaria por pregunta + error de $\delta$}\\[2pt] $\mathcal{L}=\sum_{j=1}^{6}\sum_{k=0}^{19}\mathrm{BCE}\bigl([n_j^{\mathrm{etiq}}>k],\,P(n_j>k)\bigr)+5\,(\delta-\delta^{\mathrm{etiq}})^2$\\[1pt] \footnotesize al entrenar, cada cabeza recibe en $\mathbf{m}$ las láminas de la etiqueta; $\delta$ estandarizado",
   nota=r"Cada cabeza responde «¿más de $k$ láminas?» para $k=0\ldots19$; $n_j$ (0 a 20) es la cantidad de «sí». Elige sabiendo lo que ya eligieron las anteriores, para poder compensar entre materiales. Se usa en orden: material 1, después el 2 sabiendo el 1, y así hasta el 6; al final, la cabeza de $\delta$ da el preestiramiento sabiendo todas las láminas. Unos 185\,000 parámetros (6 materiales).",
 ),
 "en": dict(babel="english",
   titulo="Ordinal chained network for the tube: one head per material and one for $\delta$",
   sub=r"trunk $141\to256\to128\to64$; head $j$: $(h,\,m_1\ldots m_{j-1})\to128\to64\to s_j$; 20 questions $P(n_j>k)=\sigma(s_j+b_{jk})$; $\delta$ head: $(h,\,\mathbf{m})\to64\to\delta$",
   entrada=r"\textbf{Input}\\[-1pt]\footnotesize 141 pressures\\[-1pt]\footnotesize\itshape $\lambda(P)$",
   capa=lambda k, u: rf"\textbf{{\shortstack{{Hidden\\layer {k}}}}}\\[-1pt]\footnotesize {u} neurons",
   reln=r"\footnotesize\itshape ReLU +\\[-1pt]\footnotesize\itshape layer norm.",
   tronco="Shared trunk (the curve)",
   cabeza=lambda j: (r"\textbf{Head 1}\\ $h\to128\to64\to$ ordinal" if j == 1 else r"\textbf{$\delta$ head}\\ $(h,\,m_1\ldots m_6)\to64\to\delta$" if j == 7 else
                     rf"\textbf{{Head {j}}}\\ $(h,\,m_1\ldots m_{{{j-1}}})\to128\to64\to$ ordinal"),
   prev=r"vector of decided laminae $\mathbf{m}$",
   pre=r"\textbf{Preprocessing}\\ smoothing: degree-6\\ polynomial in $P$\\ standardization",
   busq=r"\textbf{Check}\\ (optional)\\ refines $\delta$\\ (one variable);\\ tries $3^N$ neighbours\\ $n_j-1,\ n_j,\ n_j+1$\\ and keeps the one\\ with least $e_{\max}$",
   sal=r"\textbf{Tube design}\\ laminae $n_1\ldots n_6$\\ ($h_j=n_j\times0.025$\,mm)\\ and prestretch $\delta$",
   loss=r"\textbf{Loss: binary cross-entropy per question + $\delta$ error}\\[2pt] $\mathcal{L}=\sum_{j=1}^{6}\sum_{k=0}^{19}\mathrm{BCE}\bigl([n_j^{\mathrm{label}}>k],\,P(n_j>k)\bigr)+5\,(\delta-\delta^{\mathrm{label}})^2$\\[1pt] \footnotesize in training, each head receives in $\mathbf{m}$ the laminae of the label; standardized $\delta$",
   nota=r"Each head answers ``more than $k$ laminae?'' for $k=0\ldots19$; $n_j$ (0 to 20) is the number of ``yes''. It chooses knowing what the previous heads chose, so it can compensate between materials. It is used in order: material 1, then 2 knowing 1, and so on up to 6; finally, the $\delta$ head gives the prestretch knowing all the laminae. About 185\,000 parameters (6 materials).",
 ),
}
COL = ["blue!70!black/blue!15", "green!60!black/green!15", "orange!90!black/orange!20", "red!75!black/red!18"]
XS, YN = [0.0, 2.2, 4.4, 6.6], [1.25, 0.75, 0.25, -0.25, -0.75, -1.25]
YH = [4.2 - 1.4 * j for j in range(7)]           # cabezas
XB, XH, XO = 7.75, 10.35, 13.25                      # bus h, centro de las cabezas, salidas n_j

for lang, t in T.items():
    L = []
    a = L.append
    a(r"\documentclass[border=6pt]{standalone}")
    a(rf"\usepackage[utf8]{{inputenc}}\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}\usepackage[{t['babel']}]{{babel}}")
    a(r"\usepackage{amsmath,amssymb}\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,calc,decorations.pathreplacing}")
    a(r"\begin{document}\begin{tikzpicture}[font=\small, n/.style={circle,draw,thick,minimum size=5.2mm,inner sep=0pt},")
    a(r" box/.style={draw,rounded corners=2pt,fill=gray!8,align=center,inner sep=4pt,font=\footnotesize},")
    a(r" hd/.style={draw=violet!80!black,thick,rounded corners=3pt,fill=violet!6,align=center,inner sep=3pt,font=\scriptsize,text width=4.2cm,minimum height=0.95cm}]")
    a(rf"\node[font=\bfseries\normalsize] at (8.2,6.75) {{{t['titulo']}}};")
    a(rf"\node[gray] at (8.2,6.3) {{{t['sub']}}};")
    # aristas del tronco
    for k in range(3):
        for y1 in YN:
            for y2 in YN:
                a(rf"\draw[-{{Stealth[length=1.4mm]}},gray!45,very thin] ({XS[k] + 0.26:.2f},{y1:.2f})--({XS[k + 1] - 0.26:.2f},{y2:.2f});")
    for k, x in enumerate(XS):
        d, f = COL[k].split("/")
        for i, y in enumerate(YN):
            a(rf"\node[n,draw={d},fill={f}] (n{k}{i}) at ({x:.2f},{y:.2f}) {{}};")
        a(rf"\node[font=\tiny] at ({x:.2f},0.02) {{$\vdots$}};")
        if k == 0:
            a(rf"\node[align=center,anchor=south] at ({x:.2f},1.65) {{{t['entrada']}}};")
        else:
            a(rf"\node[align=center,anchor=south] at ({x:.2f},1.65) {{{t['capa'](k, (256, 128, 64)[k - 1])}\\[-1pt]{t['reln']}}};")
    a(rf"\draw[decorate,decoration={{brace,mirror,amplitude=5pt}},thick,gray] (1.9,-1.65)--(6.9,-1.65) node[midway,below=5pt,font=\footnotesize,gray!30!black] {{{t['tronco']}}};")
    # bus h
    for i, y in enumerate(YN):
        a(rf"\draw[gray!60,thin] ({XS[3] + 0.26:.2f},{y:.2f})--({XB:.2f},0);")
    a(rf"\draw[very thick,violet!60!black] ({XB:.2f},{YH[0]:.2f})--({XB:.2f},{YH[-1]:.2f});")
    a(rf"\fill[violet!60!black] ({XB:.2f},0) circle (1.6pt) node[left=2pt,font=\footnotesize] {{$h$}};")
    for j, y in enumerate(YH, 1):
        a(rf"\node[hd] (h{j}) at ({XH:.2f},{y:.2f}) {{{t['cabeza'](j)}}};")
        a(rf"\draw[-{{Stealth}},thick,violet!60!black] ({XB:.2f},{y:.2f})--(h{j}.west);")
        a(rf"\node[n,draw=violet!80!black,fill=violet!18] (o{j}) at ({XO:.2f},{y:.2f}) {{}};")
        a(rf"\node[right=0.5mm of o{j},font=\footnotesize] {{{'$n_{%d}$' % j if j < 7 else chr(36) + chr(92) + 'delta' + chr(36)}}};")
        a(rf"\draw[-{{Stealth}},thick] (h{j}.east)--(o{j}.west);")
    # realimentación: n_j entra a las cabezas siguientes (bus a la derecha que baja y entra por arriba de cada cabeza)
    XR = XO + 0.85
    for j in range(1, 7):
        a(rf"\draw[-{{Stealth[length=1.5mm]}},dashed,violet!70!black] (o{j}.south)--({XO:.2f},{YH[j - 1] - 0.62:.2f})--({XH + 1.35:.2f},{YH[j - 1] - 0.62:.2f})--(h{j + 1}.north -| {XH + 1.35:.2f},0);")
        a(rf"\node[font=\scriptsize,violet!70!black,below right=-1pt] at ({XO:.2f},{YH[j - 1] - 0.62:.2f}) {{$m_{{{j}}}$}};")
    a(rf"\draw[-{{Stealth[length=1.5mm]}},dashed,violet!70!black] ({XB - 0.1:.2f},{YH[0] + 1.0:.2f})--++(0.8,0) node[right,font=\scriptsize,violet!70!black] {{{t['prev']}: $m_j=n_j/20$ entra a las cabezas siguientes}};" if lang == "es" else rf"\draw[-{{Stealth[length=1.5mm]}},dashed,violet!70!black] ({XB - 0.1:.2f},{YH[0] + 1.0:.2f})--++(0.8,0) node[right,font=\scriptsize,violet!70!black] {{{t['prev']}: $m_j=n_j/20$ goes into the following heads}};")
    # preprocesamiento
    a(rf"\node[box,anchor=east] (pre) at (-0.75,0) {{{t['pre']}}};")
    a(r"\draw[-{Stealth},thick] (pre.east)--node[above,font=\tiny]{$X$}(n02.west);")
    # llave de salidas, búsqueda y salida
    a(rf"\draw[decorate,decoration={{brace,amplitude=7pt}},thick] ({XR + 0.15:.2f},{YH[0] + 0.3:.2f})--({XR + 0.15:.2f},{YH[-1] - 0.3:.2f});")
    a(rf"\node[box,fill=blue!6,draw=blue!50!black,thick] (bus) at ({XR + 2.35:.2f},0) {{{t['busq']}}};")
    a(rf"\draw[-{{Stealth}},thick] ({XR + 0.45:.2f},0)--(bus.west);")
    a(rf"\node[box,fill=green!8,draw=green!50!black,thick] (sal) at ({XR + 2.35:.2f},-3.7) {{{t['sal']}}};")
    a(r"\draw[-{Stealth},thick] (bus.south)--(sal.north);")
    # pérdida
    a(rf"\node[box,fill=green!8,draw=green!50!black,thick,font=\small] (loss) at (3.6,-5.75) {{{t['loss']}}};")
    a(r"\draw[-{Stealth},dashed,red!70!black,thick] (loss.north)--(3.6,-2.2);")
    a(rf"\draw[-{{Stealth}},dashed,blue!70!black,thick] (loss.east)-|(h7.south);")
    a(rf"\node[text width=19cm,align=center,font=\footnotesize,gray!30!black] at (8.2,-7.2) {{\hyphenpenalty=10000 {t['nota']}}};")
    a(r"\end{tikzpicture}\end{document}")
    open(f"red_cadena_tubo_{lang}.tex", "w").write("\n".join(L) + "\n")
print("ok")
