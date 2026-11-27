S = "/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad/arten"
s = open(f"{S}/article_final_en.tex").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
i = s.index(r"\subsection{Thin wall and thick wall}"); j = s.index("%=====================================================================\n\\section{Discussion}")
s = s[:i] + open(f"{S}/sec4_body_en.tex").read() + "\n" + open(f"{S}/sec4_res_en.tex").read() + "\n" + s[j:]
rep(r"""whereas with the thick-walled model 40--41 of 41 do. Design becomes a nonlinear integer problem, solved in three steps: initialization, nonlinear least squares, and minimax integer search against the fitted curve. There, a network used as a starting point avoids the local minima that, with measured curves, lead to unusable designs (worst error from 468.7\,\% down to 2.3\,\%).""",
r"""whereas with the thick-walled model all 41 do. Design becomes a nonlinear integer problem; as a reference it is solved in three steps (initialization, nonlinear least squares and integer minimax), which meet the target for all 41 subjects. The same chained network, with one more head for the prestretch, gives the laminae of each layer in one evaluation: with a $\pm1$ lamina check it meets the target for all 41 subjects with 6 materials and for 40 with 5, and with noisy measured curves it meets it for more subjects than the minimax, with a smaller worst case.""")
rep(r"and (v) a three-step procedure in which a network acts as a starting point.", r"and (v) the same chained network applied to the thick-walled tube, compared with a three-step optimization procedure.")
i = s.index(r"In the thick-wall case the fit is iterative, local and dependent on the starting point:"); j = s.index("\n", i)
s = s[:i] + r"""In the thick-walled tube the same network, with the same ordinal output, gives the laminae of each layer and the prestretch; with the $\pm1$ lamina check it matches the three-step procedure on clean curves and outperforms it on noisy curves, because it chooses among designs seen in training and does not chase the noise. This is the idea of amortized optimization~\cite{amos}: a model learned on many solved cases gives the solution of a new one, and a short search finishes it. The network does not replace the minimax as the reference for checking a design.""" + s[j:]
rep(r"a single noise realization per subject and case was used.", r"a single noise realization per subject and case was used. To respect the 20-lamina cap the stiffest materials were made stiffer, which must be verified with real materials.")
rep(r"""with the thick-walled model, 40--41 of 41 do. The problem is a nonlinear integer one.""", r"""with the thick-walled model, all 41 do. The problem is a nonlinear integer one.""")
i = s.index(r"\item The three-step procedure ---initialization"); j = s.index(r"\end{enumerate}", i)
s = s[:i] + r"""\item With 0.025\,mm laminae and at most 20 per material, the three-step procedure ---initialization, nonlinear least squares and integer minimax--- meets the target for all 41 subjects, with 6 and with 5 materials, in less than one second per subject.
\item The same chained network, trained for the tube, gives the laminae of each layer in one evaluation; with refined $\delta$ and a $\pm1$ lamina check it meets the target for all 41 subjects with 6 materials and for 40 with 5. With noisy measured curves it meets it for more subjects than the minimax (29 versus 24 with 6 materials) and with a smaller worst case. A single network thus serves both levels.
""" + s[j:]
rep("\\section{Training of the neural networks}", open(f"{S}/annex_calc_en.tex").read() + "\n\\section{Training of the neural networks}")
rep(r"and the thick-wall network (block A4), each with 6 and with 5 materials.", r"and the tube chained network, which maps the curve $\lambda(P)$ to the laminae of each layer and the prestretch (Section~\ref{sec:aplicacion}), each with 6 and with 5 materials.")
rep("{curvas_aprendizaje_en.png}", "{curvas_aprendizaje_v59_en.png}")
rep(r"bottom: thick-wall network (mean squared error of the standardized design).", r"bottom: tube network (the same sum plus 5 times the mean squared error of the standardized $\delta$). For the first-level network the final weights are kept; for the tube network, those of the validation minimum.")
rep(r"the chained network uses the binary cross-entropy summed over the questions and the materials (in nats) and the thick-wall network the mean squared error of the standardized design, so the two sets of values are not comparable with each other.",
    r"both networks use the binary cross-entropy summed over the questions and the materials (in nats); the tube network also adds 5 times the mean squared error of the standardized $\delta$, which is below 0.02.")
rep("Thick wall, 6 mat. & 11\\,220 / 1530 / 2250 & 425 / 305 & 5 & 0.342 & 0.462 & 0.484 & 1.42\\\\", "Tube chained network, 6 mat. & 11\\,220 / 1530 / 2250 & 288 / 228 & 5 & 14.91 & 21.72 & 22.57 & 1.51\\\\")
rep("Thick wall, 5 mat. & 11\\,220 / 1530 / 2250 & 629 / 509 & 4 & 0.153 & 0.227 & 0.250 & 1.64\\\\", "Tube chained network, 5 mat. & 11\\,220 / 1530 / 2250 & 409 / 349 & 6 & 8.88 & 14.61 & 14.69 & 1.65\\\\")
i = s.index(r"\paragraph{Thick-wall network.}"); j = s.index(r"\clearpage", i)
s = s[:i] + r"""\paragraph{Tube chained network.} Validation departs from training within the first 30 epochs and levels off, while the training loss keeps decreasing. Early stopping keeps the weights at the validation minimum, and the test loss stays close to the validation loss (22.57 versus 21.72 and 14.69 versus 14.61): the network does not degrade on new curves. The loss is higher than at the first level, and the reason lies in the problem and not in the network: in the tube there are even more integer designs with almost the same curve.

\emph{Conditioning.} For 60 arteries of the physiological families (with the moduli not yet recalibrated), the Jacobian matrix $\mathbf{D}$ of the graft curve with respect to the design $\theta=(h_1\ldots h_N,\delta)$ was computed by finite differences, with each parameter in standard deviations and the curve in \% of its range. With 6 materials, in the most sensitive direction the curve changes by 8.5 times its range per deviation; in the least sensitive one, by only 1\,\%. The condition number has a median of 900 with 6 materials and 300 with 5, versus $\mathrm{cond}(A)=28$ at the first level. The least sensitive direction is typically more thickness of one material and less of its neighbour. From the curve, those designs are indistinguishable for any method, and the network cannot guess which of them is the label.

\emph{The measure that matters.} The network gives exactly the minimax laminae for 7\,\% of the synthetic test arteries with 6 materials and 26\,\% with 5; even so, with the $\pm1$ check, its design meets the target for 294 and 297 of 300. As at the first level, the network is judged by the error on the curve and not by matching the minimax.

""" + s[j:]
rep(r"""\section{Laminae and errors of each curve}\label{sec:tablas}
Integer laminae of each material and errors of the chained network, of the chained network with the $\pm1$ search and of the minimax, for the 41 Masson subjects and for the 63 external curves inside $y_1$--$y_2$, with 6 and with 5 materials. Laminae are listed from material 1 to $N$.""",
r"""\section{Laminae and errors of each curve}\label{sec:tablas}
\textbf{First level, stress--stretch curve} (Tables~\ref{tab:suj6} to \ref{tab:ext5}): integer laminae of each material and errors of the chained network, of the chained network with the $\pm1$ search and of the minimax, for the 41 Masson subjects and for the 63 external curves inside $y_1$--$y_2$, with 6 and with 5 materials. \textbf{Tube level, pressure--diameter curve} (Tables~\ref{tab:tubo6} and \ref{tab:tubo5}): laminae of each layer, prestretch and errors of the chained network with the $\pm1$ check and of the three-step minimax, for the 41 subjects. Laminae are listed from material 1 to $N$.""")
rep("\\input{tab_ext5_en.tex}", "\\input{tab_ext5_en.tex}\n\\input{tab_tubo6_en.tex}\n\\input{tab_tubo5_en.tex}")
rep(r"\date{November 9, 2026}", r"\date{November 11, 2026}")
open(f"{S}/article_final_en.tex", "w").write(s)
print("ok")
