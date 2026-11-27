import json, sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
N=int(sys.argv[1]); S="/tmp/claude-0/-home-claude/1d17ef37-45c1-5d3c-b4ae-3f2d7b75bb24/scratchpad"; os.chdir(f"{S}/v58_{N}")
nb=json.load(open(f"curva_j_v58_{N}.ipynb")); cs=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g={}
for k in range(13): exec(compile(cs[k],f"c{k}","exec"),g)
Y=g["POB_Y"]; Zm=g["MINIMAX"]["41 sujetos"].astype(int); Zr=g["predecir_cadena"](Y).astype(int); Zb=g["mejorar_vecinos"](Y,Zr).astype(int)
f=lambda Z: g["err_pct"](Y, Z @ g["A"].T)
emM,erM=f(Zm); emR,erR=f(Zr); emB,erB=f(Zb)
P=g["POB"]
T=pd.DataFrame({"sujeto":P["id"].values,"conjunto":np.where(P["test"].values,"prueba","entrenamiento"),
 "RN láminas":[str(z.tolist()) for z in Zr],"RN Erms %":erR,"RN e_max %":emR,
 "RN±1 láminas":[str(z.tolist()) for z in Zb],"RN±1 Erms %":erB,"RN±1 e_max %":emB,
 "Minimax láminas":[str(z.tolist()) for z in Zm],"Minimax Erms %":erM,"Minimax e_max %":emM,
 "igual":(Zr==Zm).all(1)})
T.to_csv(f"tabla_41_RN_vs_minimax_v58_{N}.csv",index=False)
for c,D in T.groupby("conjunto"):
    print(c,len(D),{k:(int((D[k+" e_max %"]<=3).sum()),round(D[k+" e_max %"].median(),2),round(D[k+" e_max %"].max(),2)) for k in ("RN","RN±1","Minimax")})
print(T[T[["RN e_max %","RN±1 e_max %","Minimax e_max %"]].max(1)>3][["sujeto","conjunto","RN e_max %","RN±1 e_max %","Minimax e_max %"]])
