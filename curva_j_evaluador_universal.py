# %% [markdown]
# # EVALUADOR UNIVERSAL DE CURVAS J (PINN V5 vs. ÓPTIMO DIRECTO)
# ### Desarrollado por Gemini Notebook para Pablo
# 
# Este script interactivo para Jupyter Notebook permite evaluar cualquier curva J del usuario 
# cargada desde un archivo de Excel o definida manualmente por coordenadas de puntos.
# El evaluador proyectará la curva sobre tus **5 funciones base personalizables** (leídas de Excel) 
# y comparará la reconstrucción de tu **Red Neuronal PINN V5** contra el solver de mínimos cuadrados directos (OLS).
# 
# ### 🚀 Características Principales:
# 1. **Carga Dinámica de Bases:** Puedes subir tu propio archivo Excel con las 5 curvas base (malla $x$ en col. 0, bases en col. 1 a 5).
# 2. **Doble Entrada de Usuario:** Elige cargar una curva experimental desde Excel o ingresar puntos de control manualmente en una lista.
# 3. **Interpolación Inteligente:** Remuestrea automáticamente los puntos del usuario a la malla de la red (201 puntos) usando interpolación lineal (quebrada) o PCHIP (suave/monótona).
# 4. **Análisis de Errores Avanzado:** Calcula métricas científicas ($R^2$, RMSE, MAE, MaxAE) y dibuja el gráfico de NRMSE% local punto a punto referido a $y_{\max}$.

# %%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# %matplotlib inline

# ==============================================================================
# ── 1. CONFIGURACIÓN DEL EVALUADOR ──────────────────────────────────────────
# ==============================================================================

# A. Modo de entrada de la curva J a evaluar:
# - 'excel': Lee el archivo Excel especificado abajo.
# - 'puntos': Utiliza la lista de puntos manuales PUNTOS_USUARIO.
MODO_ENTRADA = 'excel'

# B. Tipo de interpolación (solo aplicable si MODO_ENTRADA = 'puntos'):
# - 'lineal': Une los puntos con segmentos rectos (curva quebrada pura).
# - 'pchip': Une los puntos con una curva cúbica monótona suave sin sobrepasos.
TIPO_INTERPOLACION_PUNTOS = 'lineal'  

# C. Ruta del archivo Excel con la curva experimental (si MODO_ENTRADA = 'excel')
ARCHIVO_EXCEL_CURVA_J = "Curva_J.xls"

# D. Lista de puntos de control (x, y) definidos por el usuario (si MODO_ENTRADA = 'puntos')
# Se garantiza que x se encuentre en [0.0, 2.0] e y dentro de la envolvente biomimética
PUNTOS_USUARIO = [
    (0.00, 30.0),
    (0.40, 75.0),
    (0.43, 85.0),
    (0.92, 195.0),
    (1.33, 1800.0),
    (1.68, 5600.0),
    (2.00, 14000.0)
]

# E. Archivos del modelo y curvas base de la simulación
ARCHIVO_CURVAS_BASE = "funciones_base_5_corregido.xlsx"
ARCHIVO_MODELO_KERAS = "modelo_rn_curva_j_desde_cero_v8.keras"
ARCHIVO_SCALERS_PKL  = "scalers_curva_j_desde_cero_v8.pkl"

# F. Parámetros del espacio funcional V5 (Biomímesis)
MAX_COEF = 25.0                 # Cota superior de Biomímesis para c_i
Y_MIN_ABS = -2.0                # Suelo físico del espacio funcional
RANGO_ALPHA_ENVOLVENTE = (1.8, 3.2) # Exponentes de la envolvente de entrenamiento

# %% [markdown]
# ### 2. Funciones de Carga de Archivos e Inferencia

# %%
def cargar_archivos_sistema():
    """Carga las bases optimizadas, el modelo entrenado y los escaladores StandardScaler."""
    print("📂 Cargando base funcional optimizada...")
    if not os.path.exists(ARCHIVO_CURVAS_BASE):
        raise FileNotFoundError(f"No se encontró el archivo de curvas base '{ARCHIVO_CURVAS_BASE}'.")
    
    df_base = pd.read_excel(ARCHIVO_CURVAS_BASE, header=None)
    x_base = df_base.iloc[:, 0].values.astype(np.float64)
    A_base = np.column_stack([df_base.iloc[:, i].values for i in range(1, 6)]).astype(np.float64)
    
    print("🧠 Cargando modelo de Red Neuronal y Escaladores V5...")
    if not os.path.exists(ARCHIVO_MODELO_KERAS) or not os.path.exists(ARCHIVO_SCALERS_PKL):
        print("⚠️  ¡Atención! No se encontraron los archivos del modelo o scalers de la corrida local.")
        print("   Por favor asegúrate de entrenar primero el modelo V5 o colocar los archivos en esta carpeta.")
        return x_base, A_base, None, None, None, None
    
    modelo = load_model(ARCHIVO_MODELO_KERAS, compile=False)
    with open(ARCHIVO_SCALERS_PKL, "rb") as f:
        # Desempaquetado seguro de 3 elementos (sx, sy, offset_log)
        contenido = pickle.load(f)
        sx = contenido[0]
        sy = contenido[1]
        offset_log = contenido[2] if len(contenido) > 2 else 2.1
        
    print("✅ Todos los archivos del sistema se han cargado exitosamente.")
    return x_base, A_base, modelo, sx, sy, offset_log

# %% [markdown]
# ### 3. Procesamiento e Interpolación de la Entrada del Usuario

# %%
def generar_curva_objetivo(x_base):
    """Obtiene o construye la curva experimental del usuario y la mapea a la malla base."""
    if MODO_ENTRADA == 'excel':
        print(f"📖 Cargando curva experimental desde '{ARCHIVO_EXCEL_CURVA_J}'...")
        if not os.path.exists(ARCHIVO_EXCEL_CURVA_J):
            raise FileNotFoundError(f"No se encontró el archivo Excel de curva '{ARCHIVO_EXCEL_CURVA_J}'.")
        df_curva = pd.read_excel(ARCHIVO_EXCEL_CURVA_J, header=None)
        x_user = df_curva.iloc[:, 0].values.astype(np.float64)
        y_user = df_curva.iloc[:, 1].values.astype(np.float64)
        
        # Mapeo por interpolación lineal a la grilla de x_base (201 puntos)
        y_objetivo = np.interp(x_base, x_user, y_user)
        puntos_ctrl = list(zip(x_user, y_user))
        
    else:
        print(f"✏️  Generando curva desde puntos manuales ({TIPO_INTERPOLACION_PUNTOS})...")
        puntos_ordenados = sorted(PUNTOS_USUARIO, key=lambda p: p[0])
        x_ctrl = np.array([p[0] for p in puntos_ordenados], dtype=np.float64)
        y_ctrl = np.array([p[1] for p in puntos_ordenados], dtype=np.float64)
        
        if TIPO_INTERPOLACION_PUNTOS == 'pchip':
            y_objetivo = PchipInterpolator(x_ctrl, y_ctrl)(x_base)
        else: # 'lineal'
            y_objetivo = np.interp(x_base, x_ctrl, y_ctrl)
        puntos_ctrl = puntos_ordenados
            
    # Clip de seguridad para que la curva no viole el suelo físico
    y_objetivo = np.maximum(y_objetivo, Y_MIN_ABS)
    return y_objetivo, puntos_ctrl

# %% [markdown]
# ### 4. Ejecución de Ajustes: OLS vs. PINN V5

# %%
def resolver_ajustes(A_base, b_target, modelo, sx, sy, offset_log):
    """Resuelve la regresión por mínimos cuadrados acotados y por inferencia PINN."""
    # 1. Óptimo directo vía mínimos cuadrados acotados lsq_linear [0, MAX_COEF]
    res_lsq = lsq_linear(A_base, b_target, bounds=(0, MAX_COEF))
    coef_ols = res_lsq.x
    y_ols = A_base @ coef_ols
    
    # 2. Inferencia de la Red Neuronal (PINN)
    if modelo is not None:
        # Preprocesamiento logarítmico seguro
        X_log = np.log(b_target + offset_log).reshape(1, -1)
        Xn = sx.transform(X_log)
        
        Yn_pred = modelo.predict(Xn, verbose=0)
        coef_rn = sy.inverse_transform(Yn_pred)[0]
        # Clipping físico estricto de Biomímesis [0, MAX_COEF]
        coef_rn = np.clip(coef_rn, 0.0, MAX_COEF)
        y_rn = A_base @ coef_rn
    else:
        coef_rn, y_rn = None, None
        
    return coef_ols, y_ols, coef_rn, y_rn

# %% [markdown]
# ### 5. Reporte de Métricas Científicas e Inferencia de Gráficos

# %%
def calcular_metricas(real, aprox):
    rmse = np.sqrt(mean_squared_error(real, aprox))
    return {
        "R2": r2_score(real, aprox),
        "RMSE": rmse,
        "MAE": mean_absolute_error(real, aprox),
        "MaxAE": np.max(np.abs(real - aprox)),
        "NRMSE": (rmse / real.max()) * 100.0
    }

def graficar_resultados(x, b_real, b_rn, b_ols, pts_ctrl):
    """Renderiza los gráficos de ajuste, residuos y NRMSE% punto a punto."""
    y_max = b_real.max()
    cota_inf_env = np.ones_like(x) * Y_MIN_ABS
    cota_sup_env = 30.0 * np.exp(3.2 * x)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel 1: Ajuste General e Envolvente
    axes[0].fill_between(x, cota_inf_env, cota_sup_env, color='#e5e7eb', alpha=0.5, label='Envolvente Física')
    axes[0].plot(x, b_real, 'k-', linewidth=2.5, label='Curva Objetivo')
    if b_ols is not None:
        axes[0].plot(x, b_ols, 'b:', linewidth=2.0, label='Óptimo Directo (OLS)')
    if b_rn is not None:
        axes[0].plot(x, b_rn, 'r--', linewidth=2.0, label='Reconstrucción PINN V5')
    if pts_ctrl is not None:
        px, py = zip(*pts_ctrl)
        axes[0].scatter(px, py, color='black', zorder=5, s=35, label='Puntos Control')
        
    axes[0].set_xlabel('Malla x', fontsize=11)
    axes[0].set_ylabel('Amplitud y', fontsize=11)
    axes[0].set_title('Curva Objetivo vs. Reconstrucciones', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Gráfica de Residuos Locales
    if b_ols is not None:
        axes[1].plot(x, b_real - b_ols, 'b:', linewidth=2.0, label='Residuo OLS (Geométrico)')
    if b_rn is not None:
        axes[1].plot(x, b_real - b_rn, 'r-', linewidth=1.8, label='Residuo PINN V5')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.6)
    axes[1].set_xlabel('Malla x', fontsize=11)
    axes[1].set_ylabel('Error Residual (Real - Predicho)', fontsize=11)
    axes[1].set_title('Distribución de Residuos Locales', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Error NRMSE% Punto a Punto
    if b_ols is not None:
        nrmse_ols = np.abs(b_real - b_ols) / y_max * 100.0
        axes[2].plot(x, nrmse_ols, 'b:', linewidth=2.0, label='Mínimos Cuadrados (OLS)')
    if b_rn is not None:
        nrmse_rn = np.abs(b_real - b_rn) / y_max * 100.0
        axes[2].plot(x, nrmse_rn, 'r-', linewidth=1.8, label='Red Neuronal PINN (V5)')
    axes[2].axhline(1.0, color='gray', linestyle=':', alpha=0.8, label='Umbral Tolerancia 1%')
    axes[2].set_xlabel('Malla x', fontsize=11)
    axes[2].set_ylabel('NRMSE Local (% de $y_{\\max}$)', fontsize=11)
    axes[2].set_title('Error NRMSE% Local Punto a Punto', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9, loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 6. Bloque de Ejecución Principal

# %%
def ejecutar_pipeline_evaluacion():
    # A. Carga de archivos de sistema
    x_base, A_base, model, sx, sy, offset_log = cargar_archivos_sistema()
    
    # B. Construcción de curva de usuario
    y_target, pts_ctrl = generar_curva_objetivo(x_base)
    
    # C. Resolución de ajustes
    c_ols, y_ols, c_rn, y_rn = resolver_ajustes(A_base, y_target, model, sx, sy, offset_log)
    
    # D. Reportería en Consola de Coeficientes
    print("\n" + "="*80)
    print("🎯 REPORTE COMPARATIVO DE COEFICIENTES (BIOMÍMESIS)")
    print("="*80)
    print(f"Coeficiente | {'Óptimo Directo (OLS)':<22} | {'Red Neuronal PINN V5':<22} | {'Diferencia':<12}")
    print("-" * 80)
    for m in range(5):
        val_ols = c_ols[m]
        val_rn = c_rn[m] if c_rn is not None else np.nan
        diff = val_ols - val_rn if c_rn is not None else np.nan
        print(f"c{m+1:<10} | {val_ols:<22.6f} | {val_rn:<22.6f} | {diff:<12.6f}")
    
    # E. Reportería de Métricas de Precisión
    m_ols = calcular_metricas(y_target, y_ols)
    print("\n" + "="*80)
    print("📊 MÉTRICAS DE PRECISIÓN DE AJUSTE GLOBAL")
    print("="*80)
    print(f"Métrica      | {'Óptimo Directo (OLS)':<22} | {'Red Neuronal PINN V5':<22}")
    print("-" * 80)
    m_rn = calcular_metricas(y_target, y_rn) if y_rn is not None else None
    for k in ["R2", "RMSE", "MAE", "MaxAE", "NRMSE"]:
        val_ols = m_ols[k]
        val_rn = m_rn[k] if m_rn is not None else np.nan
        suffix = "%" if k == "NRMSE" else ""
        print(f"{k:<12} | {val_ols:<22.6f}{suffix} | {val_rn:<22.6f}{suffix}")
    
    # F. Visualización Gráfica Interactiva
    graficar_resultados(x_base, y_target, y_rn, y_ols, pts_ctrl)

# Ejecutamos el evaluador
ejecutar_pipeline_evaluacion()
# %%
