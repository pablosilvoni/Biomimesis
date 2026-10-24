# ==============================================================================
# CÉLULA TÍPICA DE PREPROCESAMIENTO Y NORMALIZACIÓN (Ej. Célula 4 o 5)
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Instanciar los escaladores
sx = StandardScaler()  # Para las entradas (X_log)
sy = StandardScaler()  # Para las salidas (Y_coef)

# 2. Ajustar y transformar entradas y salidas
X_scaled = sx.fit_transform(X_log)
Y_scaled = sy.fit_transform(Y_coef)

# 3. Partición de entrenamiento y prueba (Train / Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

print(f"✅ Datos normalizados y divididos:")
print(f"   • X_train shape: {X_train.shape}")
print(f"   • y_train shape: {y_train.shape}")
print(f"   • sy.mean_ (Media de coeficientes): {sy.mean_}")
print(f"   • sy.scale_ (Escala de coeficientes): {sy.scale_}")