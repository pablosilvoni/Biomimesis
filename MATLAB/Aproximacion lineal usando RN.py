import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Generar datos correctamente
x = torch.linspace(0.2, 1, 100).reshape(-1, 1)  # Tensor de 100x1

# Definir funciones base CORRECTAMENTE (asegurando compatibilidad con PyTorch)
def y(x): return 9.8416 * torch.exp(4.1608 * x) - 27.5991
def g(x): return 0.788 * torch.exp(5.9726 * x) + 7.1028
def h(x): return 4.7143 * torch.exp(4.5321 * x) + 16.4298

# 2. Crear matriz de características (asegurar dimensiones correctas)
features = torch.cat([
    y(x),
    g(x), 
    h(x)
], dim=1)  # Resultado debería ser 100x3

# 3. Definir modelo (capa lineal simple)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)  # 3 entradas, 1 salida
        
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 4. Configurar entrenamiento
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Datos objetivo
y_true = 30 * torch.exp(3 * x)

# 5. Entrenamiento
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, y_true)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 6. Obtener coeficientes
a, b, c = model.linear.weight[0].detach().numpy()
print(f"\nCoeficientes finales:\na = {a:.6f}\nb = {b:.6f}\nc = {c:.6f}")

# 7. Visualización
with torch.no_grad():
    predictions = model(features)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_true.numpy(), 'b-', label='Función real')
plt.plot(x.numpy(), predictions.numpy(), 'r--', label='Aproximación')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Aproximación con Red Neuronal')
plt.grid(True)
plt.show()

