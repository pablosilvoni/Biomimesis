import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETROS MATERIALES - Sommer & Holzapfel (2012), Tabla 1 (CCA Media-Íntima, VI r)
# ==============================================================================
mu = 118.9       # [kPa] Módulo de cizalladura de la matriz isotrópica
k1 = 36.1        # [kPa] Rigidez equivalente de fibras de colágeno
k2 = 2.8         # [-] Factor de endurecimiento no lineal
phi_deg = 8.5    # [grados] Ángulo medio de fibras
phi = np.radians(phi_deg)
rho = 1.0        # [-] Parámetro de dispersión de fibras (alineación perfecta)

# GEOMETRÍA LIBRE DE CARGA
R_i = 3.10       # [mm] Radio interno

H_0 = 0.36       # [mm] Espesor de pared
R_o = R_i + H_0  # [mm] Radio externo (3.46 mm)
lambda_z = 1.00  # Pre-estiramiento axial

def calcular_presion_intraluminal(lambda_theta_i, n_puntos=200):
    """
    Calcula la presión intraluminal p_i mediante la integración exacta 
    de la diferencia de esfuerzos de Cauchy a través del espesor deformado.
    """
    # Malla radial en configuración libre de carga R ∈ [R_i, R_o]
    R_grid = np.linspace(R_i, R_o, n_puntos)
    
    # Radio interno deformado r_i
    r_i = R_i * lambda_theta_i
    
    # Campo de radios deformados r(R) por incompresibilidad (J = 1)
    r_grid = np.sqrt(r_i**2 + (R_grid**2 - R_i**2) / lambda_z)
    
    # Estiramientos locales a lo largo del espesor de pared
    l_theta = r_grid / R_grid
    l_r = 1.0 / (l_theta * lambda_z)
    
    # Invariantes isocóricos de deformación
    I1 = l_r**2 + l_theta**2 + lambda_z**2
    I4 = l_theta**2 * (np.cos(phi)**2) + lambda_z**2 * (np.sin(phi)**2)
    
    # Derivadas parciales de la función de energía de deformación HGO
    Q = k2 * ((1.0 - rho) * (I1 - 3.0)**2 + rho * (I4 - 1.0)**2)
    dW_dI1 = 0.5 * mu + k1 * (1.0 - rho) * (I1 - 3.0) * np.exp(Q)
    dW_dI4 = k1 * rho * (I4 - 1.0) * np.exp(Q)
    
    # Diferencia de esfuerzos principales de Cauchy (sigma_theta - sigma_r)
    d_sigma = 2.0 * (l_theta**2 - l_r**2) * dW_dI1 + 2.0 * l_theta**2 * (np.cos(phi)**2) * dW_dI4
    
    # Integrando de la ecuación de equilibrio radial: (sigma_theta - sigma_r) / r * dr
    integrand = (d_sigma / r_grid) * (R_grid / (lambda_z * r_grid))
    
    # Integración numérica por regla del trapecio a lo largo del espesor
    return trapezoid(integrand, R_grid)

if __name__ == '__main__':
    # Evaluamos en el rango experimental lambda_theta ∈ [1.00, 1.41]
    lt_array = np.linspace(1.00, 1.41, 150)
    pi_array_kPa = np.array([calcular_presion_intraluminal(lt) for lt in lt_array])
    pi_array_mmHg = pi_array_kPa * 7.50062

    # Generación del gráfico formal
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

    ax1.plot(lt_array, pi_array_kPa, color='#1f77b4', linewidth=2.5, label=r'Modelo HGO Exacto (Pared Gruesa)')
    ax1.set_xlabel(r'Estiramiento Circunferencial $\lambda_\theta$ [-]', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel(r'Presión Intraluminal $p_i$ [kPa]', fontsize=12, fontweight='bold', color='#1f77b4', labelpad=8)
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlim([1.00, 1.45])

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Presión Intraluminal $p_i$ [mmHg]', fontsize=12, fontweight='bold', color='#2ca02c', labelpad=8)
    ax2.set_ylim([0, 25 * 7.50062])
    ax2.tick_params(axis='y', labelcolor='#2ca02c', labelsize=11)
    ax2.grid(False)

    plt.title('Figura 6a: Sommer & Holzapfel (2012) - CCA Media-Íntima (Donante VI r)\nFormulación Constitutiva Exacta', fontsize=12, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('sommer_fig6a_formulacion_exacta.png', dpi=300)
    print("Gráfico generado exitosamente.")
