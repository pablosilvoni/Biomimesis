% =========================================================================
% EVALUACIÓN Y FITTING CON COEFICIENTES PREDICHOS POR LA RED NEURONAL
% =========================================================================
clc; clear; close all;

archivo_bases = 'Bases_Escaladas_Optimas.xlsx';

r2_mco = 0.998826;
r2_rn = 0.998606;

% Verificar la existencia del archivo de bases escaladas
if ~exist(archivo_bases, 'file')
    error('El archivo "%s" no existe. Ejecutá primero el script del PASO 2.', archivo_bases);
end

% --- 1. CARGA DE DATOS Y BASES ESCALADAS (4 RAMPAS) ---
opts = detectImportOptions(archivo_bases);
tabla_excel = readtable(archivo_bases, opts);

x_grid = tabla_excel.x;
A_scaled = table2array(tabla_excel(:, 2:end)); % Matriz de 4 bases escaladas phi_1 a phi_4

% --- 2. PARÁMETROS Y CURVA REAL ---
K = 30.0; alfa = 2;
y_real = K * exp(alfa * x_grid);

% --- 3. COEFICIENTES PREDICHOS POR LA RED NEURONAL (V9) ---
c_rn = [1.107911, 0.344708, 0.169004, 0.139995]'; % Predicción de la RN

% Coeficientes de referencia del Óptimo MCO Restringido
c_ols = [1.098984, 0.339893, 0.186425, 0.137456]';

% --- 4. RECONSTRUCCIÓN DE LAS CURVAS Y MÉTRICAS ---
y_pred_rn  = A_scaled * c_rn;
y_pred_ols = A_scaled * c_ols;

% Métricas de la Red Neuronal
r2_rn    = 1 - sum((y_real - y_pred_rn).^2) / sum((y_real - mean(y_real)).^2);
rmse_rn  = sqrt(mean((y_real - y_pred_rn).^2));
nrmse_rn = (rmse_rn / max(y_real)) * 100;

% Métricas del Óptimo MCO
r2_ols    = 1 - sum((y_real - y_pred_ols).^2) / sum((y_real - mean(y_real)).^2);
rmse_ols  = sqrt(mean((y_real - y_pred_ols).^2));
nrmse_ols = (rmse_ols / max(y_real)) * 100;

% --- 5. SALIDA EN CONSOLA ---
fprintf('=========================================================================\n');
fprintf('       EVALUACIÓN DEL FITTING: RED NEURONAL VS ÓPTIMO MCO\n');
fprintf('=========================================================================\n');
fprintf('Suma de Coeficientes RN  : %.4f (<= 15.0)\n', sum(c_rn));
fprintf('-------------------------------------------------------------------------\n');
fprintf('RED NEURONAL (V9): R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f%%\n', r2_rn, rmse_rn, nrmse_rn);
fprintf('ÓPTIMO MCO (c_i=3): R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f%%\n', r2_ols, rmse_ols, nrmse_ols);
fprintf('=========================================================================\n\n');

% --- 6. VISUALIZACIÓN GRÁFICA CON FORMATO LATEX (2 FIGURAS SEPARADAS) ---

% Texto de la leyenda formateado para LaTeX
texto_leyenda_rn = sprintf('Red Neuronal($R^2 = %.5f$)', r2_rn);
texto_leyenda_mco = sprintf('Minimos Cuadrados($R^2 = %.5f$)', r2_mco);

% Figura 1: Ajuste de la curva
figure('Name', 'Fitting Red Neuronal vs Funcion Real', 'Color', 'w');
plot(x_grid, y_real, 'b', 'LineWidth', 2.0, 'DisplayName', '$y(x) = 30e^{2x}$'); hold on;
plot(x_grid, y_pred_ols, 'r--', 'LineWidth', 1.8, 'DisplayName', texto_leyenda_mco);
plot(x_grid, y_pred_rn, 'Color', [0.7, 0.3, 0.5], 'LineWidth', 1.8,'LineStyle', '-.', 'DisplayName', texto_leyenda_rn);
xlabel('$x$', 'Interpreter', 'latex'); 
ylabel('$y(x)$', 'Interpreter', 'latex');
title('Fitting: f(x) vs Minimos Cuadrados vs RN', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex'); 
grid on;

