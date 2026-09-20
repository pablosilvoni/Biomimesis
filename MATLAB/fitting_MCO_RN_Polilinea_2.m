% =========================================================================
% EVALUACIÓN Y FITTING CON COEFICIENTES PREDICHOS POR LA RED NEURONAL
% =========================================================================
clc; clear; close all;

archivo_bases = 'Bases_Escaladas_Optimas.xlsx';
archivo_curva_j = 'Curva_J.xls'; % Nombre del archivo Excel con la Curva J

% Verificar la existencia del archivo de bases escaladas
if ~exist(archivo_bases, 'file')
    error('El archivo "%s" no existe. Ejecutá primero el script del PASO 2.', archivo_bases);
end

% --- 1. CARGA DE DATOS Y BASES ESCALADAS (4 RAMPAS) ---
opts = detectImportOptions(archivo_bases);
tabla_excel = readtable(archivo_bases, opts);

x_grid = tabla_excel.x;
A_scaled = table2array(tabla_excel(:, 2:end)); % Matriz de 4 bases escaladas phi_1 a phi_4

% --- 2. CARGA Y ADAPTACIÓN DE LA CURVA J DESDE ARCHIVO EXCEL ---
if exist(archivo_curva_j, 'file')
    % Cargar datos desde Curva_J.xls
    tabla_j = readtable(archivo_curva_j);
    
    % Extraer coordenadas x e y (columnas 1 y 2)
    x_j_datos = tabla_j{:, 1};
    y_j_datos = tabla_j{:, 2};
    
    % Definir f_curva_J a partir de los datos interpolados del Excel
    f_curva_J = @(x) interp1(x_j_datos, y_j_datos, x, 'linear', 'extrap');
else
    warning('El archivo "%s" no se encontró. Usando función anónima por defecto.', archivo_curva_j);
    % --- FUNCIÓN EXPLÍCITA ALTERNATIVA POR DEFECTO ---
    f_curva_J = @(x) 31.668803 * (exp(6.346177 * max(0, x - 1)) - 1);
end

% Evaluación directa de la función explícita/interpolada sobre x_grid
y_real = f_curva_J(x_grid);

% Aseguramos que no dé valores negativos por imprecisiones/extrapolación
y_real = max(y_real, 0);

% --- 3. COEFICIENTES PREDICHOS POR LA RED NEURONAL (V9) Y MCO ---
c_rn  = [0.0000000, 0.000000, 0.000000, 1.040772]'; % Predicción de la RN
c_ols = [0.000000, 0.0000000, 0.000000, 1.000000 ]'; % Óptimo MCO Restringido

% --- 4. RECONSTRUCCIÓN DE LAS CURVAS Y MÉTRICAS ---
y_pred_rn  = A_scaled * c_rn;
y_pred_ols = A_scaled * c_ols;

% Métricas de la Red Neuronal
r2_rn    = 1 - sum((y_real - y_pred_rn).^2) / sum((y_real - mean(y_real)).^2);
rmse_rn  = sqrt(mean((y_real - y_pred_rn).^2));
nrmse_rn = (rmse_rn / max(y_real)) * 100;

% Métricas del Óptimo MCO
r2_ols   = 1 - sum((y_real - y_pred_ols).^2) / sum((y_real - mean(y_real)).^2);
rmse_ols = sqrt(mean((y_real - y_pred_ols).^2));
nrmse_ols = (rmse_ols / max(y_real)) * 100;

% --- 5. SALIDA EN CONSOLA ---
fprintf('=========================================================================\n');
fprintf('       EVALUACIÓN DEL FITTING: RED NEURONAL VS ÓPTIMO MCO\n');
fprintf('=========================================================================\n');
fprintf('Suma de Coeficientes RN  : %.4f (<= 15.0)\n', sum(c_rn));
fprintf('-------------------------------------------------------------------------\n');
fprintf('RED NEURONAL (V9): R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f%%\n', r2_rn, rmse_rn, nrmse_rn);
fprintf('ÓPTIMO MCO       : R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f%%\n', round(r2_ols, 3), rmse_ols, nrmse_ols);
fprintf('=========================================================================\n\n');

% --- 6. VISUALIZACIÓN GRÁFICA CON FORMATO LATEX ---
texto_leyenda_fx  = sprintf('Base 4');
texto_leyenda_rn  = sprintf('Red Neuronal ($R^2 = %.5f$)', r2_rn);
texto_leyenda_mco = sprintf('Minimos Cuadrados ($R^2 = %.5f$)', round(r2_ols, 3));

figure('Name', 'Fitting Red Neuronal vs Función Real', 'Color', 'w');

% Curva objetivo (Curva J)
plot(x_grid, y_real, 'b', 'LineWidth', 2.0, 'DisplayName', texto_leyenda_fx); hold on;

% Si existe el archivo, mostramos también los puntos discretos del Excel

%if exist(archivo_curva_j, 'file')
%    plot(x_j_datos, y_j_datos, 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'DisplayName', 'Puntos Excel');
%end

% Ajustes
plot(x_grid, y_pred_ols, 'r-', 'LineWidth', 1.8, 'DisplayName', texto_leyenda_mco);
plot(x_grid, y_pred_rn, 'Color', [0.7, 0.3, 0.5], 'LineWidth', 1.8, 'LineStyle', '-.', 'DisplayName', texto_leyenda_rn);

xlabel('$x$', 'Interpreter', 'latex'); 
ylabel('$y(x)$', 'Interpreter', 'latex');
title('Fitting: f(x) vs Minimos Cuadrados vs RN', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex'); 
grid on;