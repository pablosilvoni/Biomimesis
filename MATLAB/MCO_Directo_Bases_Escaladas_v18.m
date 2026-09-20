% =========================================================================
% ESCALADO Y AJUSTE MINIMOS CUADRADOS DESDE EXCEL DE BASES ORIGINALES (V18)
% =========================================================================
clc; clear; close all;

archivo_origen = 'Bases_Originales_MCO.xlsx';

% Verificar que el archivo existe
if ~exist(archivo_origen, 'file')
    error('El archivo "%s" no existe. Ejecutá primero el script de optimización MCO.', archivo_origen);
end

% --- 1. LECTURA DE DATOS DESDE EXCEL ---
opts_bases = detectImportOptions(archivo_origen, 'Sheet', 'BasesOriginales');
tabla_bases = readtable(archivo_origen, opts_bases);

opts_nodos = detectImportOptions(archivo_origen, 'Sheet', 'NodosOptimos');
tabla_nodos = readtable(archivo_origen, opts_nodos);

x_grid = tabla_bases.x;

% Tomamos desde la columna 3 en adelante (descartando x y g_0) para tener 4 rampas
A_raw = table2array(tabla_bases(:, 3:end)); 
nodos = table2array(tabla_nodos);

% --- 2. PARÁMETROS Y DATOS DE LA FUNCIÓN ---
K = 30.0; alfa = 3.2; MAX_SUM = 15.0;
y_real = K * exp(alfa * x_grid);

% --- 3. CONSTRUCCIÓN Y ESCALADO DE BASES ---
% Ajuste NNLS inicial sin escalar
c_base = lsqnonneg(A_raw, y_real);

% Definición del factor de escala S inicial (Target c'_i = 3.0)
S = c_base / 3.0; 

% Primer ajuste sobre las bases escaladas
A_scaled = A_raw .* S';
c_opt = lsqnonneg(A_scaled, y_real);
suma_c = sum(c_opt);

% Si la suma supera MAX_SUM, ajustamos la escala global de S
if suma_c > MAX_SUM
    S = S * (suma_c / MAX_SUM);
    A_scaled = A_raw .* S';
    c_opt = lsqnonneg(A_scaled, y_real);
end

y_pred = A_scaled * c_opt;

% --- 4. MÉTRICAS Y SALIDA EN CONSOLA ---
r2 = 1 - sum((y_real - y_pred).^2) / sum((y_real - mean(y_real)).^2);
rmse = sqrt(mean((y_real - y_pred).^2));
rmse_pct = (rmse / max(y_real)) * 100;

fprintf('=========================================================================\n');
fprintf('Nodos leídos desde Excel         : [%s]\n', num2str(nodos, '%.4f '));
fprintf('Factores de Escala S_i (Activos) : [%s]\n', num2str(S', '%.4f '));
fprintf('Coeficientes c''_i     (Activos) : [%s]\n', num2str(c_opt', '%.4f '));
fprintf('Suma de Coeficientes            : %.4f (<= %.1f)\n', sum(c_opt), MAX_SUM);
fprintf('Métricas de Error               : R^2 = %.6f | RMSE = %.2f | RMSE%% = %.2f%%\n', r2, rmse, rmse_pct);
fprintf('=========================================================================\n');

% --- 5. EXPORTACIÓN A EXCEL (Sobreescritura limpia de 5 columnas exactas) ---
nombre_salida = 'Bases_Escaladas_Optimas.xlsx';
if exist(nombre_salida, 'file')
    delete(nombre_salida);
end

nombres_vars = [{'x'}, arrayfun(@(k) sprintf('phi_%d', k), 1:length(S), 'UniformOutput', false)];
tabla_excel = array2table([x_grid, A_scaled], 'VariableNames', nombres_vars);
writetable(tabla_excel, nombre_salida);

% --- 6. GRÁFICOS ---
figure('Name', 'Ajuste Mínimos Cuadrados');
plot(x_grid, y_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Real'); hold on;
plot(x_grid, y_pred, 'r--', 'LineWidth', 1.8, 'DisplayName', sprintf('NNLS Opt (R^2 = %.4f)', r2));
xlabel('x'); ylabel('y'); title('Ajuste por Mínimos Cuadrados'); legend('Location', 'northwest'); grid on;

figure('Name', 'Bases Escaladas');
plot(x_grid, A_scaled, 'LineWidth', 1.5);
xlabel('x'); ylabel('\phi_i(x)'); title('Bases Escaladas \phi_i(x) > 0 (Nodos Óptimos)'); grid on;

% Leyenda dinámica para las 4 rampas activas
leg = cell(1, length(S));
leg{1} = sprintf('$\\phi_1(x) = %.2f(x - %.2f)$', S(1), nodos(1));
for i = 2:length(S)
    leg{i} = sprintf('$\\phi_{%d}(x) = %.2f \\cdot \\max(0, x - %.4f)$', i, S(i), nodos(i));
end
legend(leg, 'Location', 'northwest', 'Interpreter', 'latex');