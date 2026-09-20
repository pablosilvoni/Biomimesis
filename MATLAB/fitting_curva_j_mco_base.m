function fitting_curva_j_mco_base()
    % =========================================================================
    % SCRIPT DE MATLAB: FITTING DE CURVA J MEDIANTE MÍNIMOS CUADRADOS (MCO)
    % SOBRE UNA BASE DE FUNCIONES DADA (FÓRMULAS O DESDE ARCHIVO EXCEL)
    % =========================================================================
    % Permite dos modalidades para definir la CURVA OBJETIVO f_target:
    %    1. 'FORMULA' : Evalúa la función matemática analítica.
    %    2. 'EXCEL'   : Lee la curva objetivo desde un archivo .xlsx (Columna 1: x, Columna 2: y).
    %
    % Permite dos modalidades para definir la BASE DE FUNCIONES:
    %    1. 'FORMULAS' : Genera analíticamente las funciones base rampa/bisagra.
    %    2. 'EXCEL'    : Lee la matriz de base directamente de un archivo .xlsx.
    % =========================================================================
    clc; clear; close all;

    % -------------------------------------------------------------------------
    % 1. CONFIGURACIÓN DE MODOS Y ARCHIVOS
    % -------------------------------------------------------------------------
    MODO_TARGET  = 'FORMULA';  % Opciones: 'FORMULA' o 'EXCEL'
    ARCHIVO_TARGET_EXCEL = 'curva_target.xlsx'; % Archivo Excel si MODO_TARGET = 'EXCEL'

    MODO_BASE    = 'FORMULAS'; % Opciones: 'FORMULAS' o 'EXCEL'
    ARCHIVO_BASE_EXCEL   = 'Bases_Rampa_Escaladas_Nodos_Optimos.xlsx'; % Archivo Excel si MODO_BASE = 'EXCEL'

    % Dominio y resolución de discretización
    a = 0.0;
    b = 2.0;
    M_puntos = 1000;
    x_grid = linspace(a, b, M_puntos)';

    % -------------------------------------------------------------------------
    % 2. CONSTRUCCIÓN / CARGA DE LA CURVA OBJETIVO f_target
    % -------------------------------------------------------------------------
    if strcmpi(MODO_TARGET, 'FORMULA')
        fprintf('-> Generando Curva Objetivo mediante FÓRMULA analítica...\n');
        K = 30.0;
        alfa = 3.2;
        f_target = @(x) K * exp(alfa * x);
        y_real = f_target(x_grid);
        desc_target = sprintf('y = %.1f * exp(%.2f * x)', K, alfa);

    elseif strcmpi(MODO_TARGET, 'EXCEL')
        fprintf('-> Leyendo Curva Objetivo desde archivo Excel: %s...\n', ARCHIVO_TARGET_EXCEL);
        if ~exist(ARCHIVO_TARGET_EXCEL, 'file')
            error('El archivo Excel de la curva objetivo "%s" no existe.', ARCHIVO_TARGET_EXCEL);
        end
        
        data_target = readmatrix(ARCHIVO_TARGET_EXCEL);
        x_target_excel = data_target(:, 1);
        y_target_excel = data_target(:, 2);
        
        % Interpolación si la malla del Excel difiere de x_grid
        y_real = interp1(x_target_excel, y_target_excel, x_grid, 'linear', 'extrap');
        desc_target = sprintf('Cargada desde Excel (%s)', ARCHIVO_TARGET_EXCEL);
    else
        error('MODO_TARGET no válido. Use ''FORMULA'' o ''EXCEL''.');
    end

    % -------------------------------------------------------------------------
    % 3. CONSTRUCCIÓN / CARGA DE LA MATRIZ DE DISEÑO A (BASES)
    % -------------------------------------------------------------------------
    if strcmpi(MODO_BASE, 'FORMULAS')
        fprintf('-> Generando matriz de base mediante FÓRMULAS analíticas...\n');
        
        % Nodos de quiebre (5 nodos = 4 tramos)
        nodos = [0.0, 1.0613, 1.5082, 1.7920, 2.0];
        
        % Construcción de la matriz A con funciones rampa desplazadas
        A = construir_matriz_rampa(x_grid, nodos);
        nombres_bases = arrayfun(@(i) sprintf('\\phi_%d(x)', i-1), 1:size(A,2), 'UniformOutput', false);

    elseif strcmpi(MODO_BASE, 'EXCEL')
        fprintf('-> Leyendo matriz de base desde archivo Excel: %s...\n', ARCHIVO_BASE_EXCEL);
        if ~exist(ARCHIVO_BASE_EXCEL, 'file')
            error('El archivo Excel de las bases "%s" no existe.', ARCHIVO_BASE_EXCEL);
        end
        
        data_base_excel = readmatrix(ARCHIVO_BASE_EXCEL);
        x_base_excel = data_base_excel(:, 1);
        
        % Si el Excel tiene columnas adicionales como y_real o y_pred, tomar solo las columnas de phi
        % Asumimos que las columnas de bases van desde la columna 2 hasta la penúltima o según corresponda
        if size(data_base_excel, 2) > 6
            A_excel = data_base_excel(:, 2:6); % Toma phi_0 a phi_4
        else
            A_excel = data_base_excel(:, 2:end);
        end
        
        % Interpolación si la malla del Excel difiere de x_grid
        A = zeros(length(x_grid), size(A_excel, 2));
        for col = 1:size(A_excel, 2)
            A(:, col) = interp1(x_base_excel, A_excel(:, col), x_grid, 'linear', 'extrap');
        end
        nombres_bases = arrayfun(@(i) sprintf('\\phi_%d(x)', i-1), 1:size(A,2), 'UniformOutput', false);
    else
        error('MODO_BASE no válido. Use ''FORMULAS'' o ''EXCEL''.');
    end

    % -------------------------------------------------------------------------
    % 4. CÁLCULO DEL FITTING POR MÍNIMOS CUADRADOS
    % -------------------------------------------------------------------------
    % 1) MCO Libre (sin restricciones)
    c_mco = A \ y_real;
    y_pred_mco = A * c_mco;

    % 2) MCO No Negativo (c_i >= 0)
    c_nnls = lsqnonneg(A, y_real);
    y_pred_nnls = A * c_nnls;

    % -------------------------------------------------------------------------
    % 5. CÁLCULO DE MÉTRICAS DE PRECISIÓN
    % -------------------------------------------------------------------------
    [r2_mco, rmse_mco, nrmse_mco] = calcular_metricas(y_real, y_pred_mco);
    [r2_nnls, rmse_nnls, nrmse_nnls] = calcular_metricas(y_real, y_pred_nnls);

    % -------------------------------------------------------------------------
    % 6. REPORTE DE RESULTADOS Y COEFICIENTES EN CONSOLA
    % -------------------------------------------------------------------------
    fprintf('\n=========================================================================\n');
    fprintf('   AJUSTE DE CURVA J POR MÍNIMOS CUADRADOS (FITTING CON BASE DADA)\n');
    fprintf('=========================================================================\n');
    fprintf('Curva Objetivo : %s  en  x \\in [%.1f, %.1f]\n', desc_target, a, b);
    fprintf('Modo de Base   : %s  (Dimensión de la base N = %d)\n\n', MODO_BASE, size(A, 2));

    fprintf('--- 1. RESULTADOS Y COEFICIENTES MCO LIBRE ---\n');
    for i = 1:length(c_mco)
        fprintf('  Coeficiente c_%d (para %s) = %+14.6f\n', i-1, nombres_bases{i}, c_mco(i));
    end
    fprintf('  ---------------------------------------------\n');
    fprintf('  Suma de coeficientes sum(c_i) = %+14.6f\n', sum(c_mco));
    fprintf('  Métricas: R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f %%\n\n', r2_mco, rmse_mco, nrmse_mco);

    fprintf('--- 2. RESULTADOS Y COEFICIENTES MCO NO NEGATIVO (NNLS) ---\n');
    for i = 1:length(c_nnls)
        fprintf('  Coeficiente c_%d (para %s) = %+14.6f\n', i-1, nombres_bases{i}, c_nnls(i));
    end
    fprintf('  ---------------------------------------------\n');
    fprintf('  Suma de coeficientes sum(c_i) = %+14.6f\n', sum(c_nnls));
    fprintf('  Métricas: R^2 = %.6f | RMSE = %.4f | NRMSE = %.4f %%\n', r2_nnls, rmse_nnls, nrmse_nnls);
    fprintf('=========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 7. GENERACIÓN DE GRÁFICOS
    % -------------------------------------------------------------------------
    figure('Name', 'Ajuste de Curva J con MCO', 'Position', [100, 100, 850, 500]);
    plot(x_grid, y_real, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Curva J Real / Objetivo'); hold on;
    plot(x_grid, y_pred_mco, 'r--', 'LineWidth', 1.8, 'DisplayName', sprintf('Fitting MCO Libre (R^2 = %.4f)', r2_mco));
    plot(x_grid, y_pred_nnls, 'b:', 'LineWidth', 1.8, 'DisplayName', sprintf('Fitting MCO No Negativo (R^2 = %.4f)', r2_nnls));
    xlabel('x'); ylabel('y');
    title('Ajuste por Mínimos Cuadrados de la Curva J');
    legend('Location', 'northwest'); grid on;

    figure('Name', 'Funciones Base Utilizadas', 'Position', [150, 150, 850, 450]);
    plot(x_grid, A, 'LineWidth', 1.5);
    xlabel('x'); ylabel('\phi_i(x)');
    title('Funciones Base Evaluadas en la Grilla');
    legend(nombres_bases, 'Location', 'northwest'); grid on;
end

% =========================================================================
% FUNCIONES AUXILIARES LOCALES
% =========================================================================

function A = construir_matriz_rampa(x, nodos)
    N = length(nodos) - 1;
    A = zeros(length(x), N + 1);
    A(:, 1) = 1.0;                  % Base 0: constante g_0(x) = 1
    A(:, 2) = x - nodos(1);         % Base 1: lineal g_1(x) = x - x_0
    for k = 1:N-1
        A(:, k+2) = max(0, x - nodos(k+1)); % Bases 2..N: rampa max(0, x - x_k)
    end
end

function [r2, rmse, nrmse] = calcular_metricas(y_real, y_pred)
    ss_res = sum((y_real - y_pred).^2);
    ss_tot = sum((y_real - mean(y_real)).^2);
    r2 = 1 - (ss_res / ss_tot);
    rmse = sqrt(mean((y_real - y_pred).^2));
    nrmse = (rmse / max(y_real)) * 100;
end