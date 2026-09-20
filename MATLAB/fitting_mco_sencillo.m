function fitting_mco_sencillo()
    % =========================================================================
    % FITTING POR MÍNIMOS CUADRADOS ACOTADOS (NON-NEGATIVE LSQ) DE BASES DADAS
    % FIGURAS INDEPENDIENTES Y ESTILOS PERSONALIZADOS (AZUL TARGET / ROJO FIT)
    % =========================================================================
    clc; clear; close all;

    % -------------------------------------------------------------------------
    % 1. CONFIGURACIÓN DE ENTRADAS
    % -------------------------------------------------------------------------
    % Modo para la función objetivo 'f_target': 'FORMULA' o 'EXCEL'
    MODO_TARGET = 'FORMULA'; 
    
    % Restricción física de no-negatividad en los coeficientes (c_i >= 0)
    NO_NEGATIVO = true; 

    % Fórmula en formato LaTeX para la leyenda y título de las gráficas
    FORMULA_LATEX = '$y(x) = 30 \cdot e^{2 x}$';

    % Archivos Excel de entrada
    ARCHIVO_TARGET_EXCEL = 'Curva_J.xls'; 
    ARCHIVO_BASES_EXCEL  = 'funciones_base_corregido.xlsx'; 

    % Dominio de evaluación (si MODO_TARGET = 'FORMULA')
    a = 0.0;
    b = 2.0;
    M_puntos = 999;
    x_grid = linspace(a, b, M_puntos)';

    % -------------------------------------------------------------------------
    % 2. CARGA / EVALUACIÓN DE LA FUNCIÓN TARGET y(x)
    % -------------------------------------------------------------------------
    if strcmpi(MODO_TARGET, 'FORMULA')
        % Ejemplo de envolvente superior: y = 30 * exp(3.2 * x)
        f_target = @(x) 30.0 * exp(3.2 * x);
        y_real = f_target(x_grid);
        desc_target = sprintf('Fórmula analítica: %s', FORMULA_LATEX);
        etiqueta_leyenda_target = FORMULA_LATEX;
        
    elseif strcmpi(MODO_TARGET, 'EXCEL')
        if ~exist(ARCHIVO_TARGET_EXCEL, 'file')
            error('No se encuentra el archivo: %s', ARCHIVO_TARGET_EXCEL);
        end
        data_target = readmatrix(ARCHIVO_TARGET_EXCEL);
        x_grid = data_target(:, 1);
        y_real = data_target(:, 2);
        desc_target = sprintf('Excel (%s)', ARCHIVO_TARGET_EXCEL);
        etiqueta_leyenda_target = sprintf('Target Excel (%s)', ARCHIVO_TARGET_EXCEL);
    else
        error('MODO_TARGET no válido. Use ''FORMULA'' o ''EXCEL''.');
    end

    % -------------------------------------------------------------------------
    % 3. CARGA DINÁMICA DE LAS FUNCIONES BASE (N BASES)
    % -------------------------------------------------------------------------
    if ~exist(ARCHIVO_BASES_EXCEL, 'file')
        error('No se encuentra el archivo de bases: %s', ARCHIVO_BASES_EXCEL);
    end

    data_bases = readmatrix(ARCHIVO_BASES_EXCEL);
    x_bases = data_bases(:, 1);

    % Lee dinámicamente desde la columna 2 hasta el final (N_BASES = size(A_raw, 2))
    A_raw = data_bases(:, 2:end); 

    % Interpolación por si la grilla del Excel difiere de x_grid
    A = zeros(length(x_grid), size(A_raw, 2));
    for col = 1:size(A_raw, 2)
        A(:, col) = interp1(x_bases, A_raw(:, col), x_grid, 'linear', 'extrap');
    end

    % -------------------------------------------------------------------------
    % 4. CÁLCULO DEL FITTING (MCO LIBRE vs NO-NEGATIVO)
    % -------------------------------------------------------------------------
    if NO_NEGATIVO
        % Mínimos Cuadrados No-Negativos (c_i >= 0)
        c_opt = lsqnonneg(A, y_real);
        desc_metodo = 'MCO No-Negativo (lsqnonneg)';
    else
        % MCO Libre
        c_opt = A \ y_real;
        desc_metodo = 'MCO Libre (A \\ y)';
    end

    y_fitting = A * c_opt;

    % -------------------------------------------------------------------------
    % 5. CÁLCULO DE MÉTRICAS CIENTÍFICAS
    % -------------------------------------------------------------------------
    residuos = y_real - y_fitting;
    rmse = sqrt(mean(residuos.^2));
    mae = mean(abs(residuos));
    y_max = max(y_real);
    nrmse_porcentaje = (rmse / y_max) * 100;

    ss_res = sum(residuos.^2);
    ss_tot = sum((y_real - mean(y_real)).^2);
    r2 = 1 - (ss_res / ss_tot);

    % -------------------------------------------------------------------------
    % 6. REPORTE EN CONSOLA
    % -------------------------------------------------------------------------
    fprintf('=========================================================================\n');
    fprintf('                 RESULTADOS DEL FITTING EN MATLAB (N = %d)               \n', size(A, 2));
    fprintf('=========================================================================\n');
    fprintf('Origen de Curva Target : %s\n', desc_target);
    fprintf('Archivo de Bases       : %s\n', ARCHIVO_BASES_EXCEL);
    fprintf('Método de Ajuste       : %s\n', desc_metodo);
    fprintf('Cantidad de Bases N    : %d\n\n', size(A, 2));

    fprintf('--- COEFICIENTES DEL FITTING (c_i) ---\n');
    for i = 1:length(c_opt)
        fprintf('  c_%d = %+14.6f\n', i, c_opt(i));
    end
    fprintf('  ------------------------------------\n');
    fprintf('  Suma de c_i = %+14.6f\n\n', sum(c_opt));

    fprintf('--- MÉTRICAS DE ERROR ---\n');
    fprintf('  R^2 Score               = %.6f\n', r2);
    fprintf('  RMSE                    = %.6f\n', rmse);
    fprintf('  MAE                     = %.6f\n', mae);
    fprintf('  Máximo de y(x)          = %.6f\n', y_max);
    fprintf('  NRMSE%% (ref. al máx)    = %.4f %%\n', nrmse_porcentaje);
    fprintf('=========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 7. GRÁFICOS SEPARADOS (3 VENTANAS INDEPENDIENTES)
    % -------------------------------------------------------------------------
    
    % FIGURA 1: Función Explícita vs Fitting
    figure('Name', 'Figura 1: Función Target vs Fitting MCO', 'Position', [100, 150, 750, 500]);
    plot(x_grid, y_real, 'b-', 'LineWidth', 2.2, 'DisplayName', etiqueta_leyenda_target); hold on;
    
    etiqueta_fitting = sprintf('Fitting ($R^2 = %.5f$)', r2);
    plot(x_grid, y_fitting, 'r--', 'LineWidth', 1.8, 'DisplayName', etiqueta_fitting);
    
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12); 
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 12);
    title('Ajuste por Mínimos Cuadrados', 'Interpreter', 'latex', 'FontSize', 14);
    legend('Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 11); 
    grid on;

    % FIGURA 2: Distribución de Residuos Locales
    figure('Name', 'Figura 2: Residuos Locales', 'Position', [200, 150, 750, 500]);
    plot(x_grid, residuos, 'b-', 'LineWidth', 1.5);
    yline(0, 'k--', 'Alpha', 0.6);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12); 
    ylabel('Residuo ($y_{real} - y_{fit}$)', 'Interpreter', 'latex', 'FontSize', 12);
    title('Distribución de Residuos Locales', 'Interpreter', 'latex', 'FontSize', 14);
    grid on;

    % FIGURA 3: Error NRMSE% Local Punto a Punto
    figure('Name', 'Figura 3: Error NRMSE% Local', 'Position', [300, 150, 750, 500]);
    nrmse_local = (abs(residuos) / y_max) * 100;
    plot(x_grid, nrmse_local, 'm-', 'LineWidth', 1.5);
    yline(1.0, 'g:', 'LineWidth', 1.5, 'Label', 'Tolerancia 1%', 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12); 
    ylabel('NRMSE Local ($\% y_{max}$)', 'Interpreter', 'latex', 'FontSize', 12);
    title('Error NRMSE\% Local Punto a Punto', 'Interpreter', 'latex', 'FontSize', 14);
    grid on;
end