function Optimizacion_minimos_cuadrados_discreto_MCO_v3()
    % =========================================================================
    % PASO 1: OPTIMIZACIÓN DE NODOS Y EXPORTACIÓN SOLO DE BASES ACTIVAS (> 0)
    % =========================================================================
    clc; clear; close all;

    % Parámetros de la función objetivo
    K = 30; alfa = 3.2; f = @(x) K * exp(alfa * x);
    a = 0; b = 2; N = 4;

    % Muestra discreta (M = 1000 puntos)
    M_muestras = 1000; 
    x_datos = linspace(a, b, M_muestras)'; 
    y_datos = f(x_datos);

    % Optimización de nodos con fmincon
    obj = @(x) error_ecmp_mco(x_datos, y_datos, [a, x(:)', b]);
    x0 = a + (1:N-1)*(b-a)/N; 
    options = optimoptions('fmincon', 'Display', 'off');
    x_opt = fmincon(obj, x0, [],[],[],[], a * ones(1,N-1), b * ones(1,N-1), ...
                    @(x) deal(-diff([a,x,b]) + 1e-4, []), options);

    nodos = [a, x_opt(:)', b];
    [ECMP, X, theta, X_base] = error_ecmp_mco(x_datos, y_datos, nodos);

    % --- IDENTIFICACIÓN Y FILTRADO DE BASES ACTIVAS (NO NULAS) ---
    % Mantenemos solo las columnas donde el coeficiente MCO o la base no sea nula
    idx_activas = find(abs(theta') > 1e-8 & max(abs(X_base), [], 1) > 1e-8);

    theta_filtrado = theta(idx_activas);
    X_base_filtrado = X_base(:, idx_activas);

    fprintf('=========================================================================\n');
    fprintf('   PASO 1: OPTIMIZACIÓN DE NODOS Y FILTRADO DE BASES ACTIVAS\n');
    fprintf('=========================================================================\n');
    fprintf('ECMP_max = %.4f%%\n', ECMP);
    fprintf('Nodos óptimos encontrados : [%s]\n', num2str(nodos, '%.4f '));
    fprintf('Bases Totales Generadas   : %d\n', N + 1);
    fprintf('Bases Activas Retenidas   : %d (Se descartaron las bases nulas)\n\n', length(idx_activas));

    % Impresión de Coeficientes y Funciones Base Retenidas
    fprintf('--- COEFICIENTES Y BASES ACTIVAS RETENIDAS ---\n');
    for i = 1:length(idx_activas)
        k = idx_activas(i);
        fprintf('Base g_%d(x): Coeficiente a_%d = %+12.4f\n', k-1, k-1, theta_filtrado(i));
    end
    fprintf('\n');

    % --- EXPORTACIÓN A EXCEL (SOLO BASES ACTIVAS) ---
    nombre_archivo = 'Bases_Originales_MCO.xlsx';
    fprintf('Exportando únicamente bases activas a "%s"...\n', nombre_archivo);

    % Nombres de columnas según el subíndice original (ej: g_1, g_2, ...)
    nombres_cols = [{'x'}, arrayfun(@(k) sprintf('g_%d', k-1), idx_activas, 'UniformOutput', false)];
    tabla_bases = array2table([x_datos, X_base_filtrado], 'VariableNames', nombres_cols);
    writetable(tabla_bases, nombre_archivo, 'Sheet', 'BasesOriginales');

    tabla_nodos = array2table(nodos, 'VariableNames', arrayfun(@(k) sprintf('nodo_%d', k-1), 1:length(nodos), 'UniformOutput', false));
    writetable(tabla_nodos, nombre_archivo, 'Sheet', 'NodosOptimos');

    fprintf('¡Paso 1 completado exitosamente sin bases nulas!\n\n');
end

% --- FUNCIONES AUXILIARES ---
function [ECMP, X, theta, X_base] = error_ecmp_mco(x_datos, y_datos, X)
    N = length(X) - 1;
    if any(diff(X) <= 1e-6)
        ECMP = 1e6; theta = zeros(N+1,1); X_base = zeros(length(x_datos), N+1); return; 
    end
    M_puntos = length(x_datos);
    X_base = zeros(M_puntos, N+1);
    X_base(:, 1) = 1;                             % g_0(x) = 1 (Constante)
    X_base(:, 2) = x_datos - X(1);                 % g_1(x) = x - x_0
    for k = 1:N-1
        X_base(:, k+2) = max(0, x_datos - X(k+1));% g_k+1(x) = max(0, x - x_k)
    end
    theta = X_base \ y_datos;
    y_pred = X_base * theta;
    err_sq = mean((y_datos - y_pred).^2);
    ECMP = (100 / max(y_pred)) * sqrt(err_sq);
end