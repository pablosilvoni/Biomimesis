function MCO_NNLS_Bases_NoNegativas()
    clc; clear; close all;

    % --- 1. PARÁMETROS DEL PROBLEMA ---
    K = 30.0;
    alfa = 3.2;
    a = 0.0;
    b = 2.0;

    f = @(x) K * exp(alfa * x);
    x_grid = linspace(a, b, 1000)';
    y_real = f(x_grid);
    y_max = max(y_real); % Valor máximo de la función real en [0, 2]

    % Particiones de Nodos
    nodos_uni = [0.0, 0.5, 1.0, 1.5, 2.0];
    nodos_opt = [0.0, 1.0613, 1.5082, 1.7920, 2.0];

    % --- 2. BASES RAMPA CRUDAS ---
    A_raw_uni = construir_matriz_rampa(x_grid, nodos_uni);
    A_raw_opt = construir_matriz_rampa(x_grid, nodos_opt);

    % Ajuste NNLS en bases crudas (Garantiza c_raw >= 0)
    c_raw_uni = lsqnonneg(A_raw_uni, y_real);
    c_raw_opt = lsqnonneg(A_raw_opt, y_real);

    % --- 3. FACTORES DE ESCALA POSITIVOS S_i ---
    C_TARGET = 3.0;
    S_uni = c_raw_uni / C_TARGET; % S_i >= 0
    S_opt = c_raw_opt / C_TARGET; % S_i >= 0

    % --- 4. BASES ESCALADAS \phi_i(x) = S_i * g_i(x) >= 0 ---
    A_scaled_uni = A_raw_uni .* S_uni';
    A_scaled_opt = A_raw_opt .* S_opt';

    % Ajuste NNLS sobre Bases Escaladas
    c_scaled_uni = lsqnonneg(A_scaled_uni, y_real);
    c_scaled_opt = lsqnonneg(A_scaled_opt, y_real);

    y_pred_uni = A_scaled_uni * c_scaled_uni;
    y_pred_opt = A_scaled_opt * c_scaled_opt;

    % Métricas de Precisión (R2 y RMSE)
    r2_uni = 1 - sum((y_real - y_pred_uni).^2) / sum((y_real - mean(y_real)).^2);
    r2_opt = 1 - sum((y_real - y_pred_opt).^2) / sum((y_real - mean(y_real)).^2);

    rmse_uni = sqrt(mean((y_real - y_pred_uni).^2));
    rmse_opt = sqrt(mean((y_real - y_pred_opt).^2));

    % --- CÁLCULO DEL RMSE% RESPECTO AL MÁXIMO DE Y_REAL ---
    rmse_pct_uni = (rmse_uni / y_max) * 100;
    rmse_pct_opt = (rmse_opt / y_max) * 100;

    % --- 5. REPORTES EN CONSOLA Y FÓRMULAS MATEMÁTICAS ---
    fprintf('=========================================================================\n');
    fprintf('  AJUSTE MÍNIMOS CUADRADOS NO NEGATIVOS (NNLS)\n');
    fprintf('=========================================================================\n');
    fprintf('Función Objetivo: y = %.1f * exp(%.2f * x) en [%.1f, %.1f]\n', K, alfa, a, b);
    fprintf('Valor Máximo y(x) : %.2f\n\n', y_max);

    fprintf('--- 1. RESULTADOS Y FÓRMULAS CON NODOS UNIFORMES ---\n');
    fprintf('Factores de Escala S_i : [%.2f, %.2f, %.2f, %.2f, %.2f]\n', S_uni);
    fprintf('Coeficientes c''_i NNLS : [%.2f, %.2f, %.2f, %.2f, %.2f]\n', c_scaled_uni);
    fprintf('Métricas              : R^2 = %.6f | RMSE = %.2f | RMSE%% = %.4f%%\n', r2_uni, rmse_uni, rmse_pct_uni);
    fprintf('Fórmulas de las Bases Escaladas phi_i(x):\n');
    imprimir_formulas_bases(S_uni, nodos_uni);
    fprintf('\n');

    fprintf('--- 2. RESULTADOS Y FÓRMULAS CON NODOS ÓPTIMOS ---\n');
    fprintf('Factores de Escala S_i : [%.2f, %.2f, %.2f, %.2f, %.2f]\n', S_opt);
    fprintf('Coeficientes c''_i NNLS : [%.2f, %.2f, %.2f, %.2f, %.2f]\n', c_scaled_opt);
    fprintf('Métricas              : R^2 = %.6f | RMSE = %.2f | RMSE%% = %.4f%%\n', r2_opt, rmse_opt, rmse_pct_opt);
    fprintf('Fórmulas de las Bases Escaladas phi_i(x):\n');
    imprimir_formulas_bases(S_opt, nodos_opt);
    fprintf('=========================================================================\n\n');

    % --- 6. GENERACIÓN DE FIGURAS ---
    figure('Name', 'Ajuste NNLS con Bases Escaladas', 'Position', [100, 100, 800, 500]);
    plot(x_grid, y_real, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Función Real y = 30*exp(3.2x)'); hold on;
    plot(x_grid, y_pred_uni, 'b--', 'LineWidth', 1.5, 'DisplayName', sprintf('NNLS Malla Uniforme (RMSE%% = %.2f%%)', rmse_pct_uni));
    plot(x_grid, y_pred_opt, 'r-', 'LineWidth', 1.8, 'DisplayName', sprintf('NNLS Nodos Óptimos (RMSE%% = %.2f%%)', rmse_pct_opt));
    xlabel('x'); ylabel('y');
    title('Ajuste NNLS (Bases y Coeficientes >= 0)');
    legend('Location', 'northwest'); grid on;

    figure('Name', 'Bases Rampa Escaladas No Negativas', 'Position', [150, 150, 1000, 450]);
    subplot(1, 2, 1);
    plot(x_grid, A_scaled_uni, 'LineWidth', 1.5);
    title('Bases \phi_i(x) \ge 0 (Nodos Uniformes)');
    xlabel('x'); ylabel('\phi_i(x)'); grid on;
    legend('\phi_0', '\phi_1', '\phi_2', '\phi_3', '\phi_4', 'Location', 'northwest');

    subplot(1, 2, 2);
    plot(x_grid, A_scaled_opt, 'LineWidth', 1.5);
    title('Bases \phi_i(x) \ge 0 (Nodos Óptimos)');
    xlabel('x'); ylabel('\phi_i(x)'); grid on;
    legend('\phi_0', '\phi_1', '\phi_2', '\phi_3', '\phi_4', 'Location', 'northwest');
end

function A = construir_matriz_rampa(x, nodos)
    N = length(nodos) - 1;
    A = zeros(length(x), N + 1);
    A(:, 1) = 1.0;
    A(:, 2) = x - nodos(1);
    for k = 1:N-1
        A(:, k+2) = max(0, x - nodos(k+1));
    end
end

function imprimir_formulas_bases(S, nodos)
    fprintf('  phi_0(x) = %10.4f * 1\n', S(1));
    fprintf('  phi_1(x) = %10.4f * (x - %.4f)\n', S(2), nodos(1));
    for k = 1:length(nodos)-2
        fprintf('  phi_%d(x) = %10.4f * max(0, x - %.4f)\n', k+1, S(k+2), nodos(k+1));
    end
end