% =========================================================================
% AJUSTE NO LINEAL EXPONENCIAL: Presión (Pi) vs Estiramiento (lambda_theta)
% =========================================================================
%clear; clc; close all;

% 1. Conjunto de datos extraídos
%x_data = [1.000, 1.020, 1.100, 1.150, 1.200, 1.250, 1.300, 1.350, 1.400]';
%y_data = [0.000, 0.200, 0.850, 1.125, 1.875, 3.330, 5.000, 7.500, 17.500]';

x_data = x;
y_data = y;

% 2. Definición de la función modelo: P_i(lambda) = a * (exp(b * (lambda - 1)) - 1)
% p(1) representa 'a', p(2) representa 'b'
modelo = @(p, x) p(1) * (exp(p(2) * (x - 1)) - 1);

% 3. Estimación inicial [a0, b0] y opciones de optimización
p0 = [0.1, 10.0];
opciones = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxFunctionEvaluations', 10000);

% 4. Ajuste por mínimos cuadrados no lineales
popt = lsqcurvefit(modelo, p0, x_data, y_data, [], [], opciones);

% 5. Mostrar resultados en consola
fprintf('--- PARÁMETROS DEL FITTING ---\n');
fprintf('a = %.4f kPa\n', popt(1));
fprintf('b = %.4f\n', popt(2));
fprintf('Ecuación: P_i = %.4f * (exp(%.4f * (lambda - 1)) - 1)\n', popt(1), popt(2));

% 6. Generar gráfico comparativo
x_grid = linspace(1.00, 1.40, 200);
y_fit = modelo(popt, x_grid);

figure('Color', 'w');
plot(x_data, y_data, 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'r', 'DisplayName', 'Datos Extraídos (9 pts)');
hold on;
plot(x_grid, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Ajuste: a=%.4f, b=%.4f', popt(1), popt(2)));
grid on;
xlabel('Estiramiento Circunferencial \lambda_\theta [-]');
ylabel('Presión Intraluminal P_i [kPa]');
title('Ajuste Exponencial en MATLAB');
legend('Location', 'northwest');