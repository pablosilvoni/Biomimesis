% 1. Generar datos de ejemplo
x = linspace(0, 2, 200)';

% Definir la función objetivo
alpha = 2.2;  % Valor de alpha
y_exacta = 30 * exp(alpha * x);

% Datos de aproximación (cargar desde archivo o variable)
y_aprox1 = lg2;      % Aproximación numérica (Lagrange)
y_aprox2 = rn2;     % Aproximación con Red Neuronal


% 2. Graficar
figure('Color', 'white', 'Position', [100, 100, 700, 500]);
hold on;

% Línea exacta (azul)
h1 = plot(x, y_exacta, 'Color', [0 0.4470 0.7410], 'LineWidth', 2.5);

% Línea aproximación numérica (roja)
h2 = plot(x, y_aprox1, 'r', 'LineWidth', 1.8);

% Línea Red Neuronal (verde discontinua)
h3 = plot(x, y_aprox2, 'Color', [0.13 0.55 0.13], 'LineStyle', '-.', 'LineWidth', 1.8);

% 3. Configurar la gráfica
grid on;
xlabel('x', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('f(x)', 'Interpreter', 'latex', 'FontSize', 12);
title(['Comparacion de aproximaciones para \alpha = ' num2str(alpha)], ...
      'FontSize', 13);  % Sin LaTeX para evitar errores con acentos

% 4. Leyenda con handles (SIN LaTeX para evitar errores con acentos)
legend([h1, h2, h3], ...
       {['y = 30 exp(' num2str(alpha) 'x)'], ...
        'Aprox. Numerica (Lagrange)', ...
        'Red Neuronal'}, ...
       'FontSize', 11, 'Location', 'northwest');

hold off;