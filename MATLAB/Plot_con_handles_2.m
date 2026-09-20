% 1. Generar datos de ejemplo
<<<<<<< HEAD
% x = linspace(0, 2, 200)';

% Definir la función objetivo
alpha = 2.75;  % Valor de alpha
y_exacta = 30 * exp(alpha*x);

% Datos de aproximación (cargar desde archivo o variable)
y_aprox1 = a2;      % Aproximación numérica (Lagrange)
y_aprox2 = rn2;     % Aproximación con Red Neuronal

=======
x = linspace(0, 2, 200)';

% Definir la función objetivo
alpha = 2.2;  % Valor de alpha
y_exacta = 30 * exp(alpha * x);

% Datos de aproximación (cargar desde archivo o variable)
y_aprox1 = lg2;      % Aproximación numérica (Lagrange)
y_aprox2 = rn2;     % Aproximación con Red Neuronal


>>>>>>> 5989a53da681086e780c08a059bbc04268c324cb
% 2. Graficar
figure('Color', 'white', 'Position', [100, 100, 700, 500]);
hold on;

% Línea exacta (azul)
<<<<<<< HEAD
h1 = plot(x, y_exacta, 'b', 'LineWidth', 2.5);
=======
h1 = plot(x, y_exacta, 'Color', [0 0.4470 0.7410], 'LineWidth', 2.5);
>>>>>>> 5989a53da681086e780c08a059bbc04268c324cb

% Línea aproximación numérica (roja)
h2 = plot(x, y_aprox1, 'r', 'LineWidth', 1.8);

% Línea Red Neuronal (verde discontinua)
h3 = plot(x, y_aprox2, 'Color', [0.13 0.55 0.13], 'LineStyle', '-.', 'LineWidth', 1.8);

% 3. Configurar la gráfica
grid on;
<<<<<<< HEAD
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 12);
=======
xlabel('x', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('f(x)', 'Interpreter', 'latex', 'FontSize', 12);
>>>>>>> 5989a53da681086e780c08a059bbc04268c324cb
title(['Comparacion de aproximaciones para \alpha = ' num2str(alpha)], ...
      'FontSize', 13);  % Sin LaTeX para evitar errores con acentos

% 4. Leyenda con handles (SIN LaTeX para evitar errores con acentos)
legend([h1, h2, h3], ...
       {['y = 30 exp(' num2str(alpha) 'x)'], ...
<<<<<<< HEAD
        'Mínimos Cuadrados', ...
        'Red Neuronal'}, ...
        'FontSize', 11, 'Location', 'northwest');
=======
        'Aprox. Numerica (Lagrange)', ...
        'Red Neuronal'}, ...
       'FontSize', 11, 'Location', 'northwest');
>>>>>>> 5989a53da681086e780c08a059bbc04268c324cb

hold off;