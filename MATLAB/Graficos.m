% 1. Generar datos de ejemplo
x = linspace(0, 2, 200)';

% Definir la función objetivo
alpha = 3.2;  % Valor de alpha
y_exacta = 30 * exp(alpha * x);

% Datos de aproximación
y_aprox1 = rn32int;      % RN c/Enteros



% 2. Graficar
figure('Color', 'white', 'Position', [100, 100, 700, 500]);
hold on;

% Línea exacta (azul)
h1 = plot(x, y_exacta, 'Color', [0 0.4470 0.7410], 'LineWidth', 2.5);

% Línea aproximación numérica con RN con Enteros (roja)
h2 = plot(x, y_aprox1, 'r', 'LineWidth', 1.8);


% 3. Configurar la gráfica (USANDO LA TEX CORRECTAMENTE)
grid on;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 12);
title(['Comparacion de aproximaciones para $\alpha = ' num2str(alpha) '$'], ...
      'Interpreter', 'latex', 'FontSize', 13);

% 4. Leyenda con LaTeX (sin acentos)
legend([h1, h2], ...
       {['$y = 30 e^{' num2str(alpha) 'x}$'], ...
         'RN c/enteros'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');

hold off;