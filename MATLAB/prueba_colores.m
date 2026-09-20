x = linspace(0, 2*pi, 200);

% Tres funciones con amplitudes y fases para que se crucen en varios puntos
y1 = sin(x);            % Azul ('b')
y2 = cos(x);            % Rojo ('r')
y3 = 0.8 * sin(2*x);    % Naranja/Dorado (Tercer color de alto contraste)

figure;
hold on;

% Graficar las 3 curvas en el mismo eje
plot(x, y1, 'Color', [0.00 0.00 1.00], 'LineWidth', 2.5, 'DisplayName', 'y1 = sin(x)');
plot(x, y2, 'Color', [1.00 0.00 0.00], 'LineWidth', 2.5, 'DisplayName', 'y2 = cos(x)');
plot(x, y3, 'Color', [0.0 0.50 0.00], 'LineWidth', 2.5, 'DisplayName', 'y3 = 0.8*sin(2x)');

grid on;
legend('Location', 'northeast');
xlabel('x');
ylabel('f(x)');
title('Contraste visual de curvas superpuestas');
hold off;