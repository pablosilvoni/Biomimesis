% ============================================
% CÓDIGO CORREGIDO - LEGEND SIN PROBLEMAS
% ============================================

% 1. Datos
x = linspace(0, 2, 200)';
y1 = 30 * exp(2.1 * x);
y2 = 30 * exp(3.2 * x);

% 2. Curva J (polilínea)
X_puntos = [0; 0.4; 0.43; 0.92; 1.33; 1.68; 2.0];
Y_puntos = [30; 85; 95; 395; 1258; 3657; 10000];
x_poli = linspace(0, 2, 200)';
y_poli = interp1(X_puntos, Y_puntos, x_poli, 'linear');
y_exacta = y_poli;

% 3. Aproximaciones
y_aprox1 = rn1;  % Aprox. Numérica (KKT)

% 4. Graficar
figure('Color', 'white', 'Position', [100, 100, 700, 500]);
hold on;

h1 = plot(x, y_exacta, 'Color', [0 0.4470 0.7410], 'LineWidth', 2);
h2 = plot(x, y_aprox1, 'r-.', 'LineWidth', 2);

% 5. Configurar
grid on;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparación de aproximaciones con Curva J', ...
      'Interpreter', 'latex', 'FontSize', 13);

% 🔥 CORRECCIÓN: Legend con texto SIMPLE (sin LaTeX para evitar problemas)
legend([h1, h2], ...
       {'Curva J', ...
        'Red Neuronal'}, ...  % <--- SIN TILDE Y SIN LaTeX
       'FontSize', 11, ...
       'Location', 'northwest');

hold off;
