% 1. Generar datos de ejemplo
x = linspace(0, 2, 200)';
y_exacta = y_o
y_aprox1 = a22              % Aproximación numérica
y_aprox2 = rn22              % Aproximación con RN

% 2. Graficar guardando los handles de cada línea
figure;
h1 = plot(x, y_exacta, 'Color',[0 0.4470 0.7410], 'LineWidth', 2);   % Línea azul MATLAB
hold on;
h2 = plot(x, y_aprox1, 'r', 'LineWidth', 1.5); % Línea roja continua
h3 = plot(x, y_aprox2, 'Color', [0.13 0.55 0.13], 'LineStyle', '-.','LineWidth',1.5); % Linea verde bosque standard
grid on;

% 3. Crear la leyenda asociando los handles con sus fórmulas LaTeX
legend([h1, h2, h3], ...
       {'$$y = e^{2.2x}$$', ...
        'Aproximación Numérica', ... 
        'Aproximación con Red Neuronal'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest')
