% 1. Generar datos de ejemplo
x = linspace(0, 2, 200)';
alpha = 2;

% 2. DEFINIR LOS N PUNTOS DE LA POLILÍNEA (TÚ LOS PROVEES)
%    ¡Asegúrate de que estén dentro del área sombreada!
X_puntos = [0; 0.4; 0.43; 0.92; 1.33; 1.68; 2.0];  % Coordenadas x de los N puntos
Y_puntos = [30; 85; 95; 395; 1258; 3657; 10000];   % Coordenadas y de los 5 puntos

% 3. GENERAR LA POLILÍNEA (interpolación lineal entre los puntos)
x_poli = linspace(0, 2, 200)';                    % Mismo rango que las exponenciales
y_poli = interp1(X_puntos, Y_puntos, x_poli, 'linear');  % Polilínea lineal

y_exacta = y_poli;
y_aprox1 = a1;
y_aprox2 = rn1;

% 2. Graficar
figure('Color', 'white', 'Position', [100, 100, 700, 500]);
hold on;

h1 = plot(x, y_exacta,  'Color', [0 0.4470 0.7410], 'LineWidth', 2.5);
h2 = plot(x, y_aprox1,  'r',                         'LineWidth', 1.8);
h3 = plot(x, y_aprox2,  'Color', [0.13 0.55 0.13], ...
          'LineStyle', '-.', 'LineWidth', 1.8);

% 3. Configurar la grafica
grid on;

xlabel('$x$', ...
       'Interpreter', 'latex', 'FontSize', 12);

ylabel('$f(x)$', ...
       'Interpreter', 'latex', 'FontSize', 12);

title(sprintf('Comparaci\\''on de aproximaciones'), ...
      'Interpreter', 'latex', 'FontSize', 13);

legend([h1, h2, h3], ...
       {sprintf('$\\textrm{Curva J}$'), ...
        '$\\textrm{Aprox. num\\''erica (KKT)}$', ...
        '$\\textrm{Red Neuronal}$'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
hold off;