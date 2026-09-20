% 1. Datos base
x = linspace(0, 2, 200)';
y_exacta = y_o;

% 2. Generar aproximación por mínimos cuadrados (Grado 1)
% p = polyfit(x, y_exacta, 1); 
y_aprox = a23;

% =====================================================================
% 3. CÁLCULO DEL ERROR CUADRÁTICO PUNTO A PUNTO
% =====================================================================
% Elevamos al cuadrado la diferencia en cada punto usando el operador '.^'
error_cuadratico_punto_a_punto = (y_exacta - y_aprox).^2; 

% El ECM global (un solo número) es simplemente el promedio de este vector:
ecm_global = mean(error_cuadratico_punto_a_punto);
fprintf('El Error Cuadrático Medio (ECM) global es: %f\n', ecm_global);

% =====================================================================
% 4. Gráfica del Error Cuadrático
% =====================================================================
figure;

% Graficar el error punto a punto
plot(x, error_cuadratico_punto_a_punto, 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2.5);
hold on;

% Dibujar una línea horizontal discontinua que represente el promedio (ECM Global)
line([x(1), x(end)], [ecm_global, ecm_global], 'Color', [0, 0, 0], 'LineStyle', '--', 'LineWidth', 1.5);

grid on;
title('Análisis del Error Cuadrático Punto a Punto', 'FontSize', 12);
xlabel('Eje X');
ylabel('Error Cuadrático $$(y_{exacta} - y_{aprox})^2$$', 'Interpreter', 'latex', 'FontSize', 12);

% Leyenda explicativa
legend({'Error Cuadrático Individual', 'Error Cuadrático Medio (Promedio)'}, 'Location', 'northwest');
