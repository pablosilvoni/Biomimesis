% 1. Datos base
x = linspace(0, 2, 200)';
y_exacta = lg32 ;

% 2. Generar aproximación por mínimos cuadrados (Grado 1)
% p = polyfit(x, y_exacta, 1); 
y_aprox = rn32;

% =====================================================================
% 3. CÁLCULO DEL ERROR ABSOLUTO (RMSE PUNTO A PUNTO) Y RMSE GLOBAL
% =====================================================================
% Error absoluto en cada coordenada (mismas unidades de la gráfica)
error_punto_a_punto = abs(y_exacta - y_aprox); 

% RMSE Global (el promedio cuadrático de referencia)
rmse_global = sqrt(mean((y_exacta - y_aprox).^2));

% =====================================================================
% 4. Gráfica del Error en la misma escala de tus unidades
% =====================================================================
figure;

% Graficar la evolución de la distancia del error
h_error = plot(x, error_punto_a_punto, 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2.5);
hold on;

% Dibujar una línea horizontal con el valor del RMSE Global
h_rmse = line([x(1), x(end)], [rmse_global, rmse_global], ...
              'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);

grid on;
title('Error Absoluto - Aprox Numérica vs Aprox RN', 'FontSize', 12);
%xlabel('Eje X');
ylabel('$|y_{num} - y_{rn}|$', 'Interpreter', 'latex', 'FontSize', 14);

% Crear la leyenda combinando texto y el valor del RMSE calculado
texto_leyenda_rmse = sprintf('RMSE = %.4f', rmse_global);
legend([h_error, h_rmse], {'Diferencia absoluta por punto', texto_leyenda_rmse}, ...
       'Location', 'northwest', 'FontSize', 11);
