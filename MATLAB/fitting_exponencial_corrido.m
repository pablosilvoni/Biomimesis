% =========================================================================
% Ajuste Exponencial Personalizado en MATLAB: y = K*(exp(b*(x - x0)) - 1)
% =========================================================================

% 1. Datos experimentales
x = [1.0400; 1.0600; 1.0750; 1.0800; 1.1000; 1.1200; 1.1400; 1.1600; 1.1800];
y = [11.3600; 16.0900; 20.4100; 22.0000; 28.9700; 36.9000; 45.7000; 55.2700; 65.5400];

% 2. Definición del modelo de ajuste personalizado
ft = fittype('K*(exp(b*(x - x0)) - 1)', ...
             'independent', 'x', ...
             'coefficients', {'K', 'b', 'x0'});

% 3. Configuración de opciones y valores iniciales de búsqueda (StartPoint: [K, b, x0])
opts = fitoptions('Method', 'NonlinearLeastSquares');
opts.StartPoint = [30.0, 5.0, 0.95];

% 4. Ejecución del fitting
[fitresult, gof] = fit(x, y, ft, opts);

% 5. Mostrar resultados en consola
disp('--- Parámetros Ajustados ---');
disp(fitresult);

fprintf('R-cuadrado (R^2): %.6f\n', gof.rsquare);
fprintf('RMSE: %.6f\n\n', gof.rmse);

% Extraer coeficientes individuales
K  = fitresult.K;
b  = fitresult.b;
x0 = fitresult.x0;

% 6. Gráfico comparativo
figure;
plot(fitresult, x, y);
grid on;
title('Ajuste de Función Exponencial Personalizada');
xlabel('x');
ylabel('y');
legend('Datos experimentales', 'Ajuste: y = K*(exp(b*(x - x0)) - 1)', 'Location', 'NorthWest');