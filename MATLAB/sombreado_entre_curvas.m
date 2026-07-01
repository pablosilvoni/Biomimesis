% 1. Definir los datos
x = 0:0.01:2;
y1 = 30*exp(2.1*x);
y2 = 30*exp(3.2*x);

% 2. Crear los vectores para la función fill
X = [x, fliplr(x)];            % Eje x hacia adelante y hacia atrás
Y = [y1, fliplr(y2)];          % Eje y de curva 1, luego curva 2 invertida

% 3. Dibujar el área sombreada y las líneas
figure;
fill(X, Y, [0.8 0.8 1]);       % Color azul claro para el área
hold on;
plot(x, y1, 'b', 'LineWidth', 2);
plot(x, y2, 'r', 'LineWidth', 2);

% --- Modificación para quitar la notación científica ---
ax = gca;                      % Obtiene los ejes de la figura actual
ax.YAxis.Exponent = 0;         % Elimina el multiplicador x10^4 de la esquina
ytickformat('%.0f')            % Muestra los números completos como enteros
% ------------------------------------------------------

grid on
hold off;