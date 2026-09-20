% 1. Definir los datos de la curva original
x = linspace(0, 2, 200);
y_original = 15*exp(2*x) + 15*exp(3.2*x);

% 2. Generar el ruido
desviacion_estandar = 100;
ruido = desviacion_estandar * randn(size(y_original));

% 3. Agregar el ruido a la curva
y_ruidosa = y_original + ruido;

% 4. Graficar para comparar
plot(x, y_original, 'b-', 'LineWidth', 2);
hold on;
plot(x, y_ruidosa, 'r.');
legend('Curva original', 'Curva con ruido');
grid on;
a = x';
b = y_ruidosa';
y_ruidosa = [a b];
writematrix(y_ruidosa,'y_objetivo.xls')
hold off;