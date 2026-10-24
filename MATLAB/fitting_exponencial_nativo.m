% Datos
x = [1.0400; 1.0600; 1.0750; 1.0800; 1.1000; 1.1200; 1.1400; 1.1600; 1.1800];
y = [11.3600; 16.0900; 20.4100; 22.0000; 28.9700; 36.9000; 45.7000; 55.2700; 65.5400];

% Definición del modelo como función anónima: p(1)=K, p(2)=b, p(3)=x0
modelfun = @(p, x) p(1) * (exp(p(2) * (x - p(3))) - 1);

% Valores iniciales [K, b, x0]
p0 = [30.0, 5.0, 0.95];

% Ajuste no lineal
beta = nlinfit(x, y, modelfun, p0);

fprintf('K  = %.6f\n', beta(1));
fprintf('b  = %.6f\n', beta(2));
fprintf('x0 = %.6f\n', beta(3));