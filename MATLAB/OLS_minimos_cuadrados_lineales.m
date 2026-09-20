function y = OLS_minimos_cuadrados_lineales(x,fobj,f1,f2,f3,f4)

%% Datos de ejemplo (o generación de función objetivo)
x = x;
y_obj = fobj;

%% Definir 4 funciones base 
phi1 = f1;
phi2 = f2;  % 
phi3 = f3;
phi4 = f4;  % 
%% Construir matriz de diseño
Phi = [phi1, phi2, phi3, phi4];

%% Plot funciones base
plot(x,phi1,x,phi2,x,phi3,x,phi4)
%%
% plot(x,Phi(:,1),x,Phi(:,2),x,Phi(:,3),x,Phi(:,4))

%% Resolver por mínimos cuadrados (backslash - recomendado)
c = Phi \ y_obj;  % c = [c1; c2; c3; c4]

%% Evaluar ajuste
y_fit = Phi * c;

%% Plot de las funciones

plot(x,y_obj,x,y_fit)

%% Métricas de error
RMSE = sqrt(mean((y_obj - y_fit).^2));
MAE = mean(abs(y_obj - y_fit));
R2 = 1 - sum((y_obj - y_fit).^2) / sum((y_obj - mean(y_obj)).^2);

disp('Coeficientes:'); disp(c');
disp(['RMSE: ', num2str(RMSE)]);
disp(['MAE: ', num2str(MAE)]);
disp(['R²: ', num2str(R2)]);

end