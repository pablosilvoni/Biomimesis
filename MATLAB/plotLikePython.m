function plotLikePython(r, xn)
    % Crear figura con estilo matplotlib             
    % CORRECTO:
    fig = figure('Color', 'white', ...  % Coma y paréntesis balanceados
             'Position', [100 100 1000 600]);


    % Subplot 1: Estilo matplotlib por defecto
    ax1 = subplot(2, 3, 1);
    scatter(ax1, r, xn, 20, 'filled', ...  % size=20 como matplotlib
            'MarkerFaceColor', [0.1216 0.4667 0.7059], ...  #1f77b4 en RGB
            'MarkerFaceAlpha', 0.8);
    xlim(ax1, [0 4]); ylim(ax1, [0 1]);
    grid(ax1, 'on'); grid(ax1, 'minor');
    title(ax1, 'Estilo Matplotlib Default');
    
    % Subplot 2: Estilo que usaste en Python (α=0.3)
    ax2 = subplot(2, 3, 2);
    scatter(ax2, r, xn, 0.5, 'filled', ...
            'MarkerFaceColor', 'blue', ...
            'MarkerFaceAlpha', 0.3, ...
            'MarkerEdgeColor', 'none');
    xlim(ax2, [0 4]); ylim(ax2, [0 1]);
    grid(ax2, 'on');
    title(ax2, 'α=0.3 (tu código Python)');
    
    % Subplot 3: Para 100,000 puntos optimizado
    ax3 = subplot(2, 3, 3);
    scatter(ax3, r, xn, 0.1, 'filled', ...
            'MarkerFaceColor', [0 0.447 0.741], ...  % MATLAB blue
            'MarkerFaceAlpha', 0.05);  % Muy transparente
    xlim(ax3, [0 4]); ylim(ax3, [0 1]);
    grid(ax3, 'on');
    title(ax3, 'Optimizado 100K puntos (α=0.05)');
    
    % Subplot 4: Con color según r
    ax4 = subplot(2, 3, 4);
    scatter(ax4, r, xn, 0.5, r, 'filled');  % Color por r
    colormap(ax4, jet);
    colorbar(ax4);
    xlim(ax4, [0 4]); ylim(ax4, [0 1]);
    grid(ax4, 'on');
    title(ax4, 'Color por valor de r');
    
    % Subplot 5: Solo región caótica
    ax5 = subplot(2, 3, 5);
    idx = r > 3.5;
    scatter(ax5, r(idx), xn(idx), 0.1, 'filled', ...
            'MarkerFaceColor', 'red', ...
            'MarkerFaceAlpha', 0.1);
    xlim(ax5, [3.5 4]); ylim(ax5, [0 1]);
    grid(ax5, 'on');
    title(ax5, 'Solo región caótica');
    
    % Subplot 6: Usando plot (más rápido)
    ax6 = subplot(2, 3, 6);
    plot(ax6, r, xn, '.', 'Color', [0 0 0], 'MarkerSize', 0.1);
    xlim(ax6, [0 4]); ylim(ax6, [0 1]);
    grid(ax6, 'on');
    title(ax6, 'Usando plot() - Más rápido');
    
    % Ajustar espaciado
    sgtitle('Comparación de Estilos de Visualización', 'FontSize', 14);
end