import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    print("Iniciando a demonstração visual de Regressão Linear...")
    # Gerar dados aleatórios (distribuição parecida com uma reta)
    np.random.seed(42)  # para reprodutibilidade
    X = 2 * np.random.rand(100, 1)        # 100 pontos entre 0 e 2
    y = 4 + 3 * X + np.random.randn(100, 1) # Reta ideal: y = 4 + 3x (mais ruído Gaussiano simulando o mundo real)

    # Configurações do modelo inicial (valores aleatórios ruins)
    # A equação da reta é y = b + w*x
    # theta[0] = b (viés/intercepto), theta[1] = w (peso/inclinação)
    theta = np.random.randn(2, 1)
    
    # Adicionar termo x0 = 1 para o viés (b) em todos os pontos X
    # Isso facilita a multiplicação de matrizes: y = X_b * theta
    X_b = np.c_[np.ones((100, 1)), X]

    # Hiperparâmetros do treinamento
    eta = 0.05       # Taxa de aprendizado (tamanho do passo a cada ajuste)
    n_iterations = 60 # Número de vezes que vamos ajustar a reta (épocas)
    m = 100          # Número de instâncias/pontos de dados

    # --- Configuração Visual do Matplotlib ---
    plt.ion() # Ativa Modo interativo para animação em tempo real
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dos dados "reais" (em azul)
    ax.scatter(X, y, color='dodgerblue', edgecolor='k', label='Dados Observados (Dataset)', alpha=0.7, s=50)

    # Inicializar a linha da Regressão (em vermelho)
    x_line = np.array([[0], [2]])                      # Dois pontos extremos no eixo X para traçar a reta
    x_line_b = np.c_[np.ones((2, 1)), x_line]          # Adiciona o viés aos pontos da reta
    y_predict = x_line_b.dot(theta)                    # Primeira previsão (comleta/aleatória)
    
    line, = ax.plot(x_line, y_predict, color='crimson', linewidth=4, label='Modelo de Regressão (Aprendendo...)')

    # Estilização do gráfico
    ax.set_xlabel("X (Variável Independente)", fontsize=12, fontweight='bold')
    ax.set_ylabel("y (Variável Dependente que queremos prever)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    
    print("Iniciando o loop de Gradiente Descendente...")
    # Loop de Treinamento (Algoritmo do Gradiente Descendente)
    for iteration in range(n_iterations):
        # 1. Calcular o erro atual e a direção do ajuste (gradientes)
        # grad = 2/m * soma( (Previsão - Real) * X ) -> Derivada da Função de Custo (MSE)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        
        # 2. Atualizar os pesos (dar um passo na direção oposta ao erro)
        theta = theta - eta * gradients
        
        # 3. Atualizar a linha no gráfico com os novos parâmetros
        y_predict = x_line_b.dot(theta)
        line.set_ydata(y_predict)
        
        # Atualizar título mostrando o aprendizado em tempo real
        ax.set_title(f"Treinamento de Regressão Linear (Gradiente Descendente)\n"
                     f"Época: {iteration+1:02d} | "
                     f"Ajuste atual: Viés(b)={theta[0][0]:.2f}, Inclinação(w)={theta[1][0]:.2f}", fontsize=14)
        
        # Renderizar o frame da animação
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.12) # Pausa para conseguir ver a animação fluída a olho nu

    # Fim do treinamento
    plt.ioff() # Desliga o modo interativo
    ax.set_title(f"Treinamento Concluído!\n"
                 f"Pesos Finais: Viés={theta[0][0]:.2f} (Ideal era 4.0), Inclinação={theta[1][0]:.2f} (Ideal era 3.0)", 
                 fontsize=14, color='darkgreen', fontweight='bold')
    
    line.set_label('Modelo Final Treinado')
    ax.legend(loc="upper left", fontsize=11)
    print("Treinamento finalizado. Feche a janela do gráfico para encerrar o script.")
    plt.show()

if __name__ == "__main__":
    main()
