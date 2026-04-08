# Estudo de Caso: Regressão Linear Visual

Este diretório contém uma análise conceitual explicativa de como os modelos de regressão aprendem e se comportam, junto a um script totalmente visual para demonstração.

## O que é Regressão?
A análise de regressão é uma técnica base (tanto em estatística Clássica quanto em Aprendizado de Máquina/IA) usada para modelar a relação entre uma **variável dependente** (geralmente designada como $y$, a coisa que queremos prever) e uma ou mais **variáveis independentes** (designadas como $X$, os dados que usamos para fazer a previsão).

No caso mais simples, a **Regressão Linear Simples**, o objetivo da máquina é encontrar a "melhor linha reta" que passe através de um conjunto de pontos num gráfico de dispersão, de modo que a distância total dos dados originais para essa linha desenhada seja a menor possível.

Matematicamente, a equação dessa reta procurada é:
$$y = wx + b$$

Onde:
- **$y$**: É a previsão em si.
- **$x$**: É a variável de entrada (o dado que o sistema está lendo naquele momento).
- **$w$ (Weight/Peso)**: É a inclinação da reta. Diz o quanto $y$ aumenta quando $x$ aumenta.
- **$b$ (Bias/Viés)**: É onde a reta corta o eixo vertical (o eixo $y$).

## A Mágica: Como o modelo "Aprende"?
Quando construímos um modelo de regressão sob a perspectiva do Machine Learning moderno, o sistema inicialmente é "burro": ele chuta valores perfeitamente aleatórios para $w$ (o Peso) e $b$ (o Viés). Isso resulta numa reta bizarra que corta o gráfico de forma inútil e distante dos dados reais.

Para que o modelo melhore sozinho até encontrar a linha perfeita, ele usa um algoritmo de otimização universal chamado **Gradiente Descendente** (Gradient Descent). O fluxo é o seguinte:

1. **Avaliando o Quão Ruim Estamos (Função de Custo):** A cada tentativa, a IA mede a distância da linha vermelha que ela desenhou até as bolinhas azuis (dados reais). Uma métrica comum é o Erro Quadrático Médio (MSE). Esse valor indica matematicamente a "dor" do algoritmo no momento.
2. **Descobrindo a Direção Certa (Cálculo do Gradiente):** O algoritmo calcula as derivadas dessa função de erro para os pesos atuais. Em termos simples, ele confere: *"Se eu inclinar a reta um pouquinho mais pra cima, o erro total cai ou sobe?"*. Essa bússola matemática aponta pra direção onde o gráfico de Erro cai ladeira abaixo.
3. **Dando um Passo (Atualização):** O sistema muda um pouco os valores da reta (ajusta $w$ e $b$) baseando-se nessa ladeira do erro que acabou de calcular. A agressividade dessa correção é dita pela *Taxa de Aprendizado (Learning Rate)*.
4. **Iteração Infinita (Épocas):** Esse processo todo é repetido dezenas ou milhares de vezes. A cada vez (que chamamos de *época* de treinamento), a reta dá uma escorregada mais próxima do centro perfeito do aglomerado de dados. Ao final, o erro para de cair, e nós dizemos que o modelo **Convergiu**. Ele aprendeu.

---

## Executando o Laboratório Visual
Você nos pediu algo que pudesse **ver rolando a regressão**. Desenvolvemos o script `regressao_visual.py` para te mostrar esse algoritmo por baixo dos panos sem jargão entediante, mas puramente com matemática interativa em tempo real.

### Requisitos Prévios
Certifique-se de que possui as bibliotecas comuns instaladas:
```bash
pip install numpy matplotlib
```

### Como Rodar
Num Terminal na pasta `estudo_de_caso` ou do root do VS Code, dispare:
```bash
python /home/uchoa/esbmc/pibic/estudo_de_caso/regressao_visual.py
```

### O que você verá?
Uma janela flutuante se abrirá:
- As bolinhas azuis não se movem. Elas são a amostra de problemas reais espalhados que inventamos aleatoriamente (mas as escondemos num molde de funçao onde o perfeito seria `peso=3` e `viés=4`).
- A **linha vermelha grossa** é o modelo de Machine Learning. Ela vai nascer num ângulo louco, e a cada décimo de segundo passará por um novo recálculo de **Gradiente Descendente**. 
- Você visualizará ela escorregando e rodando lentamente, perseguindo matematicamente os dados até travar elegantemente em seu centro ótimo. O título da janela mostrará o valor de viés e peso mudando no exato momento da otimização. O aprendizado da máquina em tempo real.
