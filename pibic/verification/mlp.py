# MLP Simples - 2 inputs, 1 hidden layer com 3 neuronios, 1 output
# Propriedades de seguranca para ser verificado pelo ESBMC.

def relu(x: float) -> float:
    if x > 0:
        return x
    return 0.0

def hidden_layer(in1: float, in2: float) -> tuple[float, float, float]:
    # Neuronio 1
    w11: float =  0.5
    w12: float = -0.2
    b1: float =   0.1
    h1: float = relu(w11*in1 + w12*in2 + b1)
    
    # Neuronio 2
    w21: float = -0.3
    w22: float =  0.4
    b2: float =   0.0
    h2: float = relu(w21*in1 + w22*in2 + b2)
    
    # Neuronio 3
    w31: float =  0.1
    w32: float =  0.1
    b3: float =  -0.5
    h3: float = relu(w31*in1 + w32*in2 + b3)
    
    return h1, h2, h3

def output_layer(h1: float, h2: float, h3: float) -> float:
    # Neuronio de saida
    wo1: float =  0.8
    wo2: float =  0.2
    wo3: float = -0.1
    bo: float  =  0.05
    return relu(wo1*h1 + wo2*h2 + wo3*h3 + bo)

def mlp_forward(in1: float, in2: float) -> float:
    h1, h2, h3 = hidden_layer(in1, in2)
    return output_layer(h1, h2, h3)

def main() -> None:
    in1: float
    in2: float
    # Regiao de input: [0, 1]x[0, 1]
    if in1 >= 0.0 and in1 <= 1.0 and in2 >= 0.0 and in2 <= 1.0:
        out: float = mlp_forward(in1, in2)
        
        # Como todas as funcoes de ativacao sao ReLU e so multiplicacoes e somas ocorrem
        # A propria MLp tem limite inferior trivial >= 0
        assert out >= 0.0
        
        # Limite analitico superior
        # max_h1 = max(0.5*1 + -0.2*0 + 0.1) = 0.6
        # max_h2 = max(-0.3*0 + 0.4*1 + 0.0) = 0.4
        # max_h3 = max(0.1*1 + 0.1*1 + -0.5) = 0.0 (pois 0.2 - 0.5 < 0 -> ReLU = 0)
        # out = max(0.8*max_h1 + 0.2*max_h2 + -0.1*min_h3 + 0.05)
        # se w_o variavel eh negativa, minimizamos o h dele para maximizar out
        # min_h3 = 0.0, max_h1 = 0.6, max_h2=0.4
        # out_max <= 0.8*0.6 + 0.2*0.4 + -0.1*0 + 0.05 = 0.48 + 0.08 + 0.05 = 0.61
        assert out <= 0.61
main()
