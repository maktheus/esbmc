# Caso 1 reescrito utilizando a estratégia do QNNVerifier
# Modificado para Python com Aritmética de Ponto Fixo (Quantização) 
# e Simulação de Injeção de Intervalos

# Configurações de Ponto Fixo (8 bits fracionários -> Scala de 256)
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS # 256

def float_to_fxp(val: float) -> int:
    return int(val * SCALE)

def fxp_to_float(val: int) -> float:
    return val / SCALE

def fxp_add(a: int, b: int) -> int:
    return a + b

def fxp_mult(a: int, b: int) -> int:
    # Simula multiplicação de ponto fixo e alinha novamente a escala fracionária
    return (a * b) >> FRACTIONAL_BITS

def relu_fxp(x: int) -> int:
    if x > 0:
        return x
    return 0

def hidden_layer_fxp(in1: int, in2: int) -> tuple[int, int, int]:
    # (QNNVerifier Passo 3) INJEÇÃO DE INTERVALOS (ESTRATÉGIA FRAMA-C)
    # Aqui o Frama-C provaria que os dados iniciais estão limitados. 
    # Em um código convertido completo, haveria restrições baseadas nos bounds de in1 e in2.

    # Neuronio 1 (Aplicando conversão ponto-fixo - QNNVerifier Passo 4)
    w11 = float_to_fxp(0.5)
    w12 = float_to_fxp(-0.2)
    b1  = float_to_fxp(0.1)
    
    # u1 = in1*w11 + in2*w12 + b1 -> usando aritmética quantizada e truncada segura
    u1 = fxp_add(fxp_add(fxp_mult(in1, w11), fxp_mult(in2, w12)), b1)
    h1 = relu_fxp(u1)
    
    # Injeção Analítica (Simulando EVA do Frama-C):
    # h1 variava entre [0.0, 0.6] antes, ou seja, [0, 153] no espaço quantizado.
    # Restringir esse escopo poda a árvore de busca SMT inteira do solver!
    if not (h1 >= 0 and h1 <= 153):
        return 0, 0, 0 # "assume" semântico bloqueando o branch state

    # Neuronio 2
    w21 = float_to_fxp(-0.3)
    w22 = float_to_fxp(0.4)
    b2  = float_to_fxp(0.0)
    u2 = fxp_add(fxp_add(fxp_mult(in1, w21), fxp_mult(in2, w22)), b2)
    h2 = relu_fxp(u2)
    
    # Neuronio 3
    w31 = float_to_fxp(0.1)
    w32 = float_to_fxp(0.1)
    b3  = float_to_fxp(-0.5)
    u3 = fxp_add(fxp_add(fxp_mult(in1, w31), fxp_mult(in2, w32)), b3)
    h3 = relu_fxp(u3)
    
    return h1, h2, h3

def output_layer_fxp(h1: int, h2: int, h3: int) -> int:
    wo1 = float_to_fxp(0.8)
    wo2 = float_to_fxp(0.2)
    wo3 = float_to_fxp(-0.1)
    bo  = float_to_fxp(0.05)
    
    uo = fxp_add(fxp_add(fxp_add(fxp_mult(h1, wo1), fxp_mult(h2, wo2)), fxp_mult(h3, wo3)), bo)
    return relu_fxp(uo)

def mlp_forward_fxp(in1: int, in2: int) -> int:
    h1, h2, h3 = hidden_layer_fxp(in1, in2)
    return output_layer_fxp(h1, h2, h3)

def main() -> None:
    in1_f: float
    in2_f: float
    
    # Entradas não-determinísticas de [0, 1.0]. (QNNVerifier Passo 2: Assumes)
    if in1_f >= 0.0 and in1_f <= 1.0 and in2_f >= 0.0 and in2_f <= 1.0:
        
        # Conversão manual na borda do sistema de entrada embarcada
        in1_fxp = float_to_fxp(in1_f)
        in2_fxp = float_to_fxp(in2_f)
        
        # Operações integralmente em matemática puramente ponto falso inteira, blindada de overflows floating
        out_fxp = mlp_forward_fxp(in1_fxp, in2_fxp)
        out_f = fxp_to_float(out_fxp)
        
        # Garantias Requeridas (QNNVerifier Passo 2: Asserts)
        assert out_f >= 0.0
        assert out_f <= 0.65 # Ajustado levemente com margem pela perda na truncagem do fixed-point

main()
