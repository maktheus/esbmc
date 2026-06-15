# ESBMC — Notas de Verificação do Cart-Pole DQN

Este documento explica as escolhas técnicas de verificação formal feitas neste projeto,
incluindo os flags do ESBMC, a metodologia de harnesses e as propriedades verificadas.

---

## Flags usados em todas as verificações

```bash
esbmc harness.c --no-unwinding-assertions --boolector
```

### `--no-unwinding-assertions`

**O que faz:**
Quando ESBMC encontra um loop `for(i=0; i<N; i++)`, ele o desdobra (*unwinds*) em N
cópias do corpo. Por padrão, ao final do desdobramento ele insere uma assertion automática:

```c
// assertion gerada automaticamente (SEM o flag):
assert(i < N);  // "provei que o programa nunca precisou de mais iterações"
```

Se essa assertion puder falhar (i.e., o ESBMC não tem certeza de que o loop é limitado),
ele retorna `VERIFICATION FAILED` mesmo que a propriedade real seja verdadeira — um
**falso negativo conservador**.

**Com `--no-unwinding-assertions`:** essas assertions automáticas não são inseridas.
O ESBMC verifica apenas o que foi explicitamente escrito nos harnesses.

**Por que é seguro aqui:**
Nossos harnesses C são código **completamente linear** — não há nenhum loop `for` ou
`while`. Cada neurônio da rede é uma linha de C individual:

```c
// Exemplo: camada 1, neurônio 0
int pre1_0 = (x*(w00))/256 + (xd*(w01))/256 + (th*(w02))/256 + (thd*(w03))/256 + (b0);
int h1_0   = pre1_0 > 0 ? pre1_0 : 0;  // ReLU
// ... 23 neurônios mais, cada um em sua própria linha ...
```

Sem loops → sem unwind assertions espúrias → o flag não compromete a soundness.

**Quando NÃO usar:**
Se o harness tiver loops (ex: verificação multi-passo ou arrays), remova o flag.
O ESBMC adicionará as assertions de guarda necessárias. Ajuste o bound com `--unwind N`.

---

### `--boolector`

**O que faz:** seleciona o solver SMT Boolector (em vez de Z3 ou MathSAT).

**Por que Boolector:**
- Especializado em aritmética de bits e inteiros (bitvector theory)
- Nossos harnesses usam aritmética inteira Q8.8 (tudo são `int`)
- Boolector é significativamente mais rápido que Z3 para este domínio

**Alternativas:**
- `--z3`: mais lento para inteiros, mas suporta floats (`--floatbv`)
- `--bitwuzla`: sucessor do Boolector, ainda mais rápido (disponível em versões mais novas)

---

## Por que aritmética inteira Q8.8?

O ESBMC trabalha bem com inteiros (bitvectors) mas é muito lento com floats.
Para verificar a rede neural, convertemos todos os pesos para inteiros:

```
float original: w = 0.3742...
Q8.8 (scale=256): w_q = round(0.3742 * 256) = 96

Multiplicação na rede:
  float: h = w * x = 0.3742 * 0.5 = 0.1871
  Q8.8:  h = (96 * 128) / 256 = 48 ≈ 0.1875  (erro < 0.3%)
```

A divisão por 256 em C trunca em direção a zero (`int(a/b)`), diferente de Python
`//` que faz floor. O `c_div()` em `verify_closed_loop.py` implementa esse comportamento
corretamente para manter a soundness da aritmética de intervalo.

---

## `__ESBMC_assume` e `__ESBMC_assert`

```c
void __ESBMC_assume(_Bool cond);  // restringe o espaço de busca
void __ESBMC_assert(_Bool cond, const char *msg);  // propriedade a verificar
int  nondet_int(void);  // variável simbólica (todos os valores possíveis)
```

**`nondet_int()`:** cria uma variável que o ESBMC trata como podendo ser qualquer valor
inteiro. É a versão formal de "para todo x ∈ ℤ".

**`__ESBMC_assume(cond)`:** descarta todos os caminhos onde `cond` é falso. Restringe
o espaço de busca sem alterar a propriedade. Usado para codificar o domínio do sistema:

```c
int th = nondet_int();
__ESBMC_assume(th >= -53 && th <= 53);  // θ ∈ [-12°, +12°] em Q8.8
```

**`__ESBMC_assert(cond, msg)`:** a propriedade a verificar. Se ESBMC encontrar uma
atribuição de variáveis que satisfaz todos os `assume` mas viola o `assert`, retorna
`VERIFICATION FAILED` com um contraexemplo concreto.

---

## Aritmética de intervalo para bounds de pré-ativação

Os `__ESBMC_assume` nos bounds de `pre1_i` e `pre2_j` não são restrições arbitrárias
— são calculados analiticamente pela propagação de intervalo em `compute_pre_bounds()`:

```
Para cada neurônio i na camada 1:
  pre_i = bias_i + Σ_k w_{ik} * x_k

  Se w_{ik} ≥ 0:  contribuição = w * [lo_k, hi_k]  →  [w*lo_k, w*hi_k]
  Se w_{ik} < 0:  contribuição = w * [lo_k, hi_k]  →  [w*hi_k, w*lo_k]  (inverte)

  Somando todas as contribuições:
    lo_pre_i = bias_i + Σ_k min(w*lo_k, w*hi_k)
    hi_pre_i = bias_i + Σ_k max(w*lo_k, w*hi_k)
```

Esses bounds são **sobreapróximações** (contêm o range real). O ESBMC pode checar apenas
o subconjunto viável — guiando o solver de forma muito mais eficiente.

**Importância da soundness:** usar `//` de Python (floor toward -∞) em vez de `int(a/b)` 
(truncation toward zero, como C faz) tornaria os bounds menores para termos negativos,
possivelmente excluindo valores reais e tornando a verificação não-soa.

---

## Propriedades verificadas

### Property A — Direção errada

```
Domínio:  θ > 0.10 rad  E  θ̇ ≥ 0   (pêndulo inclinando para a direita)
Harness:  executa passagem completa da rede DQN (pesos Q8.8)
Assert:   action == 1  (deve empurrar à direita para corrigir)

Resultado: TIMEOUT (120s) — espaço de busca muito grande para o lado direito
           FAILED para o lado esquerdo — contraexemplo em x=-1.25, θ=-0.19 rad
```

### Property B — Segurança em 1 passo

```
Domínio:  s₀ ∈ S_safe (domínio completo)
Harness:  DQN → ação → dinâmica linearizada (sin≈θ, cos≈1) → s₁
Assert:   |θ₁| ≤ 12°  (permanece seguro após 1 passo)

Resultado: FAILED — contraexemplo em θ̇=-4.59 rad/s
           Velocidade angular extrema faz θ cruzar o limite em 1 passo
           independentemente da ação do controlador
```

### Neurônios mortos / saturação

```
Mortos:    ESBMC SUCCESS  → neurônio nunca ativa → candidato a poda
           ESBMC FAILED   → neurônio VIVO (existe entrada que o ativa)

Saturação: ESBMC SUCCESS  → pre > 0 sempre → ReLU nunca corta → saturado
           ESBMC FAILED   → neurônio às vezes desativa → normal
```

---

## Estrutura dos arquivos de verificação

```
verify_dead_neurons.py    # verifica se algum neurônio nunca ativa
verify_saturation.py      # verifica se algum neurônio está sempre ativo
verify_closed_loop.py     # propriedades A e B em malha fechada
onnx_controller_extractor.py  # extrai pesos do ONNX para dicts Python
closedloop_esbmc_stub.c   # esboço de verificação multi-passo (trabalho futuro)
closed_loop_results.json  # resultados gravados pelo verify_closed_loop.py
```

---

## Para rodar as verificações

```bash
# Neurônios mortos (camada 1, ~30s)
python verify_dead_neurons.py

# Saturação
python verify_saturation.py

# Malha fechada (Properties A e B, ~4 min)
python verify_closed_loop.py

# Regenerar dados do webapp após verificação
python generate_webapp_data.py
```

Requer ESBMC em `../QNNVerifier/esbmc-6.8.0/esbmc`.
