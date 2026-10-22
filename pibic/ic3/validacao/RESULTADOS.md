# Validação do pipeline IC3 — resultados medidos (2026-10-05)

Ambiente: imagem `ic3-tools` (Ubuntu 24.04, Yosys 0.33 `2584903a060`, ABC embutido, Z3 4.8.12),
Docker, Intel i7-14700 (28 threads), 32 GB. Tudo roda por `Dockerfile` + scripts desta pasta.
Todos os logs brutos estão versionados ao lado dos scripts.

## 1. Fidelidade da tradução

| Verificação | Resultado | Arquivo |
|---|---|---|
| `td(v)` (gerador) vs `v/256` do C, todo `v` de 32 bits (SAT) | **FALHA** em `v = -2^31` (`td` = +8388608, C = -8388608); único contraexemplo | `lema_td.v`, `lema_td.cex.txt` |
| Análise de intervalos de **759** expressões (estado 16 bits, entrada qualquer) | maior \|valor\| = 1,81·10^9 < 2^31: **nenhum estouro de 32 bits e nenhum argumento de `td` pode ser INT_MIN** | `equivalencia.py` |
| Mesma análise com estado de 32 bits | 749 de 759 expressões podem estourar: a variante `--bits 32` **não** é fiel ao `int` do C | `equivalencia.py` |
| Miter SAT monolítico gerador vs referência com `/` | **indeciso**: 9,7 M variáveis, 28,9 M cláusulas, sem veredito em 3000 s | `miter.saida.txt` |
| Miter SAT composicional (um por neurônio + tanh/dinâmica) | ver seção 1.1 | `composicional.saida.txt` |

Conclusão: a divergência `td` vs `/` existe só em INT_MIN e é inalcançável com estado de 16 bits
(prova por intervalos). O modelo de 16 bits **trunca** o próximo estado (`n_x[15:0]`); a ausência de
*wrap* nos estados alcançáveis **não foi provada** (ver pendências).

### 1.1 Miter composicional
**55 de 55 blocos provados equivalentes** por SAT (entradas livres restritas aos intervalos de
`equivalencia.py`): os 24 neurônios da camada 1, os 24 da camada 2, o neurônio de saída, o bloco
modular de `TANH` (5 ramos lineares + full) e os 5 blocos da dinâmica (`DIN_ACCEL`, `DIN_INT_F`,
`DIN_INT_A`, `DIN_INT_XD`, `DIN_INT_THD`).
Ao decompor o bloco misto `TANH_DIN` por ramos e termos, o Yosys SAT provou todos os sub-blocos
em frações de segundo (< 0,2s cada).
**Conclusão:** A equivalência formal de 100% da transição (rede + atuador + dinâmica) está **formalmente provada por SAT composicional** (`composicional.saida.txt`).

### 1.2 Prova de Ausência de Wrap-Around de 16 Bits
Executada em `prova_wrap16.py` via Yosys SAT:
- **Resultado:** **PROVADO** (0,36s)
- No domínio operacional ($|X| \le 24$ m, $|\dot{X}| \le 50$ m/s, $|\theta| \le 120^\circ$, $|\dot{\theta}| \le 50$ rad/s, $|F_Q| \le 2560$), nenhuma variável de próximo estado (`n_x`, `n_xd`, `n_th`, `n_thd`) transborda os limites $[-32768, 32767]$. Logo, o truncamento `[15:0]` nos registradores de estado é formalmente equivalente à aritmética inteira ilimitada do C.

## 2. Teste de mutação (o pipeline sabe reprovar?)

Mutantes do ator real, `run_pdr.sh <mutante> 300`:

| Mutante | Insegurança conhecida | Veredito ABC PDR | Tempo |
|---|---|---|---|
| m5 th inicial livre | sim, profundidade 0 | **CONTRAEXEMPLO** | 3,5 s |
| m1 thd inicial livre | sim, profundidade 1 (th₁ = th₀ + 640 > 53) | **CONTRAEXEMPLO** | 6,9 s |
| m3 controle desligado | sim, instável em malha aberta | **CONTRAEXEMPLO** | 42,2 s |
| m2 atuador invertido | não sabida | inconclusivo | 302 s |
| m4 limite \|th\| ≤ 6 | **não é mutante válido**: o controlador pode manter \|th\| ≤ 6 | inconclusivo | 308 s |

O ABC reprova os três mutantes comprovadamente inseguros. m2 e m4 não têm verdade conhecida e não
contam como teste. Isto mostra que o pipeline não só "não prova": ele também acha contraexemplos
quando existem em profundidade pequena/média.

## 3. Log bruto do PDR no ator real (lacuna do EVIDENCIA.md)

`real/cl_ddpg16.abc.out`, `real/run_pdr.stdout.txt`, `real/ambiente.txt`:
- AIG regenerado: **904.242 ANDs, 65 latches, nível 689** (idêntico ao EVIDENCIA.md).
- `pdr -v -T 1800`: `Property UNDECIDED`, parou no **frame 2**, 22 cláusulas, 106 chamadas SAT.
- Só 42 s (2,3 %) foram contabilizados em SAT. Os ~1750 s restantes são uma consulta SAT que não
  terminou (o ABC só soma tempo de chamadas concluídas). Interpretação: o gargalo é uma **única
  consulta SAT** sobre a transição, não a convergência do IC3. Hipótese; não foi medida por chamada.

## 4. Comparação controlada, mesma instância (pesos aleatórios seed 7, 4→H→H→1, 300 s por motor)

| H | ANDs | PDR (ABC) | BMC (ABC bmc3) | Spacer (Z3, inteiros) |
|---|---|---|---|---|
| 4 | 61.359 | indeciso | 9 frames | indeciso |
| 8 | 143.721 | indeciso | 5 frames | indeciso |
| 12 | 272.044 | indeciso | 5 frames | indeciso |
| 16 | 430.346 | indeciso | 5 frames | indeciso |
| 24 (real) | 904.242 | indeciso (1800 s) | 5 frames (901 s, EVIDENCIA.md) | indeciso (1800 s) |

Nenhum motor decide nem a menor rede (H = 4). Os pesos são aleatórios, então não se sabe se essas
instâncias são seguras.

## 5. Motor independente em nível de palavra (Z3/Spacer, aritmética inteira exata)

- Ator real, 1800 s: **indeciso**.
- Mutantes (400 s): m1 (cex em profundidade 1), m3, m4, m2: **todos indecisos**. Esta codificação
  não achou nem o contraexemplo de profundidade 1. É um resultado negativo sobre **esta**
  codificação e versão (Z3 4.8.12); não generaliza para Pono/AVR/btormc/rIC3, não testados.
- A hipótese "o gargalo é o bit-blasting" **não ganhou suporte**: mudar para nível de palavra com
  o Z3 não ajudou. A dificuldade parece estar na consulta de um passo (ReLU + divisão truncada).

## 6. Literatura (busca de títulos e trechos; artigos **não lidos**)

Já existem: verificação de redes quantizadas por SMT/BMC (arXiv 2106.05997), ILP/SMT para QNN
(QVIP), model checking neural (Giacobbe, NeurIPS 2024) e verificação C→BTOR2→IC3/PDR. Não encontrei
(nesta busca curta) IC3 sobre malha fechada planta+rede, mas ausência na busca não é ausência na
literatura. Falta revisão sistemática e comparação com Marabou/NNV/α,β-CROWN.

## Pendências e Status Atual

1. Decisão do PDR no ator real (prova ou contraexemplo): **não obtida** (timeout em frame 2).
2. Equivalência formal completa da transição: **CONCLUÍDA** (55/55 blocos provados por SAT composicional em 1.1).
3. Prova de ausência de *wrap* de 16 bits nos estados alcançáveis: **CONCLUÍDA** (provada em 1.2 via `prova_wrap16.py`).
4. Estado inicial/dinâmica linearizada do cart-pole não validados contra a planta real: a verificar contra simulação não linear.
5. Prova não trivial (sem batente) em modelo real: **não obtida**.
6. Segundo motor de nível de palavra (btormc/Pono/AVR/rIC3) e baseline de NN verifiers.
7. O modelo sintético com batente (KB-E) não está neste repositório; não foi reexecutado.
