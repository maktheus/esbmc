# Auditoria da rodada canônica

## Escopo

Rodada executada em 30/08/2026, isolada em `evidencias/final_2026`. A rodada
canônica inicial preservou os scripts originais; em seguida, os dois scripts de
auditoria estrutural foram corrigidos no diretório `cartpole/` e suas versões
anteriores foram preservadas em `legacy/`. Todos os harnesses desta rodada foram
gerados a partir de `webapp/public/ddpg_weights_q88.json`.

## Quantização

O script `scripts/canonical_quantization.py` reproduziu a amostragem com
`RandomState(42)`, quatro variáveis uniformes e 10.000 estados. Os valores
recalculados coincidem com `cartpole/quantization_report.json`:

| métrica | valor |
|---|---:|
| erro absoluto médio | 0,0778007758573 N |
| mediana | 0,0390625 N |
| p95 | 0,2262005212121 N |
| p99 | 1,0927201961733 N |
| máximo | 7,7589197857363 N |
| máximo relativo a 10 N | 77,5892% |

O pior estado da amostra é o índice 8243:

```text
estado = [-0,822201926522; -2,491220208961; 0,005176165709; 0,777817617120]
Float32 = +6,430794785736 N
Q8.8    = -1,328125000000 N
erro    = 7,758919785736 N
z_Q8.8  = -35
```

O erro não é apenas um pequeno erro de arredondamento: neste estado ocorre
inversão do sinal da ação. Isso é compatível com uma mudança de região de
ativação ReLU e com a aproximação de `tanh` perto da origem. O relatório deve
informar a cauda (p95/p99) e o máximo, evitando chamar o resultado de
“fidelidade total” em relação ao Float32. A afirmação correta é que o harness e
o runtime Q8.8 compartilham a mesma implementação quantizada.

Uma varredura independente de `z=-10000,...,10000` está em
`outputs/tanh_approximation.json`: o erro absoluto máximo da aproximação de
`tanh` é `0,0457493` (4,5749% do intervalo unitário), em `z=-281`. Portanto,
“erro inferior a 3%” não é sustentado pela implementação atual.

## Reprodução dos contraexemplos

`canonical_concrete_checks.py` gerou dois harnesses C e os verificou com ESBMC
6.8.0 e `--no-unwinding-assertions --boolector`. Como as entradas são fixas,
o ESBMC simplificou as fórmulas antes de invocar o solver; portanto, esse
resultado não demonstra que Boolector esteja disponível no executável.

| check | estado Q8.8 | asserção fixa | resultado | tempo |
|---|---|---|---|---:|
| A-direita | `[-3,-1260,28,0]` | `z < 0` | SUCCESSFUL | 0,609 s |
| B-segurança | `[-193,-1004,-47,-390]` | `theta_new < -53` | SUCCESSFUL | 0,313 s |

Esses `SUCCESSFUL` significam que as asserções foram provadas para os estados
fixados, validando a reprodução da violação publicada. Não são provas
universais das propriedades A ou B.

Uma tentativa universal curta foi registrada em
`outputs/short_universal_checks.json`. O executável Windows informa que o
Boolector não foi compilado (`The boolector solver has not been built into this
version of ESBMC`); por isso, a rodada também tentou Z3:

- C — limite de força, Z3: `SUCCESSFUL` em 3,5 s;
- A-direita universal, Z3: `TIMEOUT` em 12 s;
- ambas as tentativas Boolector: `UNKNOWN` por solver indisponível, não por
  timeout.

Logo, qualquer resultado histórico atribuído a Boolector precisa guardar o
binário efetivamente usado. O arquivo sem extensão `QNNVerifier/.../esbmc` é um
ELF Linux; o `esbmc.exe` Windows disponível nesta máquina não contém Boolector.

## Auditoria estrutural corrigida

Os scripts corrigidos constroem a camada 2 a partir do mesmo vetor simbólico da
camada 1, usam os bounds derivados do grafo completo e registram comando, tempo,
status e contraexemplo. Com Z3 e os limites declarados:

| propriedade | resultado | interpretação |
|---|---|---|
| Atividade, camada 1 | 24/24 `FAILED` | há contraexemplo ativo para cada neurônio |
| Atividade, camada 2 | 17/24 `FAILED`; 7 `TIMEOUT` | sete casos permanecem inconclusivos |
| Não saturação, camada 1 | 24/24 `FAILED` | há contraexemplo com pré-ativação negativa |
| Não saturação, camada 2 | 22/24 `FAILED`; 2 `TIMEOUT` | dois casos permanecem inconclusivos |
| Saída `z >= 0` | `FAILED` | existe contraexemplo |
| Saída `z <= 0` | `TIMEOUT` | não permite concluir |

Os sumários auditáveis estão em `outputs/corrected_dead_neurons.json` e
`outputs/corrected_saturation.json`; os arquivos C e logs individuais ficam em
`harnesses/` e `logs/`. Assim, não há base para declarar “48/48 neurônios
ativos” ou responsividade global: os resultados corretos são parciais e
explicitamente condicionais aos veredictos obtidos.

## Divergência de malha fechada

O texto de apresentação registra `TIMEOUT` para A-direita e A-esquerda, mas o
JSON DDPG registra `FAILED` para ambas. O resultado A-direita possui estado
completo e foi reproduzido nesta rodada. O resultado A-esquerda possui somente
`z=27`, sem estado completo, e não pode ser reproduzido até que o log bruto ou
um novo ESBMC counterexample seja preservado.

Os campos `time_seconds` do JSON histórico também não são auditáveis: o script
atual não registra tempo e os valores parecem estar em outra unidade ou vir de
uma rodada diferente.

## Bloqueios científicos

- O checkpoint `.pth` ainda precisa ser lido diretamente para confirmar a
  origem dos JSONs exportados.
- A rodada universal da malha fechada A-direita ainda está inconclusiva por
  `TIMEOUT`; não se deve convertê-la em prova ou refutação.
