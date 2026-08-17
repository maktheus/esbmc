# Verificação Formal de IA com ESBMC — PIBIC

Aplicação de *bounded model checking* (ESBMC) e *model checking* indutivo
(IC3/PDR) a sistemas de IA: redes neurais quantizadas, kernels de inferência,
agentes neuro-simbólicos e controle híbrido.

> **Estado do backlog:** [`KANBAN.md`](KANBAN.md). Cada tarefa carrega o arquivo
> e a linha que a originaram, e o quanto essa origem foi checada — verificado por
> execução, relatado sem confirmação, ou não auditado.

---

## Início rápido

```bash
git clone --recurse-submodules <url>        # o submodulo famous_pid importa
cd esbmc/pibic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

O binário do ESBMC **já vem no repositório**
(`QNNVerifier/esbmc-6.8.0/esbmc`, Linux x86_64), então não há build obrigatório.
O wrapper o localiza sozinho nesta ordem:

1. `$ESBMC_BIN`
2. `esbmc` no `$PATH`
3. build local em `../build/src/esbmc/esbmc`
4. o binário embarcado

Se nenhum existir, a suíte **pula** com motivo explícito em vez de acumular erros.

### Quando a 6.8.0 não basta

A árvore-fonte em `../` é a **8.0.0** e não está compilada. Algumas flags só
existem nela: `--std` (necessária para C++11), `--multi-property`, `--cvc4`. A
6.8.0 rejeita com `unrecognised option` e código de saída 64 — que o wrapper
reporta como `USAGE_ERROR`, não como "propriedade violada".

```bash
cd .. && mkdir -p build && cd build
cmake .. -GNinja -DENABLE_PYTHON_FRONTEND=ON && ninja      # ~1-4 h
```

---

## Mapa do repositório

| Diretório | Conteúdo |
|---|---|
| `core_verify/` | **Wrapper canônico.** `esbmc_caller.py` (execução, status tri-estado) e `SMT_feedback_parser.py` (contraexemplos) |
| `tests/` | Suíte pytest. `test_ground_truth.py` é a rede de segurança do wrapper |
| `cases/` | Os quatro casos de uso + `llm_ffn_verification/` (FFN de GPT-2/Llama) |
| `cartpole/` | RL DQN/DDPG: treino, quantização Q8.8, harnesses ESBMC, webapp Next.js |
| `ic3/` | **Verificação ilimitada** via Verilog → yosys → AIGER → `abc pdr` |
| `verification/` | Harnesses de NN e o PID da Arduino (submódulo `famous_pid`) |
| `teste_mlp/` | Pipeline XOR: treino → ONNX → quantização → verificação |
| `artigo/`, `apresentacao/` | LaTeX (abnTeX2) |
| `results/` | Logs, CSVs e relatórios |
| `QNNVerifier/` | **Terceiros**, vendorizado. 308 MB dos 322 MB do diretório |

Há **duas taxonomias concorrentes**: os diretórios legados `1_python_models/` …
`4_control_system/` e a estrutura `cases/` + `core_verify/` do `roadmap.md`.
Ambas coexistem com código duplicado; a segunda é a canônica. Ver `KB-F04`.

---

## Interpretando um resultado

O ponto mais importante da API, e a origem do defeito mais sério que a auditoria
encontrou:

```python
from core_verify.esbmc_caller import run_esbmc, SAFE, UNSAFE

r = run_esbmc("harness.c", z3=True, floatbv=True, unwind=10)

r.status       # SAFE | UNSAFE | PARSE_ERROR | USAGE_ERROR | TIMEOUT | UNKNOWN
r.verificou    # True somente em SAFE ou UNSAFE
```

**Nunca teste `not r.is_safe` para concluir que um bug foi encontrado.** Esse
valor é falso também quando o ESBMC não rodou — flag inválida, erro de parsing,
arquivo ausente. Era assim que a suíte ficava verde sem verificar nada. Use
`r.status == UNSAFE`.

`TIMEOUT` é **indeciso**, nunca "seguro".

### Flags que não existem

`--bounds-check` e `--pointer-check` **não são reconhecidas pelo ESBMC**: esses
checks são ligados por padrão e a ferramenta só expõe as formas negativas.
No wrapper: `no_bounds_check=True`, `no_pointer_check=True`.

---

## Verificação ilimitada (`ic3/`)

O BMC responde "seguro até K passos". Para malha fechada isso não basta, e o
custo explode: 4 passos custaram 909 s e 383 MB nesta máquina. IC3/PDR devolve
um **invariante indutivo** — "seguro para sempre" — com memória independente da
profundidade da prova.

```bash
sudo apt-get install -y yosys        # traz o ABC embutido
cd ic3
python3 gen_transition_system.py --bits 16 -o cl_ddpg16.v
python3 validate_forward.py          # exige 12/12 estados bit-exatos
./run_pdr.sh cl_ddpg16.v 1800
```

`validate_forward.py` não é opcional: ele prova que o Verilog gerado computa a
**mesma** aritmética Q8.8 do harness C verificado. Sem isso, o resultado seria
sobre outro controlador.

Resultados medidos em [`ic3/EVIDENCIA.md`](ic3/EVIDENCIA.md).

---

## Ambiente registrado

Toda medição deve trazer o ambiente junto — sem isso o número não é comparável.
O padrão a seguir é [`cartpole/ESBMC_NOTES.md`](cartpole/ESBMC_NOTES.md) e
[`ic3/EVIDENCIA.md`](ic3/EVIDENCIA.md): versão do ESBMC, solver, flags
completas, timeout, CPU/RAM e seed.

O repositório ainda contém **combinações conflitantes** de versão e solver entre
o artigo, a apresentação e os READMEs. Reconciliar é `KB-D09`.

---

## Compilando o artigo

```bash
cd artigo && pdflatex relatorio && bibtex relatorio && pdflatex relatorio && pdflatex relatorio
```

Requer `abntex2` e `abntex2-alf.bst` (incluso).

---

## Limitações conhecidas

Registradas para que ninguém as descubra por acidente. Detalhe e evidência em
[`KANBAN.md`](KANBAN.md).

- **Duas das três propriedades do cartpole são vácuas** (`KB-C01`, `KB-C02`),
  confirmado por execução: a Property B não depende da saída do controlador, e
  a Property C vale por construção do `tanh`.
- **A verificação em malha fechada é de 1 passo.** O harness de 50 passos nunca
  foi implementado, e a medição mostra que não era alcançável com BMC.
- **Nenhum dos dois métodos decide o controlador DDPG real** — PDR não convergiu
  em 1802 s, BMC parou em 5 frames.
- **A verificação da Arduino-PID não é reproduzível** com o binário embarcado:
  `PID_v1.cpp` exige C++11 e a 6.8.0 não tem `--std` (`KB-D02`).
- **Perfis de caos com faixa contínua não decidem** sob `--floatbv`; só o perfil
  discreto (`Impulse`) decide, em 27 s.
- **Vários números do artigo não têm evidência no repositório** (`KB-D01` a
  `KB-D06`), e a tabela do GEMM contradiz o único CSV medido por 9× e 27×.
