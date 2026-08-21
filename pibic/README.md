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
python -m pip install -e ".[test]"
pytest tests/ -v
```

O pacote base instala apenas o wrapper, que usa a biblioteca padrão do Python.
Testes, treino, extração de modelos e gráficos ficam em grupos opcionais:

```bash
python -m pip install -e ".[test,models,plots,python-frontend]"
# ou, para reproduzir o ambiente completo legado:
python -m pip install -r requirements.txt
```

Para gerar e inspecionar o wheel sem instalar o projeto:

```bash
python -m pip wheel . --no-deps --wheel-dir dist
python -m zipfile -l dist/core_verify-*.whl
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
custo explode: 4 passos custaram 909 s e 383 MB nesta máquina. IC3/PDR busca um
**invariante indutivo** sem construir uma fórmula monolítica com K cópias da
transição. Isso não torna a memória independente da profundidade: frames e
cláusulas ainda podem crescer durante a busca.

```bash
sudo apt-get install -y yosys        # traz o ABC embutido
cd ic3
python3 gen_transition_system.py --bits 16 -o cl_ddpg16.v
python3 validate_forward.py          # teste diferencial em 12 estados
./run_pdr.sh cl_ddpg16.v 1800
```

`validate_forward.py` não é opcional: ele compara o Verilog gerado com a
referência Q8.8 em 12 estados. A coincidência das 12 amostras detecta regressões,
mas não constitui prova de equivalência universal entre as implementações.

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

## Aplicação web do Cart-Pole

A visualização é uma exportação estática Next.js e requer Node.js 20.9 ou mais
recente. O `basePath` é opcional e permite publicar em um subdiretório:

```bash
cd cartpole/webapp
npm ci
npm run lint
npm run build
# PowerShell: $env:NEXT_PUBLIC_BASE_PATH = "/pibic"
# Bash:       NEXT_PUBLIC_BASE_PATH=/pibic npm run build
```

O conteúdo exportado fica em `cartpole/webapp/out/`.

---

## Limitações conhecidas

Registradas para que ninguém as descubra por acidente. Detalhe e evidência em
[`KANBAN.md`](KANBAN.md).

- **As Properties B e C originais do Cart-Pole eram vácuas, mas esse defeito já
  foi corrigido.** A B agora mede segurança em dois passos; a C foi
  reclassificada como teste da quantização, e a nova Property D depende da saída
  do ator. Evidência e discriminação estão em [`KB-C01` e `KB-C02`](KANBAN.md#raia-c--propriedades-formais).
- **A verificação em malha fechada continua limitada.** A propriedade atual cobre
  dois passos; o esboço de 50 passos não foi concluído, e as medições mostram a
  parede do BMC em 4 passos no modelo sintético e 5 frames no ator real.
- **Nenhum dos dois métodos decide o controlador DDPG real** — PDR não convergiu
  em 1802 s, BMC parou em 5 frames.
- **A verificação da Arduino-PID não é reproduzível** com o binário embarcado:
  `PID_v1.cpp` exige C++11 e a 6.8.0 não tem `--std` (`KB-D02`).
- **Perfis de caos com faixa contínua não decidem** sob `--floatbv`; só o perfil
  discreto (`Impulse`) decide, em 27 s.
- **Ainda há combinações conflitantes de versão e solver** entre artigo,
  apresentação e READMEs (`KB-D09`). Os vereditos corrigidos e seus logs estão em
  [`results/EVIDENCIAS.md`](results/EVIDENCIAS.md); números sem log não devem ser
  usados como evidência.
