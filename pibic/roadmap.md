# Roadmap: Verificação Formal em Inteligência Artificial com ESBMC

Este documento define os próximos passos para solidificar a aplicação do ESBMC na verificação de sistemas híbridos de IA Generativa e Controle (referente aos 4 casos de uso analisados).

---

## 1. Objetivos Principais

1. **Automatizar Model Checking (Caso 1)**: Viabilizar a verificação automática de Modelos de Redes Neurais exportados de PyTorch (ex: conversão AST/ONNX para C) sem depender de geração puramente manual com precisão suportada pelos solvers SMT.
2. **Escalar Inferência Segura (Caso 2)**: Integrar o fluxo do ESBMC no processo de CI/CD para detecção proativa de Out-of-Bounds e Memory Leaks em matrizes dinâmicas.
3. **Robustez de Agentes Neuro-Simbólicos (Caso 3)**: Formalizar a arquitetura loop *LLM Generate -> Compile -> Verify -> Feedback*, lidando com falsos positivos de forma inteligente a tempo de execução.
4. **Resiliência em Controle Híbrido (Caso 4)**: Criar templates plug-and-play do ESBMC que simulem injeção de falhas aleatórias (Caos) em plantas industriais e de IoT.

---

## 2. Como Alcançar (Execução Técnica)

### Fase 1: Padronização e Refatoração (0 - 15 dias)
- Centralizar o núcleo (core) do ESBMC local vs distribuído.
- Migrar scripts soltos em Python integrando-os a uma suíte em `pytest` focada unicamente na orquestração do verificador.
- **Técnica**: Usar a Python Subprocess API mas encapsulando os timeouts e parses de `stdout` e `stderr` (para evitar bugs de escape ANSI e travamentos em chamadas binárias já observados).

### Fase 2: Conversor ONNX/Torch para C (15 - 30 dias)
- Criar um parser mínimo para redes *forward-feed* (Camadas Densas + ReLU).
- **Técnica**: Não usar AST do Python iterativo. Exportar os pesados para JSON/ONNX standard, e usar scripts em Python (`parser_onnx.py`) para gerar código C `verify_neural_net.c`.

### Fase 3: Instrumentação em CI/CD (30 - 45 dias)
- Criar *Actions* ou *Pipelines* que isolem o ambiente *runner* C++.
- **Técnica**: Containerizar o pacote x86-64 do build atual do ESBMC juntamente com os solver (ex: Z3, Bitwuzla) para rápida execução unitária (unit-verification).

---

## 3. Estrutura de Diretórios Recomendada

A organização atual `pibic/` demonstra bem as verticais, mas as dependências devem se tornar produtos testáveis.

```text
pibic/
├── core_verify/                     # Core wrapper lib (Python) para ESBMC
│   ├── esbmc_caller.py              # Subprocesses confiáveis e parsing
│   └── SMT_feedback_parser.py       # Extração em JSON dos contra-exemplos
├── cases/                           # (Antigos numerados 1 a 4)
│   ├── ai_model_checking/           # Caso 1 (NN Verificator)
│   │   ├── converters/              # Scripts ONNX/Torch -> C
│   │   └── templates/               # C harness stubs
│   ├── inference_safety/            # Caso 2 (C++/CUDA GEMM testing)
│   ├── agentic_verification/        # Caso 3 (Neuro-simbólico self-healing)
│   └── control_chaos_testing/       # Caso 4 (PID, Sensores nondet)
├── tests/                           # Bateria TDD
│   ├── test_ai_models.py
│   ├── test_inference.py
│   ├── test_agents.py
│   └── test_chaos.py
├── scripts/                         # Automação de Infra (Setup, Build Binaries)
└── results/                         # Gráficos, Benchmarks CSV, Relatórios MD
```

---

## 4. Estratégia de Testes e Validações

O teste aqui é **Testardo o Verificador**. Temos que validar que o pipeline de Model Checking funciona como esperado através de instâncias e falsos controlados:

### 4.1 Validação Qualitativa (TDD com Propriedades Formais)
- **TDD Inverso**: Criaremos "testes com erro conhecido" (ground-truth vulnerabilities) e exigiremos que os scripts consigam extrair o bug exato detectado pelo ESBMC.
  - *Ex*: Redes neurais que sabemos explodir o gradiente; matrizes com offset intencional de `+1`; loops de agens com corrupção de memória proposital.
- **Integração Pytest**: As asserções dos testes (ex: `assert run_esbmc(file).status == UNSAFE`) devem refletir se a verificação formal falhou quando **deveria** falhar e obteve sucesso quando **deveria** ter sucesso.

### 4.2 Validação Quantitativa (Benchmarking Contínuo)
- **Time Limits Enforced**: Definir teto global de `unwind` e um cut-off time (ex: 20 segundos para *unit-checks*, 2h para nightly builds). 
- **Métricas Chave**:
  - Propriedades Geradas (VCCs)
  - Tempo gasto na Simulação Simbólica e Geração (Symex)
  - Tempo de Resolução do SMT Solver.
- Esses KPIs devem ser extraídos periodicamente para o arquivo `.csv` que geramos no relatório inicial e integrados nos PRs usando um bot de status.

---

## 5. Bloqueios Conhecidos e Prevenções

1. **Aritmética Float Ocupando Solvers (`--floatbv`)**: 
   - *Ação*: Manter as camadas de rede pequenas para `--floatbv` (verificação de flutuação exata). Em loops de deep learning, preferir Fixed-Point Analysis.
2. **Crash Silencioso de Subprocess**:
   - *Ação*: Implementar sistema de timeouts assíncronos que coletam SIGINT e fecham child processes zumbis deixados pelo ESBMC.
3. **Escopo de Memória Multi-nível**:
   - *Ação*: Nos testes limitamos arrays grandes por Macros C `-DDIM_LIMIT=` forçadas por scripts externos. Requer disciplina estrita no design C++ no Caso 2.
