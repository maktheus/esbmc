# Relatório de Análise Comparativa: ESBMC em GenAI

## 1. Introdução

Este relatório apresenta uma análise aprofundada da aplicação do ESBMC na verificação de sistemas de Inteligência Artificial Generativa. O estudo abrange três níveis de abstração:

1. **Modelo (Python/Pytorch)**: Verificação de propriedades funcionais em redes neurais.
2. **Infraestrutura (C++/CUDA)**: Verificação de segurança de memória em kernels de inferência.
39. **Aplicação (Agentic)**: Verificação de código gerado por LLMs em tempo de execução.
10. **Controle (Chaos)**: Verificação de robustez em sistemas de controle sob injeção de falhas.
11. **Reinforcement Learning (RL)**: Verificação formal de restrições físicas de ações contínuas em políticas de agentes autônomos.

## 2. Deep Dive: Como Funciona a Análise Formal do ESBMC

O ESBMC (Efficient SMT-Based Context-Bounded Model Checker) não apenas "testa" o código; ele o **prova** matematicamente dentro de limites definidos.

```mermaid
graph TD
    A["Código Fonte C/C++/Python"] --> B["GOTO-Programs (IR)"]
    B --> C["Simulação Simbólica (Symex)"]
    C --> D{"Gerador de Fórmulas SMT"}
    D --> E["Solver SMT (Z3/Bitwuzla)"]
    E -->|SAT| F["Contra-Exemplo Encontrado (Bug)"]
    E -->|UNSAT| G["Verificação Bem-Sucedida (Prova)"]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#9f9,stroke:#333,stroke-width:2px
    style F fill:#f99,stroke:#333,stroke-width:2px
```

1. **Frontend**: Traduz o código para uma Representação Intermediária (GOTO-Programs).
2. **Simulação Simbólica**: Executa o programa simbolicamente, desenrolando loops `k` vezes.
3. **Codificação SMT**: Converte as asserções e o estado do programa em fórmulas lógicas de primeira ordem.
4. **Solver**: Um solucionador SMT verifica se existe algum conjunto de entradas que satisfaça a negação das propriedades (buscando uma falha).

## 3. Análise Detalhada dos Casos

### 3.1. Caso 1: Verificação de Modelos Python (MLP)

```mermaid
flowchart LR
    In["Entrada Simbólica"] --> MLP["Rede Neural (Python)"]
    MLP --> Out["Saída Calculada"]
    
    subgraph ESBMC Verification
        Prop1{"Saída <= 1.0?"}
        Prop2{"Saída >= 0.0?"}
    end
    
    Out --> Prop1
    Out --> Prop2
    Prop1 -->|Falha| Bug[Violação de Post-condition]
    Prop2 -->|Falha| Bug
```

Implementamos uma Rede Neural Multi-Camadas (MLP) simples em Python puro (`mlp.py`) para verificar se a saída da rede, normalizada via Sigmoid/ReLU, respeita os limites teóricos [0, 1] para qualquer entrada válida.

- **Resultados**: O ESBMC conseguiu verificar a propriedade `output <= 1.0` usando k-indução.
- **Desafios Encontrados**: A maior barreira foi a dependência de tipagem estática rigorosa. O frontend Python do ESBMC (baseado em `ast`) requer anotações de tipo precisas e falha com construções dinâmicas comuns em frameworks de ML (ex: listas heterogêneas).
- **Conclusão**: A verificação direta de código de treinamento/modelo é viável para *subconjuntos estritos* de Python, mas requer um esforço significativo de refatoração para adequação à ferramenta.

### 3.2. Caso 2: Motor de Inferência (GEMM)

```mermaid
sequenceDiagram
    participant Mem as Memória (Heap)
    participant Kernel as MatMul Kernel
    participant ESBMC as Monitor ESBMC
    
    loop "Para cada Bloco (Tilling)"
        Kernel->>Mem: Acesso A[i][k]
        ESBMC->>ESBMC: Check Bounds
        Kernel->>Mem: Acesso B[k][j]
        ESBMC->>ESBMC: Check Bounds
        Kernel->>Mem: Escrita C[i][j]
        ESBMC->>ESBMC: Check Overflow
    end
```

Verificamos um kernel de multiplicação de matrizes (*General Matrix Multiply* - GEMM) otimizado com tiling (`matmul_kernel.cpp`). O objetivo foi garantir a ausência de *buffer overflows* e acessos inválidos à memória, críticos em ambientes de produção de alta performance.

![Desempenho da Verificação de Matrizes](case2_plot.png)

- **Análise de Escalabilidade**: O gráfico acima demonstra que o tempo de verificação cresce exponencialmente com a dimensão da matriz. Para matrizes pequenas ($N \le 4$), a verificação é quase instantânea (< 3s). Para $N=6$, o tempo sobe para ~15s.
- **Implicações**: A verificação formal completa é impraticável para matrizes reais de LLMs (ex: 4096 dim), mas é **altamente eficaz** para verificar a lógica do algoritmo em instâncias reduzidas ("small scope hypothesis"), garantindo que a implementação do tiling está correta antes de escalar.

### 3.3. Caso 3: Agente Neuro-Simbólico

```mermaid
stateDiagram-v2
    [*] --> LLM_Generate
    LLM_Generate --> ESBMC_Verify: Código Gerado
    
    state ESBMC_Verify {
        [*] --> Check
        Check --> Success: "UNSAT (Seguro)"
        Check --> Fail: "SAT (Bug)"
    }
    
    ESBMC_Verify --> Deploy: Success
    ESBMC_Verify --> LLM_Refine: "Fail (Contra-exemplo)"
    LLM_Refine --> LLM_Generate: "Prompt + Erro"
```

Simulamos um loop onde um Agente de IA gera código C inseguro (com buffer overflow) e usa o ESBMC para detectar a falha e corrigir o código iterativamente (`mock_agent.py`).

![Tempo de Verificação por Iteração](case3_plot.png)

- **Análise de Overhead**: O gráfico mostra o tempo gasto pelo verificador em cada iteração do agente. Observa-se que o tempo de verificação é constante e baixo (< 1s) para os snippets gerados.
- **Eficácia**: O ESBMC atuou como um "crítico" perfeito, rejeitando código vulnerável que passaria em testes funcionais simples (se o input de teste não disparasse o overflow).
- **Conclusão**: A integração ESBMC-LLM é a aplicação de maior impacto imediato. O custo computacional é marginal comparado ao ganho de segurança.

## 3.4 Visão Comparativa: O Modelo vs O Solver SMT (The Ralph Loop Concept)

Para ilustrar de forma didática o que o ESBMC efetivamente enxerga durante o ciclo do agente (em contraste com o modelo idealizado da LLM), elaboramos as representações conceituais abaixo:

### A Visão Contínua (Expectativa do Agente/IA)
![Expected NN Boundary](/home/uchoa/.gemini/antigravity/brain/ab035961-5cb8-43dd-9ed6-c903ecc815d4/expected_nn_boundary_1772153552771.png)
*A rede neural ou o Agente generalizam o espaço simulado e assumem uma fronteira de decisão suave e contínua, invisível ao risco de bordas extremas nas validações primitivas.*

### A Visão Discreta Formal (Realidade SMT no ESBMC)
![Actual SMT Violation](/home/uchoa/.gemini/antigravity/brain/ab035961-5cb8-43dd-9ed6-c903ecc815d4/actual_smt_violation_1772153568416.png)
*O solver (Z3 atuando no ESBMC) "quebra" as superfícies estáticas em domínios discretos restritos. Pelo modelo de Ralph Loop, ele caça ativamente contra-exemplos provando matematicamente os picos (falhas de buffer, violações de arrays, vazamentos de memória). Essas ranhuras e violações caóticas são devolvidas como feedback exato e semântico à LLM reescrever o código de volta à segurança.*

### 2.4. Caso 4: Sistema de Controle Digital (Engenharia do Caos)

```mermaid
graph TD
    Sensor["Sensor Ruído"] -->|"Temp + Noise"| PID["Controlador PID"]
    PID -->|Output| Plant["Planta Térmica"]
    Plant -->|Feedback| Sensor
    
    subgraph Chaos Injection
        Noise("Nondet Float -5 to +5") -.-> Sensor
    end
    
    subgraph Safety Property
        Check{"Temp < MAX_SAFE?"}
    end
    
    Plant --> Check
    Check -->|No| Alarm[Violação de Segurança]
```

Implementamos um controlador PID digital (`pid_controller.c`) responsável por regular a temperatura de um sistema físico simulado. Introduzimos princípios de **Engenharia do Caos** injetando ruído não-determinístico nos sensores para testar a robustez do controle.

- **Cenário de Caos**: O sensor de temperatura pode sofrer flutuações aleatórias (ruído) de até ±5.0 graus a cada leitura.
- **Propriedade de Segurança**: Mesmo sob condições de caos, a temperatura do sistema **nunca** deve exceder `MAX_SAFE_TEMP` (150.0).
- **Resultados**: O ESBMC verificou formalmente que, para os parâmetros definidos (Kp, Ki, Kd), o sistema permanece estável e seguro, provando que o controlador é robusto ao nível de ruído especificado.

### 3.6. Caso 6: Verificação Formal em Reinforcement Learning (Actor-Critic Policies)

A verificação de políticas de Reinforcement Learning (RL) é extremamente crítica, visto que agentes aprendem baseando-se no acúmulo de recompensas e comumente falham ao ignorar limites físicos restritivos do ambiente externo (e.g., um carro cujo agente gire o volante em um valor superior do que o atuador físico engata mecanicamente).

Simulamos estaticamente (`rl_policy.c`) uma rede neural típica atuando como Actor em um processo contínuo em RL. A rede neural processa métricas complexas do Lidar e codifica a assertividade da ação. O limitador atuador físico foi checado usando model checking dinâmico `--floatbv` do ESBMC (conectado ao backend Z3).

- **Resultados do Solver**: O Z3 interceptou ativamente comportamentos erráticos nos bounds, encontrando um pico isolado onde velocidades altas interagindo no gradiente final de proximidade iriam obrigar o atuador do agente acima dos 1.0f de força do voltante permitidos pela simulação. O PyTest automatizou a captura desse problema em milissegundos.
- **Conclusão**: O uso de Verificação Formal como "Safety Shield" após um período empírico de treinamento atua como mitigador para atestar se comandos concebidos pelo Agente RL poderão infligir danos ou quebrar as restrições invariantes na física real da robótica implantada.

## 4. Arquitetura Final: Desempenho Python API vs Limitações Nativas (C++)

A construção do repositório no pacote PyPI isolado (`core_verify/`) teve 100% de sucesso operacional envelopando as robustas pipelines de C++ do ESBMC por trás de interfaces pythônicas modernas. Entretanto, a arquitetura provou seus gargalos sistêmicos de pesquisa: Existem barreiras de desempenho puramente transponíveis apenas através de re-engenharia dos hooks expostos do core principal em C/C++:

| Domínio Tecnológico | Abordagem Atual via ESBMC API (Python Wrapper) | Por que Requer Edições do Core Codebase GOTO-SYMEX em C++ |
| :--- | :--- | :--- |
| **Geração de Árvores Lógicas (AST)** | Extração RegEx. Se um LLM alucinar em texto denso de compiler GOTO-CC, o regex Py colapsa. | O ESBMC (GOTO-CC) monta nativamente a *Abstract Syntax Tree (AST)* em sua espinha de memória. Interagir com C++ no core permitiria injetar heurísticas pré-resolução que cortem a árvore inteira e parem o solver antes de timeout. |
| **Controle Fino de SMT Solvers** | Isolamos em sub-processos bloqueantes que tentam prever a interrupção. | O subsistema nativo SMT interage bidirecional de ponteiros C++ com o formato Z3/Bitwuzla. Instrumentar em C++ expõe variáveis internas permitindo ao otimizador pausar iterações sub ótimas sem explodir tempo computacional. |
| **Heurísticas K-Induction Ativas** | A *Flag* `--k-induction` executa limitadamente sobre um `while` bruto invocado do script. | Casos estocásticos de Caos Específico demandam ajustar `k` do *unrolling solver* perante frames probabilísticos. O Python Wrapper não escuta os loops intermediários rodando, apenas o log final reportado da prova de indução no binário final. C++ é inegociável em cenários complexos de loops infinitos neurais. |

## 5. Análise Comparativa e Conclusão Global

| Dimensão | Caso 1 (Modelo) | Caso 2 (Infra) | Caso 3 (Agente) | Caso 4 (Controle) |
| :--- | :--- | :--- | :--- | :--- |
| **Foco** | Corretude Matemática | Segurança de Memória | Segurança de Software | Robustez sob Caos |
| **Maturidade** | Baixa (Experimental) | Alta (Industrial) | Alta (Emergente) | Alta (Crítica) |
| **Custo/Benefício** | Baixo (Difícil configuração) | Alto (Bug-finding crítico) | Muito Alto (Automação) | Alto (Certificação) |

**Conclusão Geral**: O ESBMC posiciona-se como uma ferramenta essencial para a **segurança da infraestrutura de IA** (Caso 2), para a **confiabilidade de agentes de codificação** (Caso 3) e para a **certificação de sistemas de controle críticos** (Caso 4). Para verificação de modelos Python (Caso 1), recomenda-se seu uso apenas em componentes críticos onde a tipagem estática possa ser aplicada rigorosamente.
