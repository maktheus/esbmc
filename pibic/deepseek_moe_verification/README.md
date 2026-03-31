# DeepSeek MoE Router - Formal Verification (ESBMC)

Este diretório contém a nova fase do projeto acadêmico (PIBIC) focado em **Verificação Formal de Large Language Models**. Diferente da abordagem global que gera explosão de estados, este ambiente foi construído para isolar cirurgicamente a camada de **Roteamento de Especialistas (Mixture of Experts - MoE)** de modelos de classe Mundial como o DeepSeek.

## Motivação
Em LLMs baseados em MoE (como Mixtral e DeepSeek), o Roteador é responsável por balancear a carga multibilionária definindo quais "Placas de Vídeo / Especialistas" processarão o token matematicamente. Se um ruído (*Adversarial Perturbation*) causar um erro no Roteador, 100% da carga pode ser despejada em um único Especialista (*Expert Collapse*), colapsando o cluster.

## Estrutura do Projeto (A Ser Implementada)
- `scripts/`: Scripts em Python para puxar os Safetensors (.bin) da DeepSeek originais e exportar o _Gating Node_ em formato C (`.h`).
- `include/`: Cabeçalhos de memória (`router_weights.h`).
- `src/`: A lógica espelhada do `llama.cpp` que faz o cálculo `Top-K` das rotas.
- `verify.c`: A Harness oficial integrando as variáveis `nondet` do ESBMC para validar os Asserts Formais de segurança (Robutez do Roteador).

---
*Projeto em desenvolvimento ativo. Guiado pelo padrão ouro da prova matemática estática C/C++.*
