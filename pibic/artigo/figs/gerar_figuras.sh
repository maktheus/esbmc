#!/usr/bin/env bash
# gerar_figuras.sh — Regera todas as figuras do artigo.
#
# POR QUE EXISTE. Nao havia runner: cada figura era gerada a mao, e nenhum
# script lia dado do repositorio -- todos os valores eram literais embutidos no
# proprio codigo de plotagem (KB-D10). Uma figura assim nao e evidencia; e
# desenho com aparencia de medicao.
#
# Estado atual de cada script:
#
#   LEEM DADO MEDIDO
#     plot_cases3_4.py     Caso 3 <- results/case3_agent_stats.csv
#     plot_case1_weights.py       <- teste_mlp/mlp_weights.h
#     plot_rl_shield.py           <- expressao de cases/ai_model_checking/rl_policy.c
#
#   AVALIAM FUNCAO (deterministico, legitimo -- nao ha o que "medir")
#     plot_case1.py               superficie de saida do neuronio/MLP
#     plot_silu_approx.py         aproximacao de SiLU
#
#   SIMULACAO OU DIAGRAMA (ilustrativo; as legendas dizem isso)
#     plot_cases3_4.py     Caso 4  dinamica do PID
#     plot_pid_phase_portrait.py  retrato de fase (seed fixa)
#     plot_pipeline_esbmc.py      diagrama
#     plot_python_to_smt.py       diagrama
#     plot_agent_statemachine.py  diagrama
#
# Uso:  cd artigo/figs && ./gerar_figuras.sh
set -uo pipefail
cd "$(dirname "$0")"

falhas=0
for s in plot_case1.py plot_case1_weights.py plot_cases3_4.py \
         plot_rl_shield.py plot_pid_phase_portrait.py plot_silu_approx.py \
         plot_pipeline_esbmc.py plot_python_to_smt.py plot_agent_statemachine.py; do
    [ -f "$s" ] || { printf '  %-30s AUSENTE\n' "$s"; continue; }
    if out=$(python3 "$s" 2>&1); then
        printf '  %-30s ok\n' "$s"
    else
        printf '  %-30s FALHOU\n' "$s"
        printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
        falhas=$((falhas + 1))
    fi
done

echo
if [ "$falhas" -gt 0 ]; then
    echo "$falhas script(s) falharam."
    exit 1
fi
echo "Todas as figuras regeradas."
