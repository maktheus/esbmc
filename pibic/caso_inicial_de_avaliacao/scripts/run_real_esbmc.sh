#!/bin/bash
cd "$(dirname "$0")/.."

echo "=========================================================="
echo " AVALIAÇÃO FORMAL DE RUNTIME EM IA GENERATIVA (ESBMC)"
echo "=========================================================="
echo "Carregando a matriz Constante Pura do Modelo Dinamicamente..."

ESBMC_BIN="/home/uchoa/esbmc/build/src/esbmc/esbmc"

# O Llama Unwind precisa ser ao menos compatível com as Layers/Loops do próprio kernel de inferência limitados
# N=4, D=4, H=2, Vocab=64
$ESBMC_BIN src/verify_real_model_esbmc.c -I src/ -DVERIFY_ESBMC --function main --unwind 150 --timeout 500s

echo "=========================================================="
