#!/bin/bash
cd "$(dirname "$0")/.."
echo "Iniciando verificação do Llama2.c com ESBMC..."
echo "Configuração: Prompt Fixo (Olá/Embedding) N=8 x Pesos Livres (Nondeterminísticos) D=8, Max Unwinds=257..."

/home/uchoa/esbmc/build/src/esbmc/esbmc src/verify_real_prompt_esbmc.c -I src/ -DVERIFY_ESBMC --function main --unwind 257 --timeout 60s
