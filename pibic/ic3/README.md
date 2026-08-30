# Pipeline IC3/PDR

Este diretório isola a tradução da malha fechada DDPG para um sistema de
transição. A rota reproduzível hoje é:

1. `gen_transition_system.py`: pesos Q8.8 → Verilog;
2. `validate_forward.py`: compara o forward Verilog e a referência Python em
   estados concretos;
3. `run_pdr.sh`: Verilog → AIGER com Yosys → PDR com ABC.

Antes de executar, use:

```sh
python3 check_dependencies.py
python3 gen_transition_system.py --bits 16 -o cl_ddpg16.v
python3 validate_forward.py -n 12
./run_pdr.sh cl_ddpg16.v 1800
```

`YOSYS`, `ABC` e `PYTHON` podem apontar para executáveis fora do `PATH`. O
pipeline retorna `0` para prova, `1` para contraexemplo, `2` para erro de
ferramenta/entrada e `3` para resultado inconclusivo. Timeout é inconclusivo,
nunca uma prova de segurança. O stdout bruto do ABC fica em `*.abc.out` e a
síntese completa em `*.yosys.log`.

O validador é um teste diferencial amostral: qualquer divergência invalida a
tradução, mas concordância em 12 (ou N) estados não constitui prova universal de
equivalência. Textos e resultados devem descrevê-lo como validação por
amostragem, não como prova bit-exata para todo o domínio.

## Escopo para a PR atual

- **KB-E03 (curva BMC): pesquisa futura.** K=8/K=16 no ator real já está fora
  do orçamento observado; completar a curva sintética requer novas execuções e
  não corrige um defeito de merge.
- **KB-E04 (limitações): requisito de merge documental.** PDR não converge
  sempre; sua vantagem de memória é estrutural, e AIGER perde relações de
  palavra. Isso deve acompanhar qualquer tabela de resultados.
- **KB-E05 (bit-width): resultado específico do modelo.** O modelo de 16 bits
  corresponde a Q8.8; números comparando 16/32 bits só devem ser publicados com
  os dois logs brutos identificados. Não é correto tratá-los como garantia geral.
- **KB-E07 (word-level): pesquisa futura.** É uma hipótese promissora, não um
  resultado da PR. Ela exige o backend `write_btor` do Yosys e um motor como
  `btormc`, Pono ou AVR. Verifique a disponibilidade com
  `python3 check_dependencies.py --require-word-level`.

Os resultados históricos e suas lacunas estão descritos em `EVIDENCIA.md`.
Nenhum arquivo ausente deve ser reconstruído por extrapolação.
