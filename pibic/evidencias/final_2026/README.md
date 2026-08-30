# Rodada canônica de evidências — relatório final 2026

Esta pasta é uma área isolada para evidências do relatório final. Os JSONs de
entrada em `cartpole/` foram preservados; a rodada canônica usa cópias e
implementações de auditoria. Posteriormente, os dois scripts de auditoria
estrutural foram corrigidos em `cartpole/`, com as versões anteriores mantidas
em `legacy/`.

## O que foi executado

```text
python evidencias/final_2026/scripts/canonical_quantization.py
python evidencias/final_2026/scripts/canonical_concrete_checks.py
python evidencias/final_2026/scripts/build_manifest.py
```

`canonical_quantization.py` reexecuta a análise sobre os JSONs Float32 e Q8.8
exportados, com `numpy.random.RandomState(42)` e 10.000 amostras. O resultado
reproduz o relatório histórico até erro de arredondamento numérico.

`canonical_concrete_checks.py` gera harnesses C com os dois estados publicados
como contraexemplos e os submete ao ESBMC 6.8.0. Um resultado `SUCCESSFUL`
nesta pasta significa que a asserção para aquele estado fixo foi verificada; não
significa prova universal da propriedade.

`canonical_short_universal.py` tentou C (limite de força) e A-direita universal
com Boolector e Z3. O `esbmc.exe` Windows disponível não contém Boolector;
essas tentativas ficam como `UNKNOWN`. Com Z3, C terminou `SUCCESSFUL` em 3,5 s
e A-direita atingiu `TIMEOUT` em 12 s.

## Resultados da primeira rodada

- Quantização Q8.8: erro máximo `7,7589197857363 N`; média `0,0778007758573 N`;
  p95 `0,2262005212121 N`; p99 `1,0927201961733 N`.
- O maior erro da amostra é uma inversão de comportamento: Float32 `+6,4308 N`
  contra Q8.8 `-1,3281 N`, em um estado próximo de uma fronteira de ativação.
- A varredura da `tanh` em `outputs/tanh_approximation.json` encontrou erro
  absoluto máximo de `4,5749%`, não inferior a 3% como afirma o texto atual.
- A-direita: o estado publicado realmente produz `z < 0` no controlador Q8.8.
- B-segurança: o estado publicado realmente produz `theta_new < -53` em Q8.8,
  equivalente a aproximadamente `-12,09°` no modelo linearizado.

## Bloqueios ainda abertos

1. Validar diretamente o checkpoint PyTorch e confirmar que ele é a origem dos
   dois JSONs exportados.
2. Refazer a análise de neurônios com bounds corretos e com a camada 2 ligada
   concretamente à camada 1.
3. Refazer as propriedades universais de malha fechada, preservando harnesses,
   stdout/stderr e tempo real de cada execução.
4. Reconciliar a divergência entre `texto_apresentacao_pibic.md` (TIMEOUT nas
   duas direções) e `ddpg_closed_loop_results.json` (FAILED nas duas).

O arquivo `MANIFESTO.json` contém hashes SHA-256, ambiente, comandos, arquivos
de entrada e resultados gerados.
