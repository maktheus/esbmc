# Relatório Final — PIBIC/UFAM 2025–2026

Documento LaTeX do relatório final de iniciação científica sobre verificação formal
de sistemas de IA com o ESBMC. É **independente** do artigo em `../artigo/`: aquele
segue o modelo abnTeX2 de monografia; este segue a estrutura de relatório final
exigida pela PROPESP (identificação, resumo, objetivos, metodologia, resultados,
limitações, conclusão, cronograma executado, referências e apêndices).

## Compilar

```bash
make            # pdflatex -> bibtex -> pdflatex -> pdflatex
make clean      # remove intermediários
make distclean  # remove também o PDF
```

Dependências (Debian/Ubuntu):

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
  texlive-latex-extra texlive-fonts-recommended texlive-lang-portuguese \
  texlive-publishers
```

`texlive-publishers` fornece `abntex2cite` e o estilo `abntex2-alf`, usados para as
citações e a lista de referências em ABNT. O documento usa a classe `article`, e não
a classe `abntex2`, para reduzir dependências; por isso **`hyperref` é carregado antes
de `abntex2cite`** — a ordem inversa quebra o mecanismo `\abntnextkey` do pacote e as
citações saem indefinidas.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `relatorio_final.tex` | Preâmbulo e montagem do documento |
| `secoes/00_capa.tex` | Folha de identificação |
| `secoes/00_resumo.tex` | Resumo e *abstract* |
| `secoes/01_introducao.tex` … `08_cronograma.tex` | Corpo do relatório |
| `apendices/a_reprodutibilidade.tex` | Como reexecutar tudo |
| `apendices/b_indice_evidencias.tex` | Veredito → comando → log bruto |
| `referencias.bib` | Apenas as entradas efetivamente citadas |
| `figuras/` | Logo; as demais figuras vêm de `../artigo/figs/` |

## Antes de entregar

Os campos administrativos que só o autor pode preencher estão marcados em vermelho
no PDF, pela macro `\pendente{...}`. Para localizá-los:

```bash
grep -rn "pendente{" secoes/ apendices/
```

São eles: modalidade da bolsa, número do processo, os dois primeiros bimestres do
cronograma e a lista de participação em eventos.

## Procedência dos números

Todo veredito citado no texto vem de log bruto versionado. O Apêndice B liga cada
afirmação ao arquivo correspondente em `../results/logs/`; a tabela consolidada está
em `../results/EVIDENCIAS.md` e as medições indutivas em `../ic3/EVIDENCIA.md`.

Uma exceção está declarada no próprio texto: a execução histórica de PDR sobre o ator
DDPG real não teve o *stdout* preservado, e é reportada como medição do invólucro, não
como resultado plenamente reproduzível.

Duas entradas de `referencias.bib` foram **corrigidas** em relação ao `.bib` do artigo:
a autoria dos dois trabalhos de QNN (`sena2021qnn` e `song2021qnnverifier`) estava
trocada lá. As demais entradas trazem veículo e ano, mas volume e páginas ainda pedem
conferência contra a fonte antes da entrega definitiva.
