# Análise estrutural do relatório atual

> **Correção de identificação (30/08/2026):** o programa correto é **PIBIC**
> (PROPESP/UFAM), não PIBITI/PROTEC. O "formulário oficial atual" citado
> abaixo (DOCX da PROTEC) foi substituído, para fins de estrutura, pelo guia
> `modelos/UFAM_guia_relatorio_PIBIC_exatas.pdf`; ver
> `PENDENCIAS_FINAIS_AUDITORIA.md`.

Escopo analisado: `pibic/artigo/`, arquivos de resultados em `pibic/results/` e materiais mais maduros existentes no projeto, especialmente `pibic/cartpole/`.

## Diagnóstico executivo

O projeto contém uma contribuição técnica forte e material suficiente para um relatório competitivo, mas o documento atual ainda não está em condição de submissão. O maior risco não é visual: é a falta de correspondência entre algumas afirmações quantitativas e os artefatos experimentais existentes. Antes de polir a escrita, é necessário transformar o relatório em uma narrativa única, rastreável e reproduzível.

### Bloqueadores de submissão

1. **O PDF atual está inválido.** `pibic/artigo/relatorio.pdf` não possui tabela `xref`/trailer válida. Uma compilação limpa também para em `caps/3metodologia.tex`, no nó TikZ que usa a chave desconhecida `/tikz/aspect`.
2. **O documento ainda se apresenta como parcial.** `caps/4resultados.tex` usa “Resultado Parcial” e “resultados parciais”; a conclusão repete essa formulação.
3. **O template não corresponde ao formulário vigente.** O LaTeX deriva de um modelo de TCC da UEPB/CCHE, conserva campos de banca e elementos alheios ao PIBITI. O formulário oficial atual é o DOCX da PROTEC.
4. **Há alegações sem evidência correspondente ou conflitante:**
   - `results/case1_mlp.log` registra falha por ausência do módulo `ast2json`, enquanto o texto declara `VERIFICATION SUCCESSFUL`.
   - `results/case2_benchmark.csv` contém apenas N=2 (4,5900 s) e N=3 (53,2923 s), mas a tabela do relatório apresenta N=2 a 6 com tempos inferiores a 15 s.
   - `results/case3_agent_stats.csv` está vazio, embora o relatório descreva cinco iterações e tempos sub-segundo.
   - As contagens “52 topologias, 34 seguras, 15 vulneráveis e 3 timeouts” não têm um CSV/log consolidado no diretório de resultados. Gráficos associados são gerados por vetores codificados manualmente ou por dados simulados.

Nenhum desses pontos deve ser “maquiado”. O caminho para um relatório premiável é repetir os experimentos ou reescrever as afirmações com o alcance real dos dados existentes.

## Inconsistências de conteúdo e forma

- O resumo fala em quatro estudos de caso; Metodologia e Resultados apresentam seis.
- Os objetivos específicos pulam o Nível 2 e não se alinham claramente aos seis casos.
- O título combina GenAI, kernels, agentes e controladores neurais, mas o relatório não explicita uma pergunta central que una todas essas frentes.
- A bibliografia tem 18 entradas, porém o corpo cita essencialmente duas; várias referências são legadas de futebol, impedimento e YOLO.
- Permanecem campos de examinadores “CCHE/UEPB”, estrutura de TCC e arquivos antigos sem relação com o PIBITI.
- O texto contém formulações promocionais ou absolutas — “rigor absoluto”, “brutalmente”, “esmagadora”, “isola brilhantemente”, “base axiomática impenetrável” — que reduzem a credibilidade científica.
- Figuras identificam fontes como “Subagente”, o que não é uma atribuição científica adequada.
- Algumas conclusões extrapolam o que BMC demonstra. Toda garantia deve declarar explicitamente o domínio de entrada, o limite de desenrolamento, a propriedade, a versão do solver e as abstrações adotadas.

## O núcleo mais forte do projeto

O material em `pibic/cartpole/texto_apresentacao_pibic.md` possui uma narrativa experimental muito mais coerente que o relatório atual:

- controlador DDPG para Cart-Pole, arquitetura 4-24-24-1;
- quantização Q8.8 com equivalência aritmética documentada;
- 48 neurônios avaliados;
- propriedade de limite de força provada;
- propriedade de segurança em um passo refutada com contraexemplo reproduzível;
- propriedade de correção direcional com timeout, apresentada como limitação;
- comparação Float32 versus Q8.8 sobre 10.000 estados;
- aplicação web para reproduzir simulações e injetar o contraexemplo.

Essa combinação de resultado positivo, contraexemplo, limitação e demonstração reproduzível tem melhor perfil científico do que seis casos pouco conectados. A recomendação é usá-la como estudo principal, desde que o título e os objetivos do projeto oficialmente aprovado permitam. Os demais casos podem aparecer como validações complementares da metodologia, não como seis contribuições de igual peso.

## Estrutura recomendada no modelo oficial

1. **Identificação:** edição 2025/2026, orientador, aluno, modalidade, título e código exatamente como constam no projeto aprovado; marcar grande área, potencial de software/patente e eventual sigilo.
2. **Resumo:** problema, lacuna, método, objeto experimental, três resultados quantitativos verificáveis, principal limitação e contribuição; incluir ao menos três palavras-chave.
3. **Introdução:** motivação, problema específico, lacuna na literatura, pergunta de pesquisa e contribuições. Evitar uma revisão enciclopédica de IA.
4. **Objetivos:** um objetivo geral e objetivos específicos mensuráveis, cada um associado a um resultado.
5. **Metodologia:** perguntas de pesquisa, modelo/controlador, quantização, propriedades formais, domínio das entradas, hardware/software, versões, comandos, limites, critérios de sucesso, repetições e ameaças à validade.
6. **Resultados e discussão:** organizar por pergunta de pesquisa, não por pasta do repositório. Para cada afirmação, mostrar tabela/figura derivada de dados, interpretar, comparar com literatura e declarar limites.
7. **Conclusão:** responder aos objetivos, separar contribuição técnica de limitações e indicar próximos passos concretos.
8. **Referências:** remover itens legados e incluir literatura central sobre ESBMC/BMC, verificação de redes neurais, QNN, controladores neurais e quantização.
9. **Anexo oficial:** manter o termo de autorização do modelo PROTEC, preenchido conforme a situação do depósito.

## Matriz mínima de rastreabilidade

Cada resultado do texto deve possuir estes cinco elementos:

| Campo | Exemplo esperado |
|---|---|
| Propriedade | `-10 N <= força <= 10 N` |
| Domínio/limite | intervalos das quatro variáveis de estado e horizonte verificado |
| Comando/versão | comando ESBMC, versão, solver e flags |
| Artefato bruto | log, CSV ou JSON gerado automaticamente |
| Figura/tabela | script que lê apenas o artefato bruto, sem números codificados à mão |

## Ordem de trabalho recomendada

1. Confirmar no eCampus o vínculo PIBITI, o código e o título oficial.
2. Congelar a pergunta central e escolher o estudo principal.
3. Reexecutar os experimentos indispensáveis e gerar um manifesto de resultados.
4. Remover ou rebaixar toda afirmação sem evidência rastreável.
5. Migrar o conteúdo para a estrutura do modelo PROTEC, corrigir bibliografia e tom científico.
6. Compilar, revisar visualmente página a página e produzir PDF válido com menos de 2 MB.
7. Enviar com antecedência pelo perfil do orientador/técnico no eCampus.

## Critério de “digno de prêmio”

O diferencial não será a quantidade de casos, mas uma contribuição nítida, tecnicamente correta e reproduzível: problema relevante, método justificável, evidência quantitativa, contraexemplo interpretado, limitações honestas e impacto demonstrável. O projeto já tem esses ingredientes; a revisão deve concentrá-los em uma história científica única.

