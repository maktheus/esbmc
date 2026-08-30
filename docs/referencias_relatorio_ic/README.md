# Referências para o Relatório Final PIBIC 2025/2026

Coleta realizada em 30 de agosto de 2026 para apoiar a preparação do relatório final do projeto de Engenharia da Computação.

**Correção de identificação (30/08/2026):** o projeto é do **PIBIC**
(Programa Institucional de Bolsas de Iniciação Científica, administrado pela
PROPESP/UFAM), não do PIBITI (administrado pela PROTEC/UFAM). Os arquivos
`UFAM_modelo_relatorio_final_PIBITI_PROTEC.docx` e
`UFAM_edital_PIBITI_2025_2026.pdf` abaixo foram mantidos apenas como registro
histórico da pesquisa anterior; eles **não** devem ser usados como fonte de
estrutura, prazo ou identificação para este relatório. Ver
`PENDENCIAS_FINAIS_AUDITORIA.md` para o detalhamento da correção.

## Prazo e submissão

- O Edital 05/2025-PROPESP/UFAM (PIBIC 2025-2026) lista, em seu cronograma, a
  submissão do Relatório Técnico Final no e-Campus — junto com o Termo RIU —
  no período de **1º a 30 de agosto de 2026**.
- As fontes oficiais consultadas **não publicam um horário-limite** para o
  dia 30. Portanto, não há base documental para assumir 23h59. **O prazo
  termina hoje.**
- Fonte principal (edital): <https://edoc.ufam.edu.br/bitstream/123456789/9616/1/EDITAL_PIBIC%202025-2026.pdf>
- Página de editais PIBIC/PAIC: <https://propesp.ufam.edu.br/ultimas-noticias/486-pibic-paic-2025-2026.html>

## Modelos e normas baixados

| Arquivo | Uso recomendado | Observação |
|---|---|---|
| `modelos/UFAM_guia_relatorio_PIBIC_exatas.pdf` | Guia oficial da estrutura do relatório | Guia do Comitê Local de Ciências Exatas, da Terra e Engenharias do PIBIC. Deve prevalecer sobre os demais modelos: define Capa, Folha de rosto, Resumo (≤500 palavras), Sumário, Introdução (≤1 pág.), **Revisão bibliográfica (≤2 pág.)**, Métodos utilizados (≤2 pág.), Resultados e discussões (≤5 pág.), Conclusões (≤1 pág.), Agradecimentos, Referências e Cronograma. |
| `modelos/UFAM_edital_PIBIC_2025_2026.pdf` | Cronograma e regras do ciclo | Edital 05/2025-PROPESP/UFAM. Confirma o período final de 01 a 30/08/2026, sem indicar hora, e o depósito do Termo RIU junto ao Relatório Final (item 11.4). |
| `modelos/Overleaf_modelo_relatorio_PIBIC_UFAM.pdf` | Referência visual e de organização em LaTeX | Modelo público de 2021, feito para PIBIC/PAIC. Página: <https://pt.overleaf.com/latex/templates/modelo-de-relatorio-de-pibic-ufam/xgntxbfshpgb>. |
| `modelos/overleaf_fonte/main.tex` | Fonte principal exposto pelo modelo público | Foi recuperado da página do Overleaf. Depende de includes e imagens que não são expostos separadamente; por isso, não é um projeto autônomo completo. Leia `modelos/overleaf_fonte/README.md`. |
| `modelos/UFAM_modelo_relatorio_final_PIBITI_PROTEC.docx` | Histórico — não usar | Modelo do PROTEC para o PIBITI, programa distinto do PIBIC. Mantido apenas para registro da pesquisa anterior. |
| `modelos/UFAM_edital_PIBITI_2025_2026.pdf` | Histórico — não usar | Edital do PIBITI (PROTEC), programa distinto do PIBIC. Mantido apenas para registro da pesquisa anterior. |

## Documentação produzida

- `ANALISE_ESTRUTURAL_RELATORIO.md`: auditoria da estrutura LaTeX atual, inconsistências e riscos de evidência.
- `ANALISE_DOS_PREMIADOS.md`: comparação documentada dos premiados, separando observação de inferência.
- `PLANO_RELATORIO_FINAL_PREMIAVEL.md`: narrativa, perguntas, seções, matriz afirmação–evidência e checklist de submissão.
- `PLANO_DE_EXECUCAO_DETALHADO.md`: sequência operacional, arquivos, reexecução dos experimentos, critérios de aceite, cronograma e divisão de responsabilidades.
- `RASCUNHO_NARRATIVA_RELATORIO.md`: texto-base de introdução, objetivos, contribuições, discussão e conclusão provisória, com marcações de validação.
- `PENDENCIAS_FINAIS_AUDITORIA.md`: checklist final, distinguindo o que já foi confirmado do que ainda depende do eCampus, do orientador ou de assinatura.

## Artefatos executados nesta rodada

- `../../pibic/relatorio_final_2026/`: novo esqueleto LaTeX isolado, alinhado à PROTEC, com PDF de teste compilado.
- `../../pibic/evidencias/final_2026/`: rodada canônica de auditoria, manifesto, logs, harnesses e resultados de quantização/reprodução. A rodada inicial preservou os scripts; depois, os dois scripts estruturais foram corrigidos, com versões legadas preservadas.

Resultado importante da rodada: a quantização foi reproduzida, mas o pior estado amostrado inverte a ação entre Float32 (`+6,4308 N`) e Q8.8 (`-1,3281 N`). A propriedade de limite de força foi provada com Z3 em 3,5 s. Os harnesses estruturais foram corrigidos e seus resultados permanecem explicitamente parciais nos casos que terminaram em `TIMEOUT`.

## Premiações oficiais — Engenharias, 2021 a 2025

Os PDFs em `premiacoes_oficiais/` são as listas oficiais do CONIC. “Engenharias” é a grande área mais próxima do projeto; quando havia premiação específica do PIBITI, ela foi separada abaixo.

| Ano | Premiado em Engenharias (Manaus) | Orientador | Trabalho |
|---|---|---|---|
| 2021 | Robert Batista Neves | Gustavo Cunha da Silva Neto | Análise da dinâmica de vibrações com amortecimento não-linear por meio do método de Krylov-Bogoliubov |
| 2022 | Raquel de Sousa Freire | Virginia Mansanares Giacon | Produção de compósitos poliméricos reforçados com fibras de juta e malva pós tratamento de hornificação |
| 2023 | Rafael Pereira Bezerra | Nilton Pereira da Silva | Estimativa de parâmetros em problema de biotransferência de calor |
| 2024 | Levi dos Santos Carneiro | Antonio do Nascimento Silva Alves | Estudo da influência dos parâmetros de impressão 3D nas propriedades mecânicas de material compósito polimérico reforçado com fibra de carbono através do projeto de experimentos Box-Behnken |
| 2025 | Robson Marchegiani Seixas Nogueira | Luiz Eduardo Sales e Silva | Desenvolvimento de um sistema de detecção de riscos em redes elétricas da UFAM através de visão computacional com foco em árvores próximas aos condutores de distribuição de energia elétrica |

Em 2025, Amanda Nicole Silveira Spellen, orientada por Altigran Soares Silva, foi premiada em Ciências da Computação pelo trabalho “Um estudo sobre tecnologias seguras para chatbots de análise de dados acadêmicos”. Marcos Paulo Batista Canto, orientado por Iury Valente de Bessa, recebeu menção honrosa em apresentação oral no Comitê de Engenharias.

## Premiações específicas do PIBITI

- `premiacoes_pibiti/PIBITI_2023_classificacao_posters.pdf`: classificação oficial das apresentações de projetos PIBITI 2022/2023.
- `premiacoes_pibiti/PIBITI_2024_melhores_projetos_e_mencoes.pdf`: melhores projetos e menções do PIBITI 2023/2024. O resultado mais próximo deste projeto é a menção honrosa de **Davillon Maclaus Cruz Camillo**, orientado por **Iury Valente de Bessa**, em Engenharia da Computação, pelo trabalho “Desenvolvimento de uma plataforma de testes de veículos autônomos subaquáticos”.
- `premiacoes_oficiais/CONIC_2025_XXXIV_premiacao.pdf`: em sua seção PIBITI, lista três melhores projetos: Lana Cavalcante Ramos, Camyla Tamyres Ribeiro Lotas e Lívia Evelin de Souza Auzier.
- `premiacoes_pibiti/PIBITI_2024_2025_projetos_selecionados.pdf`: permite recuperar os títulos desses três projetos: membranas eletrofiadas para conservação de alimentos; pirólise de óleo de fritura residual; e rede neural para identificação de fluidos em rochas-reservatório.

## Limitação da coleta

O repositório público da UFAM disponibiliza de forma consistente as listas de premiação, mas não os relatórios finais completos de todos os vencedores. Foram localizados e guardados o trabalho CONIC integral de 2025 e um artigo integral derivado da pesquisa premiada em 2021, na pasta `trabalhos_premiados/`. O trabalho relacionado de 2024 foi analisado em páginas públicas, mas seu PDF não foi redistribuído por exigir autenticação/autorização do provedor. Para 2022 e 2023, a análise limita-se ao título oficial e a publicações/descrições públicas, com inferências explicitamente identificadas.
