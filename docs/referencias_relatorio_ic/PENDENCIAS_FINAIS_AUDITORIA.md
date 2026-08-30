# Auditoria final de pendências — PIBIC 2025/2026

Atualização de 30/08/2026. Esta rodada corrigiu um erro de identificação do
programa presente na versão anterior desta auditoria e do relatório: o projeto
é do **PIBIC**, não do PIBITI. As duas iniciativas são administradas por
pró-reitorias e editais diferentes na UFAM, com estruturas de relatório e
prazos distintos, então essa correção não é apenas terminológica.

## Identificação administrativa confirmada nesta rodada

- Programa/edição: **PIBIC 2025/2026** (não PIBITI);
- Pró-reitoria responsável: **PROPESP** — Pró-Reitoria de Pesquisa e
  Pós-Graduação (não PROTEC, que administra o PIBITI);
- Edital de referência: **Edital 05/2025-PROPESP/UFAM**, Programa
  Institucional de Bolsas de Iniciação Científica, edição 2025-2026
  — <https://edoc.ufam.edu.br/bitstream/123456789/9616/1/EDITAL_PIBIC%202025-2026.pdf>;
- Modalidades previstas no edital: PIBIC/UFAM, PIBIC/CNPq e PAIC/FAPEAM
  (bolsistas voluntários seguem edital à parte);
- Grande área CNPq: **Engenharias**;
- Estrutura do corpo do relatório: **Guia para Elaboração de Relatórios do
  PIBIC de Exatas, da Terra e Engenharia** (Comitê Local de Ciências Exatas
  e da Terra e Engenharias do PIBIC), já presente em
  `modelos/UFAM_guia_relatorio_PIBIC_exatas.pdf`. Este guia — e não o modelo
  `.docx` do PROTEC/PIBITI usado antes — é a referência estrutural correta.

## Achado crítico: prazo do relatório final é hoje

O cronograma do Edital 05/2025-PROPESP/UFAM lista a **Submissão do Relatório
Técnico Final no e-Campus, juntamente com o Termo RIU, no período de
01/08/2026 a 30/08/2026**, sem horário-limite publicado para o último dia.
Diferente do que a auditoria anterior registrou para o PIBITI (prazo até
31/08/2026), **o prazo do PIBIC termina hoje, 30/08/2026**. Não há base
documental para supor 23h59; o envio deve ser feito com a maior antecedência
possível dentro do dia.

## Correção estrutural aplicada nesta rodada

O corpo do relatório (`pibic/relatorio_final_2026/`) foi ajustado para a
identificação PIBIC/PROPESP (capa, folha de rosto, tabela "Dados do Projeto",
metadados do PDF) e recebeu um **capítulo novo de Revisão Bibliográfica**
(`secoes/01b_revisao_bibliografica.tex`), exigido explicitamente pelo guia do
Comitê de Exatas/Engenharia e que não existia como seção própria antes desta
rodada (havia apenas um esboço dentro da Introdução, sem texto redigido). O
capítulo tem quatro eixos, cada um ancorado nas referências já presentes em
`referencias.bib` (BMC, verificação de redes quantizadas, DDPG/CartPole, e a
delimitação sobre IA generativa) e ocupa cerca de duas páginas, dentro do
limite do guia. O PDF foi recompilado com sucesso (33 páginas, ~460 KB, bem
abaixo de qualquer limite de tamanho usual) e não restaram referências ou
citações não resolvidas.

## Pendências sem evidência suficiente

### 1. Agência de fomento do bolsista

`config/dados_projeto.tex` registrava **FAPEMA** (Fundação de Amparo à
Pesquisa do Maranhão) como órgão de fomento. FAPEMA não consta no Edital PIBIC
2025-2026 da UFAM (Amazonas) e não é uma das modalidades listadas
(PIBIC/UFAM, PIBIC/CNPq, PAIC/FAPEAM). É muito provável que o valor correto
seja **FAPEAM** (Fundação de Amparo à Pesquisa do Estado do Amazonas) ou,
dado que a modalidade já registrada é "Bolsa CNPq", que o campo de fomento
deva simplesmente ser **CNPq**. O campo foi marcado como `[CONFIRMAR NO
ECAMPUS]` em vez de ser corrigido por inferência; **isto precisa ser resolvido
com o orientador antes do envio**, pois é um dado de concessão de bolsa, não
um detalhe editorial.

### 2. Código do projeto no eCampus

`pibic/relatorio_final_2026/config/dados_projeto.tex` ainda contém o campo do
código. Copiar o valor exatamente do registro oficial no eCampus.

### 3. Propriedade intelectual e apresentação reservada

Os campos de potencial de patente/registro de software e apresentação
reservada exigem decisão do orientador e, se aplicável, do Núcleo de Inovação
Tecnológica. Se houver interesse de proteção, definir também eventual embargo
antes da divulgação.

### 4. Cronograma documental inicial

O apêndice do relatório contém um cronograma baseado nos registros técnicos
encontrados. Os períodos de setembro/2025 a abril/2026 ainda precisam ser
confrontados com diário de atividades, relatórios parciais ou registros do
orientador para que o cronograma seja oficialmente completo.

### 5. Termo RIU e assinaturas

Ainda é necessário obter a versão vigente do Termo de Autorização e
Declaração de distribuição não exclusiva de publicação digital no RIU
(exigido pelo item 11.4 do Edital PIBIC), preencher os dados, definir
confidencialidade/propriedade intelectual, colher as assinaturas e depositá-lo
junto ao Relatório Final no e-Campus.

### 6. Confirmação operacional do envio

Não há protocolo ou comprovante de submissão arquivado. O prazo é **hoje,
30/08/2026**, sem horário-limite publicado. Enviar com a maior antecedência
possível e preservar o recibo/comprovante gerado pelo e-Campus.

## Situação técnica

O harness corrigido, a auditoria de evidências, os diagramas, a discussão dos
resultados e a compilação do relatório (agora com identificação PIBIC/PROPESP
e o capítulo de Revisão Bibliográfica) já foram executados e verificados
nesta rodada. As pendências acima são administrativas ou documentais e não
devem ser preenchidas por inferência.
