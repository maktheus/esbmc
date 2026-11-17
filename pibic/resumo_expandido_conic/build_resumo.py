"""Preenche o modelo oficial do Resumo Expandido (XXXV CONIC) sem alterar sua formatação."""
import copy, math, re, shutil, sys, zipfile
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from PIL import Image, ImageDraw, ImageFont

MODELO, SAIDA = sys.argv[1], sys.argv[2]
F = "/usr/share/fonts/truetype/liberation/LiberationSans-%s.ttf"

# ---------- figura (barras, escala log) ----------
W, H = 1600, 800
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
fb, fr, fs = ImageFont.truetype(F % "Bold", 34), ImageFont.truetype(F % "Regular", 32), ImageFont.truetype(F % "Bold", 32)
d.text((20, 15), "Erro absoluto |Float32 − Q8.8| na saída do controlador (N), escala logarítmica", font=fb, fill="#10282c")
dados = [("Erro médio", 0.0778, "#0f8b8d"), ("Percentil 95", 0.2262, "#5b9a3c"), ("Percentil 99", 1.0927, "#e8962e"), ("Erro máximo", 7.7589, "#d62839")]
x0, wmax, lo, hi = 330, 900, math.log10(0.05), math.log10(10)
for i, (n, v, c) in enumerate(dados):
    y = 130 + i * 150
    w = (math.log10(v) - lo) / (hi - lo) * wmax
    d.text((20, y + 28), n, font=fr, fill="#10282c")
    d.rectangle([x0, y, x0 + w, y + 90], fill=c)
    d.text((x0 + w + 14, y + 28), f"{v:.4f}".replace(".", ",") + " N", font=fs, fill="#10282c")
d.line([(x0, 100), (x0, 740)], fill="#10282c", width=3)
im.save("/tmp/fig_erro.jpg", quality=92)

# ---------- documento ----------
doc = Document(MODELO)
P = doc.paragraphs

def set_text(p, text, size=None, align=None, bold_label=None):
    """Troca o texto mantendo o rPr do primeiro run; remove hyperlinks e demais runs."""
    el = p._p
    runs = el.xpath("./w:r")
    rpr = copy.deepcopy(runs[0].find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr")) if runs else None
    for ch in list(el):
        if ch.tag.endswith("}r") or ch.tag.endswith("}hyperlink"):
            el.remove(ch)
    r = p.add_run(text)
    if rpr is not None:
        r._r.insert(0, rpr)
    if size: r.font.size = Pt(size)
    if align is not None: p.alignment = align
    return r

def clone_after(p, text, **kw):
    new = copy.deepcopy(p._p); p._p.addnext(new)
    from docx.text.paragraph import Paragraph
    q = Paragraph(new, p._parent); set_text(q, text, **kw); return q

# título e autores (blocos de coautores 1-3 removidos: "se for o caso")
set_text(P[0], "VERIFICAÇÃO FORMAL DE SISTEMAS DE INTELIGÊNCIA ARTIFICIAL GENERATIVA E CONTROLADORES NEURAIS UTILIZANDO ESBMC")
set_text(P[1], "Matheus Serrão Uchôa")
set_text(P[3], "matheus.uchoa@ufam.edu.br")
set_text(P[17], "Iury Valente de Bessa")
set_text(P[19], "iurybessa@ufam.edu.br")
for i in range(4, 16): P[i]._p.getparent().remove(P[i]._p)
# (os parágrafos 4–15 são a linha em branco e os coautores 1–3)

RESUMO = ("Controladores neurais são cada vez mais usados em sistemas de controle, mas bom desempenho em simulação não garante comportamento "
 "correto em todos os estados admitidos, sobretudo com aritmética de precisão reduzida. São escassas as evidências formais rastreáveis que associem "
 "quantização, domínio de entradas e horizonte de verificação ao veredicto. Pergunta-se que garantias o ESBMC oferece a um controlador "
 "DDPG quantizado em ponto fixo Q8.8. Propõe-se um fluxo que converte os pesos para Q8.8, gera harnesses em C, verifica propriedades com "
 "o ESBMC e o solver Z3 e reproduz contraexemplos no simulador do Cart-Pole. Em 10.000 estados, o erro médio entre Float32 e Q8.8 foi de "
 "0,0778 N e o máximo de 7,7589 N, com inversão do sinal da ação no pior caso; o limite |F| ≤ 10 N foi provado em 3,5 s e a propriedade "
 "direcional ficou inconclusiva (timeout). A contribuição é um fluxo que liga cada afirmação ao domínio e ao artefato que a sustenta, "
 "distinguindo prova, refutação e resultado inconclusivo.")
assert len(RESUMO.split()) <= 200, len(RESUMO.split())
P = doc.paragraphs
idx = {p.text.strip(): i for i, p in enumerate(P) if p.text.strip()}
def find(prefix): return next(p for p in doc.paragraphs if p.text.strip().startswith(prefix))

set_text(find("A primeira página deve constar"), RESUMO)
pc = find("Palavras-Chave:")
rs = pc.runs; rs[1].text = " verificação formal; ESBMC; controlador neural; Cart-Pole; quantização Q8.8."
for r in rs[2:]: r._r.getparent().remove(r._r)

# Introdução (quebra de página: a 1ª página só leva título, autores, resumo e palavras-chave)
h_intro = next(p for p in doc.paragraphs if p.text.strip() == "INTRODUÇÃO")
h_intro.paragraph_format.page_break_before = True
set_text(find("O Resumo Expandido do XXXV"),
 "O uso de modelos de aprendizado de máquina em sistemas que tomam decisões ou acionam componentes de forma autônoma amplia a "
 "necessidade de evidências de segurança auditáveis. Um controlador neural pode agir bem nas trajetórias testadas e, ainda assim, "
 "produzir uma ação inadequada em regiões do espaço de estados nunca amostradas, o que se agrava com aritmética de precisão reduzida, "
 "em que arredondamentos e saturações deslocam a fronteira entre ações. Métodos formais baseados em teorias SMT complementam os testes "
 "ao buscar, de modo sistemático e dentro de um limite, uma atribuição que viole uma propriedade (GOPINATH et al., 2021; MENG et al., 2022).")
set_text(find("Padronizar o formato"),
 "Gopinath et al. (2021) verificam redes neurais quantizadas por model checking baseado em SMT, e Meng et al. (2022) descrevem a "
 "ferramenta QNNVerifier, que automatiza essa análise. Este trabalho complementa essa linha ao tratar um controlador neural em malha "
 "fechada (DDPG aplicado ao Cart-Pole) e ao tornar explícita a cadeia entre quantização, domínio, horizonte, veredicto e contraexemplo.")
set_text(find("O trabalho que não for"),
 "Pergunta de pesquisa: que garantias o ESBMC consegue produzir para um controlador DDPG quantizado em Q8.8, e sob quais hipóteses "
 "do modelo? Seguem objetivo, metodologia, resultados e conclusões.")
p = find("Lembre-se que a introdução"); p._p.getparent().remove(p._p)

set_text(find("O objetivo geral em um Resumo"),
 "Avaliar, por meio do ESBMC, como a quantização Q8.8 e a verificação limitada de propriedades de um controlador neural DDPG para "
 "o Cart-Pole influenciam a produção de evidências formais reprodutíveis, identificando os limites para reutilizar o fluxo em "
 "componentes de Inteligência Artificial Generativa.")
set_text(find("A seção de metodologia"),
 "Delineamento: estudo de caso experimental e reprodutível. O fluxo é: pesos do ator DDPG (4–24–24–1, ReLU nas camadas ocultas e tanh "
 "aproximada na saída) → conversão para Q8.8 (ponto fixo, fator de escala 256) e comparação com Float32 → harnesses em C com as "
 "propriedades → ESBMC com Z3 → veredicto, contraexemplo e reprodução no simulador do Cart-Pole. Setup: ESBMC 6.8.0 (executável "
 "Windows 64 bits, Windows 10) com o solver Z3, pois o Boolector não está disponível nesse executável; 10.000 estados gerados com "
 "RandomState(42); tempos-limite de 12 s (propriedade direcional), 8 s (atividade) e 4 s (saturação). Métricas: erro absoluto médio, "
 "percentis 95 e 99 e máximo entre Float32 e Q8.8, e veredicto (SUCCESSFUL, FAILED, TIMEOUT ou UNKNOWN) com o tempo do solver. "
 "Comandos, logs e hashes SHA-256 foram registrados em um manifesto. SUCCESSFUL é prova apenas dentro do domínio, do modelo e do "
 "horizonte declarados; TIMEOUT e UNKNOWN são resultados inconclusivos.")

res = find("A seção de resultados e discussão")
set_text(res,
 "Em 10.000 estados, o erro absoluto médio entre as saídas Float32 e Q8.8 foi de 0,0778 N, com percentis 95 e 99 de 0,2262 N e "
 "1,0927 N e máximo de 7,7589 N (Figura 1). No pior estado, a ação inverteu de sinal (+6,4308 N em Float32; −1,3281 N em Q8.8). A "
 "aproximação da tanh teve erro máximo de 0,0457 (4,57% do intervalo unitário). Assim, Q8.8 não deve ser tratado como equivalente a "
 "Float32 sem qualificar o domínio e a cauda de erro.")
p2 = clone_after(res,
 "Na verificação formal (Tabela 1), o limite de força |F| ≤ 10 N terminou SUCCESSFUL com o Z3 em 3,5 s; trata-se de uma prova do "
 "programa quantizado e do domínio codificado, não do controlador Float32 nem da dinâmica não linear completa. Os estados de "
 "contraexemplo publicados foram reproduzidos, o que confirma a consistência dos harnesses, mas não refuta sozinho a propriedade "
 "universal. A propriedade direcional atingiu TIMEOUT de 12 s e permanece inconclusiva.")

# figura: troca a imagem de exemplo por uma do trabalho
d_el = doc.element.body.xpath(".//wp:extent")[0]; cx, cy = 4800000, 2400000
for e in doc.element.body.xpath(".//wp:extent"): e.set("cx", str(cx)); e.set("cy", str(cy))
for e in doc.element.body.xpath(".//a:ext[@cx]"): e.set("cx", str(cx)); e.set("cy", str(cy))
cap = find("Figura 1:"); rr = cap.runs
rr[1].text = "Erro absoluto entre as saídas Float32 e Q8.8 em 10.000 estados (escala logarítmica). Fonte: autores."
for r in rr[2:]: r._r.getparent().remove(r._r)
capt = find("Tabela 1:"); rr = capt.runs
rr[1].text = "Resultados da verificação com o ESBMC e o Z3. Fonte: autores."
for r in rr[2:]: r._r.getparent().remove(r._r)

# tabela 4x6 -> 4x3
t = doc.tables[0]
for row in t.rows:
    for c in row.cells[3:]: c._tc.getparent().remove(c._tc)
grid = t._tbl.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblGrid")
for g in list(grid)[3:]: grid.remove(g)
linhas = [("Propriedade", "Veredito (Z3)", "Interpretação"),
          ("Limite de força |F| ≤ 10 N", "SUCCESSFUL (3,5 s)", "Prova no domínio do harness"),
          ("Estados de contraexemplo (replay)", "SUCCESSFUL", "Reproduz; não refuta o caso universal"),
          ("Propriedade direcional (universal)", "TIMEOUT (12 s)", "Inconclusiva")]
for row, vals in zip(t.rows, linhas):
    for c, v in zip(row.cells, vals):
        c.width = Pt(150)
        p = c.paragraphs[0]; r = p.add_run(v); r.font.name = "Bahnschrift"; r.font.size = Pt(10)
        r.bold = (row is t.rows[0]); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

clone_after_t = copy.deepcopy(res._p); t._tbl.addnext(clone_after_t)
from docx.text.paragraph import Paragraph
p3 = Paragraph(clone_after_t, res._parent)
set_text(p3,
 "Na rodada estrutural corrigida, o Z3 encontrou contraexemplo de ativação para 24/24 neurônios da camada 1 e para 17/24 da camada 2 "
 "(7 casos em TIMEOUT); na análise de saturação foram 24/24 e 22/24 (2 TIMEOUT). Portanto, não há base para afirmação universal "
 "sobre os 48 neurônios. Como o veredicto vale para um harness de malha fechada com horizonte de um passo, os números não são comparáveis "
 "em desempenho com os dos trabalhos relacionados, que verificam redes isoladas; a comparação é metodológica.")

set_text(find("Neste capítulo devem"),
 "Retomando a pergunta, o ESBMC produziu uma prova delimitada (|F| ≤ 10 N) e a reprodução de contraexemplos, mas não uma certificação "
 "global. A quantização Q8.8 pode inverter a ação em casos extremos (erro máximo de 7,7589 N), e a prova vale apenas no domínio e no "
 "horizonte do harness; a propriedade direcional e parte da análise estrutural permaneceram inconclusivas. A contribuição é um fluxo "
 "rastreável que distingue prova, reprodução e inconclusão. Limitações: dinâmica linearizada, horizonte de um passo, domínio finito, "
 "dependência do solver e do tempo-limite, e pesos ainda não confirmados no checkpoint PyTorch. Trabalhos futuros: validar o checkpoint, "
 "particionar o domínio, avaliar horizontes maiores e comparar a dinâmica linearizada com a não linear.")

ref = find("As referências devem")
set_text(ref, "GOPINATH, D.; MENG, K.; PASAREANU, C. S.; ZHANG, Y.; BESSA, I.; CORDEIRO, L. C. Verifying Quantized Neural Networks using "
 "SMT-Based Model Checking. In: IEEE/ACM INTERNATIONAL CONFERENCE ON AUTOMATED SOFTWARE ENGINEERING (ASE), 36., 2021. [S. l.]: IEEE, 2021.",
 size=10, align=WD_ALIGN_PARAGRAPH.LEFT)
ref.paragraph_format.first_line_indent = 0
ref.paragraph_format.space_after = Pt(6)
r2 = clone_after(ref, "MENG, K.; GOPINATH, D.; PASAREANU, C. S.; ZHANG, Y.; BESSA, I.; CORDEIRO, L. C. QNNVerifier: A Tool for Verifying "
 "Neural Networks using SMT-Based Model Checking. In: INTERNATIONAL CONFERENCE ON SOFTWARE ENGINEERING (ICSE), 44., Companion Proceedings, 2022.",
 size=10, align=WD_ALIGN_PARAGRAPH.LEFT)

set_text(find("A sessão de agradecimentos"),
 "Ao CNPq, pela bolsa de iniciação (PIBITI); ao orientador, Prof. Dr. Iury Valente de Bessa; e à Universidade Federal do Amazonas.")
doc.save("/tmp/_tmp.docx")

# troca a imagem de exemplo (image1.jpg, a pizza de exemplo) pela figura gerada
zin = zipfile.ZipFile("/tmp/_tmp.docx"); zout = zipfile.ZipFile(SAIDA, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/media/image1.jpg": data = open("/tmp/fig_erro.jpg", "rb").read()
    zout.writestr(it, data)
zout.close()
print("ok", SAIDA, "palavras no resumo:", len(RESUMO.split()))
