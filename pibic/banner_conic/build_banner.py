"""Preenche o modelo OFICIAL do banner (Modelo Banner - CONIC 35-1.pptx, 1080x1920, Arial) e exporta PNG.
Uso: python3 build_banner.py            (usa qr.png, se existir, senão deixa o espaço do QR)"""
import copy, math, os, subprocess, sys, shutil
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.text.text import _Paragraph
from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "modelo_oficial", "Modelo Banner - CONIC 35-1.pptx")
SAIDA_PPTX = os.path.join(AQUI, "banner_conic_pibiti.pptx")
SAIDA_PNG = os.path.join(AQUI, "banner_conic_pibiti.png")
QR = os.path.join(AQUI, "qr.png")
PX = 9525
LB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
LR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
COL_W, XL, XR, INS = 462, 73, 546, 4
BOTTOM = 1792           # limite do conteudo (logos do rodape comecam em ~1807)
HDR_H, GAP = 49, 12

# ---------------- conteudo (negrito marcado com **) ----------------
TITULO = "Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC"
AUTORES = [("¹Matheus Serrão Uchôa – Bolsa CNPq", True), ("²Prof. Dr. Iury Valente de Bessa (Orientador) – UFAM", True),
           ("¹matheus.uchoa@ufam.edu.br – ²iurybessa@ufam.edu.br", False)]
INTRO = ["Um controlador neural pode passar nos testes e falhar em estados nunca amostrados, sobretudo com aritmética de precisão reduzida (GOPINATH et al., 2021).",
         "**Pergunta:** que garantias o ESBMC dá a um controlador DDPG quantizado em Q8.8?",
         "**Hipótese:** Q8.8 preserva o comportamento do Float32 e o ESBMC prova segurança no horizonte verificado."]
METOD = ["**Fluxo:** pesos DDPG 4–24–24–1 → Q8.8 → harness em C → ESBMC + Z3 → contraexemplo reproduzido no simulador Cart-Pole.",
         "**Amostra:** 10.000 estados (RandomState 42), Float32 × Q8.8.",
         "**Setup:** ESBMC 6.8.0, Windows 10, solver Z3; tempo-limite de 4 a 12 s.",
         "**Métricas:** erro médio, P95, P99 e máximo; veredicto SUCCESSFUL, FAILED ou TIMEOUT."]
RESULT_TXT = ["**Figura 1:** erro absoluto Float32 × Q8.8 em 10.000 estados. Fonte: autores.",
              "O erro máximo de 7,7589 N inverte o sinal da ação no pior caso (+6,43 N → −1,33 N).",
              "**Provado:** |F| ≤ 10 N (Z3, 3,5 s). **Reproduzido:** contraexemplo de 1 passo. **Inconclusivo:** propriedade direcional (timeout de 12 s).",
              "Estrutura: camada 1, 24/24 neurônios ativáveis; camada 2, 17/24 e 7 timeouts."]
CONCL = ["A hipótese foi **parcialmente refutada**: Q8.8 pode inverter a ação, e a prova vale só no domínio e no horizonte do harness (1 passo, dinâmica linearizada).",
         "**Contribuição:** fluxo rastreável que distingue prova, refutação e resultado inconclusivo.",
         "**Futuro:** validar o checkpoint PyTorch, particionar o domínio e ampliar o horizonte."]
AGRAD = ["Ao CNPq, pela bolsa PIBITI; ao orientador, Prof. Dr. Iury Valente de Bessa; e à UFAM."]
REFS = ["GOPINATH, D. et al. Verifying Quantized Neural Networks using SMT-Based Model Checking. ASE, 2021.",
        "MENG, K. et al. QNNVerifier: a tool for verifying neural networks using SMT-based model checking. ICSE Companion, 2022."]

# ---------------- medicao de texto (Liberation Sans = metrica do Arial) ----------------
def fnt(px, bold=False): return ImageFont.truetype(LB if bold else LR, max(1, round(px)))
def runs(s):
    out, bold = [], False
    for i, part in enumerate(s.split("**")):
        if part: out.append((part, i % 2 == 1))
    return out
def nlines(par, size_pt, width):
    px = size_pt * 4 / 3; words = []
    for t, b in runs(par):
        for w in t.split(" "):
            if w: words.append((w, b))
    lines, cur = 1, 0.0
    sp = fnt(px).getlength(" ")
    for w, b in words:
        wl = fnt(px, b).getlength(w)
        if cur and cur + sp + wl > width: lines += 1; cur = wl
        else: cur = wl if not cur else cur + sp + wl
    return lines
def box_h(pars, size_pt, width=COL_W - 2 * INS, para_gap=0):
    lh = size_pt * 4 / 3 * 1.2
    return sum(nlines(p, size_pt, width) for p in pars) * lh + para_gap * (len(pars) - 1) + 2 * INS + 4

CHART_H_FACTOR = 400 / 912
def col_layout(items, size_pt, y0):
    """items: lista de ('sec', titulo, pars, extra_px) -> posicoes; devolve y final."""
    y, pos = y0, []
    for kind, title, pars, extra in items:
        h = box_h(pars, size_pt, para_gap=4) + extra
        pos.append((title, y, y + HDR_H + 6, h)); y += HDR_H + 6 + h + GAP
    return pos, y

# ---------------- figura ----------------
def chart(path):
    W, H = 912, 400; im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    d.text((6, 4), "Erro |Float32 − Q8.8| (N), escala log", font=fnt(34, True), fill="#10282c")
    dados = [("Médio", 0.0778, "#0f8b8d"), ("P95", 0.2262, "#5b9a3c"), ("P99", 1.0927, "#e8962e"), ("Máximo", 7.7589, "#d62839")]
    x0, wmax, lo, hi = 190, 470, math.log10(0.05), math.log10(10)
    for i, (n, v, c) in enumerate(dados):
        y = 62 + i * 82; w = (math.log10(v) - lo) / (hi - lo) * wmax
        d.text((6, y + 14), n, font=fnt(36), fill="#10282c")
        d.rectangle([x0, y, x0 + w, y + 62], fill=c)
        d.text((x0 + w + 12, y + 14), f"{v:.4f}".replace(".", ",") + " N", font=fnt(36, True), fill="#10282c")
    im.save(path)

# ---------------- escolha do tamanho ----------------
def plan(size):
    res_extra = round(COL_W * CHART_H_FACTOR) + 8
    left = [("sec", "INTRODUÇÃO", INTRO, 0), ("sec", "METODOLOGIA", METOD, 0), ("sec", "RESULTADOS E DISCUSSÕES", RESULT_TXT, res_extra)]
    right = [("sec", "CONCLUSÃO", CONCL, 0), ("sec", "AGRADECIMENTOS", AGRAD, 0), ("sec", "REFERÊNCIAS", REFS, 0)]
    return left, right

# titulo/autores
T_PT, A_PT = 44, 16
t_lines = nlines("**" + TITULO + "**", T_PT, 932 - 2 * INS)
t_h = t_lines * T_PT * 4 / 3 * 1.2 + 8
a_h = len(AUTORES) * A_PT * 4 / 3 * 1.2 + 2 * INS + 6
T_TOP = 246; A_TOP = T_TOP + t_h + 4; Y0 = A_TOP + a_h + 8
QR_H = 250 + 10
best = None
for size in [20, 19, 18, 17, 16, 15, 14, 13, 12]:
    L, R = plan(size)
    lp, ly = col_layout(L, size, Y0); rp, ry = col_layout(R, size, Y0)
    if ly - GAP <= BOTTOM and ry - GAP + QR_H <= BOTTOM:
        best = (size, lp, rp); break
if not best: sys.exit("conteudo nao cabe nem a 12 pt")
SIZE, LP, RP = best
print(f"fonte do corpo: {SIZE} pt | titulo {T_PT} pt ({t_lines} linhas) | autores {A_PT} pt | y0={Y0:.0f}")

# ---------------- preenche o modelo ----------------
prs = Presentation(MODELO); s = prs.slides[0]
sh = {x.name + str(i): x for i, x in enumerate(s.shapes)}
shapes = list(s.shapes)
def text_of(x): return x.text_frame.text if x.has_text_frame else ""
def find_text(prefix): return next(x for x in shapes if x.has_text_frame and text_of(x).strip().startswith(prefix))
def groups():
    return {g.shapes[0].text_frame.text.strip(): g for g in shapes if g.shape_type == 6}

def write(shape, pars, size, align=PP_ALIGN.JUSTIFY, left=None, top=None, width=COL_W, height=None, bold_all=False, gap=4):
    tf = shape.text_frame; tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
    body = tf._txBody; ps = body.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}p")
    base = copy.deepcopy(ps[0])
    for ch in list(base):
        if not ch.tag.endswith("}pPr"): base.remove(ch)
    ppr = base.find("{http://schemas.openxmlformats.org/drawingml/2006/main}pPr")
    if ppr is not None:
        for ch in list(ppr):
            if ch.tag.split('}')[1].startswith("bu"): ppr.remove(ch)
        for k in ("marL", "indent"): ppr.attrib.pop(k, None)
    for p in ps: body.remove(p)
    for t in pars:
        pn = copy.deepcopy(base); body.append(pn); para = _Paragraph(pn, tf)
        para.alignment = align; para.space_after = Pt(gap * 0.75); para.space_before = Pt(0); para.line_spacing = 1.0
        for txt, b in runs(t):
            r = para.add_run(); r.text = txt; r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = bool(b or bold_all)
    shape.left, shape.width = Emu((left if left is not None else shape.left // PX) * PX), Emu(width * PX)
    if top is not None: shape.top = Emu(int(top) * PX)
    if height is not None: shape.height = Emu(int(height) * PX)

# titulo
tit = [x for x in shapes if x.has_text_frame and text_of(x).strip().startswith("TÍTULO")][0]
write(tit, ["**" + TITULO + "**"], T_PT, PP_ALIGN.CENTER, left=76, top=T_TOP, width=932, height=t_h, gap=0)
aut = [x for x in shapes if x.has_text_frame and text_of(x).strip().startswith("1Nome")][0]
aut_text = [(("**" + t + "**") if b else t) for t, b in AUTORES]
write(aut, aut_text, A_PT, PP_ALIGN.CENTER, left=67, top=A_TOP, width=932, height=a_h, gap=0)

# secoes
G = groups()
body_boxes = {"INTRODUÇÃO": find_text("A introdução"), "METODOLOGIA": find_text("Descrever como"), "RESULTADOS E DISCUSSÕES": find_text("Nesta seção"),
              "CONCLUSÃO": find_text("A conclusão"), "AGRADECIMENTOS": find_text("Aqui, você"), "REFERÊNCIAS": find_text("ALVES")}
content = {"INTRODUÇÃO": INTRO, "METODOLOGIA": METOD, "RESULTADOS E DISCUSSÕES": RESULT_TXT, "CONCLUSÃO": CONCL, "AGRADECIMENTOS": AGRAD, "REFERÊNCIAS": REFS}
for col_x, P in ((XL, LP), (XR, RP)):
    for title, y_hdr, y_txt, h in P:
        g = G[title]; g.left, g.top = Emu(col_x * PX), Emu(int(y_hdr) * PX)
        extra = round(COL_W * CHART_H_FACTOR) + 8 if title == "RESULTADOS E DISCUSSÕES" else 0
        if title == "RESULTADOS E DISCUSSÕES":
            chart_png = os.path.join(AQUI, "_fig_erro.png"); chart(chart_png)
            s.shapes.add_picture(chart_png, Emu(col_x * PX), Emu(int(y_txt) * PX), width=Emu(COL_W * PX))
            write(body_boxes[title], content[title], SIZE, left=col_x, top=y_txt + extra, height=h - extra)
        else:
            write(body_boxes[title], content[title], SIZE, left=col_x, top=y_txt, height=h)

# remove instrucoes do modelo e o placeholder de logo da unidade (opcional)
for x in list(shapes):
    if x.has_text_frame and text_of(x).strip().startswith(("ATENÇÃO", "LOGO DA UNIDADE")):
        x._element.getparent().remove(x._element)

# QR: 250x250 no canto inferior direito da coluna direita
qr_y = RP[-1][1] + HDR_H + 6 + RP[-1][3] + GAP
qr_y = BOTTOM - 250   # canto inferior direito
ph = find_text("") if False else next(x for x in shapes if x.has_text_frame and "QR CODE" in text_of(x))
if os.path.exists(QR):
    ph._element.getparent().remove(ph._element)
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu((XR + COL_W - 262) * PX), Emu(int(qr_y - 6) * PX), Emu(262 * PX), Emu(262 * PX))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(255, 255, 255); bg.line.fill.background()
    s.shapes.add_picture(QR, Emu((XR + COL_W - 250) * PX), Emu(int(qr_y) * PX), width=Emu(250 * PX), height=Emu(250 * PX))
else:
    write(ph, ["**COLOQUE O QR CODE DO RESUMO EXPANDIDO AQUI**"], 14, PP_ALIGN.CENTER, left=XR + COL_W - 250, top=qr_y, width=250, height=250)
    ph.text_frame.vertical_anchor = 3  # meio
    ph.line.width = Pt(2)
prs.save(SAIDA_PPTX)

# ---------------- exporta PNG (LibreOffice -> PDF -> PNG 1080x1920) ----------------
tmp = os.path.join(AQUI, "_render"); os.makedirs(tmp, exist_ok=True)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, SAIDA_PPTX], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "1080", "-scale-to-y", "1920", os.path.join(tmp, "banner_conic_pibiti.pdf"), os.path.join(tmp, "p")], check=True)
Image.open(os.path.join(tmp, "p.png")).convert("RGB").save(SAIDA_PNG, optimize=True)
shutil.rmtree(tmp); os.remove(os.path.join(AQUI, "_fig_erro.png"))
print("ok", SAIDA_PNG, Image.open(SAIDA_PNG).size, round(os.path.getsize(SAIDA_PNG) / 1024), "KB")
