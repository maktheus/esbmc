"""Preenche o modelo OFICIAL do banner (Modelo Banner - CONIC 35-1.pptx, 1080x1920, Arial) e exporta PNG.

Uso: python3 build_banner.py        (usa qr.png, se existir; senão deixa o espaço do QR marcado)

Layout (tres faixas sobre as barras e logos do modelo):
  faixa A  INTRODUÇÃO (esq.)            | METODOLOGIA + diagrama do fluxo (dir.)
  faixa B  RESULTADOS E DISCUSSÕES (largura total): painel dos 7 casos + 2 gráficos
  faixa C  CONCLUSÃO + AGRADECIMENTOS (esq.) | REFERÊNCIAS + QR (dir.)
Todos os numeros vem de pibic/results/EVIDENCIAS.md, pibic/ic3/EVIDENCIA.md,
pibic/artigo/caps/4resultados.tex e pibic/relatorio_final_2026/secoes/.
"""
import copy, math, os, shutil, subprocess, sys
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.text.text import _Paragraph
from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "modelo_oficial", "Modelo Banner - CONIC 35-1.pptx")
SAIDA_PPTX = os.path.join(AQUI, "banner_conic_pibiti.pptx")
SAIDA_PNG = os.path.join(AQUI, "banner_conic_pibiti.png")
QR = os.path.join(AQUI, "qr.png")
TMP = os.path.join(AQUI, "_build")
PX = 9525
LB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
LR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

XL, XR, COL_W, FULL_W = 73, 546, 462, 935
INS = 4
HDR_H, HDR_GAP, GAP = 49, 5, 8
BOTTOM = 1792            # logos do rodape comecam em 1807
INK, TEAL, ORANGE = "#10282c", "#27b5b7", "#f0922b"
GREEN, RED, AMBER, GREY = "#3f8f3a", "#c9263a", "#e08a1e", "#5f6b6e"

# ------------------------------------------------------------------ conteudo
TITULO = "Verificação Formal de Sistemas de Inteligência Artificial Generativa e Controladores Neurais utilizando ESBMC"
AUTORES = [("¹Matheus Serrão Uchôa – Bolsa CNPq", True),
           ("²Prof. Dr. Iury Valente de Bessa (Orientador) – UFAM", True),
           ("¹matheus.uchoa@ufam.edu.br – ²iurybessa@ufam.edu.br", False)]

INTRO = [
    "Redes neurais hoje chegam a kernels de inferência, a agentes que escrevem código e a controladores físicos. Testes cobrem só as execuções escolhidas: a rede pode acertar nelas e falhar num estado nunca amostrado, e a precisão reduzida (Q8.8) desloca a fronteira entre ações.",
    "**Pergunta:** o que um verificador formal (ESBMC, *bounded model checking* sobre SMT) consegue **provar** em sistemas de IA generativa e de controle, a que custo, e onde ele para?",
    "**Objetivo:** aplicar o ESBMC a sete casos, do neurônio ao sistema acoplado, com domínio, horizonte e hipóteses explícitos.",
]
METOD_CAP = [
    "**Figura 1:** cada caso vira um harness em C com propriedade e domínio; o veredito é tri-estado. **Setup:** ESBMC 6.8.0 (Z3, Boolector), yosys 0.33 + ABC; 10.000 estados, RandomState(42).",
]
RES_CAP_A = "**Figura 2:** os sete casos, com veredito e tempo de solver; cada veredito aponta para um log bruto versionado. Fonte: autores."
RES_CAP_B = "**Figura 3:** a parede do BMC está em K=4, não em 50; o IC3 prova ∀t no modelo sintético, mas não decide o ator real."
RES_CAP_C = "**Figura 4:** Q8.8 no ator DDPG: erro pequeno na média, mas no pior estado a ação **inverte o sinal**."
CONCL = [
    "**1. O ESBMC prova em todas as camadas**, do neurônio (MLP, FFN de GPT-2 e LLaMA) ao kernel, ao agente e ao controlador em malha fechada; mas toda prova vale só no domínio, horizonte e representação do harness.",
    "**2. A parede é estrutural:** 4 passos na malha fechada e N=4 no GEMM. O IC3 prova ∀t em 0,37 s no modelo sintético, mas não decide o ator real; a hipótese é a representação bit a bit (próximo passo: BTOR2).",
    "**3. Contribuição:** fluxo rastreável modelo → harness → veredito → log; duas propriedades vácuas foram achadas e reescritas, e TIMEOUT nunca vira \"seguro\".",
]
AGRAD = ["Ao CNPq, pela bolsa PIBITI; ao orientador, Prof. Dr. Iury Valente de Bessa; e à UFAM."]
REFS = [
    "BRADLEY, A. R. SAT-based model checking without unrolling. VMCAI, 2011.",
    "GOPINATH, D. et al. Verifying Quantized Neural Networks using SMT-Based Model Checking. ASE, 2021.",
    "MENG, K. et al. QNNVerifier: a tool for verifying neural networks using SMT-based model checking. ICSE Companion, 2022.",
]

# ------------------------------------------------------------------ medicao de texto
def fnt(px, bold=False):
    return ImageFont.truetype(LB if bold else LR, max(1, round(px)))

def runs(s):
    """'**negrito** normal' -> [(texto, bold)]; *italico* vira texto normal."""
    out, bold = [], False
    for i, part in enumerate(s.split("**")):
        part = part.replace("*", "").replace("`", "")
        if part:
            out.append((part, i % 2 == 1))
    return out

def nlines(par, size_pt, width):
    px = size_pt * 4 / 3
    words = [(w, b) for t, b in runs(par) for w in t.split(" ") if w]
    lines, cur, sp = 1, 0.0, fnt(px).getlength(" ")
    for w, b in words:
        wl = fnt(px, b).getlength(w)
        if cur and cur + sp + wl > width:
            lines += 1; cur = wl
        else:
            cur = wl if not cur else cur + sp + wl
    return lines

def box_h(pars, size_pt, width, gap=4):
    lh = size_pt * 4 / 3 * 1.18
    return sum(nlines(p, size_pt, width - 2 * INS) for p in pars) * lh + gap * (len(pars) - 1) + 2 * INS + 4

# ------------------------------------------------------------------ figuras (renderizadas a 2x)
S = 2
def canvas(w, h):
    im = Image.new("RGB", (w * S, h * S), "white")
    return im, ImageDraw.Draw(im)

def T(d, xy, txt, px, bold=False, fill=INK, anchor="la"):
    d.text((xy[0] * S, xy[1] * S), txt, font=fnt(px * S, bold), fill=fill, anchor=anchor)

def rbox(d, box, fill, outline=None, r=8, w=2):
    x0, y0, x1, y1 = [v * S for v in box]
    d.rounded_rectangle([x0, y0, x1, y1], radius=r * S, fill=fill, outline=outline, width=w * S)

def arrow(d, p0, p1, fill=INK, w=3):
    x0, y0, x1, y1 = p0[0] * S, p0[1] * S, p1[0] * S, p1[1] * S
    d.line([(x0, y0), (x1, y1)], fill=fill, width=w * S)
    ang = math.atan2(y1 - y0, x1 - x0); L = 9 * S
    for a in (ang + 2.6, ang - 2.6):
        d.line([(x1, y1), (x1 + L * math.cos(a), y1 + L * math.sin(a))], fill=fill, width=w * S)

def wrap(d, txt, px, width, bold=False):
    f = fnt(px * S, bold); words, lines, cur = txt.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width * S or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def fig_pipeline(path, W=COL_W, H=190):
    im, d = canvas(W, H)
    bw, bh, y = 104, 66, 6
    xs = [4, 4 + bw + 18, 4 + 2 * (bw + 18)]
    boxes = [("Modelo de IA", ["PyTorch · ONNX", "pesos Q8.8", "planta / kernel"]),
             ("Harness em C", ["propriedade", "domínio (assume)", "sem loops: linear"]),
             ("ESBMC", ["GOTO → symex", "fórmula SMT", "Z3 / Boolector"])]
    for x, (t, ls) in zip(xs, boxes):
        rbox(d, (x, y, x + bw, y + bh), "#eef8f8", TEAL)
        T(d, (x + bw / 2, y + 16), t, 15, True, anchor="mm")
        for i, l in enumerate(ls):
            T(d, (x + bw / 2, y + 31 + i * 13), l, 11, anchor="mm", fill=GREY)
    for x in xs[:2]:
        arrow(d, (x + bw + 2, y + bh / 2), (x + bw + 16, y + bh / 2), TEAL)
    # vereditos (tri-estado)
    vx = xs[2] + bw + 18; vw = W - vx - 4
    for i, (lab, col) in enumerate([("SAFE", GREEN), ("UNSAFE", RED), ("TIMEOUT", AMBER)]):
        yy = y + i * 23
        rbox(d, (vx, yy, vx + vw, yy + 20), col, r=6)
        T(d, (vx + vw / 2, yy + 10), lab, 12, True, fill="white", anchor="mm")
    arrow(d, (xs[2] + bw + 2, y + bh / 2), (vx - 2, y + bh / 2), TEAL)
    # linha inferior: casos + rota ilimitada
    y2 = y + bh + 10
    T(d, (4, y2), "7 casos:", 12, True)
    T(d, (62, y2), "MLP/QNN · kernels GEMM · agente neuro-simbólico · PID sob caos ·", 11.5)
    T(d, (62, y2 + 15), "FFN de GPT-2 e LLaMA-7B · política RL · Cart-Pole DDPG (malha fechada)", 11.5)
    y3 = y2 + 36
    rbox(d, (4, y3, W - 4, y3 + 46), "#fff6ea", ORANGE)
    T(d, (12, y3 + 8), "Prova ilimitada (para todo t):", 12.5, True)
    T(d, (12, y3 + 27), "Verilog → Yosys (904 242 portas AND, 65 latches) → AIGER → ABC IC3/PDR", 11.5)
    y4 = y3 + 54
    T(d, (4, y4), "SAFE = prova no domínio e horizonte · UNSAFE = contraexemplo reproduzido · TIMEOUT = indeciso", 10.5, fill=GREY)
    im.save(path); return H

CASOS = [
    ("1", "Redes neurais (MLP XOR, QNN; stub de Transformer)", [("SAFE", GREEN, "0,08 s"), ("UNSAFE", RED, "198 s, propriedade violada")]),
    ("2", "Kernels de inferência (GEMM com tiling, atenção)", [("SAFE", GREEN, "N≤3, 17 s"), ("INDECISO", AMBER, "N=4, >300 s")]),
    ("3", "Agente neuro-simbólico: gera → verifica → corrige", [("UNSAFE→SAFE", GREEN, "2 iterações, <0,15 s")]),
    ("4", "PID sob caos: ruído ±5 no sensor, 10 passos", [("SAFE", GREEN, "T < 150 °C, 217 s")]),
    ("5", "FFN de LLM: GPT-2 e LLaMA-7B (fatias 4×8, 4×16)", [("SAFE", GREEN, "0,24–20 s, abstração intervalar")]),
    ("6", "Política RL actor-critic: ação em [−1, 1]", [("UNSAFE", RED, "contraexemplo em 2,6 s")]),
    ("7", "Cart-Pole DDPG Q8.8 em malha fechada", [("SAFE", GREEN, "|F| ≤ 10 N, 3,5 s"), ("INDECISO", AMBER, "direção, timeout 12 s")]),
]
def fig_scorecard(path, W=FULL_W, H=210):
    im, d = canvas(W, H)
    cols, rows = 2, 4
    cw, ch = (W - 10) / cols, (H - 6) / rows
    cells = CASOS + [("Σ", "12 harnesses reexecutados com log bruto", [("9 provados", GREEN, ""), ("3 contraexemplos", RED, ""), ("0 indecisos", GREY, "")])]
    for i, (num, name, chips) in enumerate(cells):
        c, r = i % cols, i // cols
        x0, y0 = c * (cw + 10), r * ch
        x1, y1 = x0 + cw, y0 + ch - 6
        rbox(d, (x0, y0, x1, y1), "#f4fafa" if num != "Σ" else "#fff6ea", TEAL if num != "Σ" else ORANGE, r=7)
        rbox(d, (x0 + 6, y0 + 7, x0 + 34, y0 + 35), TEAL if num != "Σ" else ORANGE, r=6)
        T(d, (x0 + 20, y0 + 21), num, 14, True, fill="white", anchor="mm")
        T(d, (x0 + 42, y0 + 15), name, 12.5, True, anchor="lm")
        xx = x0 + 42
        for lab, col, sub in chips:
            wlab = fnt(11 * S, True).getlength(lab) / S + 12
            rbox(d, (xx, y0 + 28, xx + wlab, y0 + 46), col, r=5)
            T(d, (xx + wlab / 2, y0 + 37), lab, 11, True, fill="white", anchor="mm")
            xx += wlab + 5
            if sub:
                T(d, (xx, y0 + 37), sub, 11.5, anchor="lm"); xx += fnt(11.5 * S).getlength(sub) / S + 12
    im.save(path); return H

def fig_bmc_wall(path, W=COL_W, H=160):
    im, d = canvas(W, H)
    T(d, (4, 3), "Tempo para decidir a malha fechada (s, escala log)", 12.5, True)
    pitch = (H - 26 - 16) / 6
    data = [("BMC K=1", 0.3, TEAL, "seguro até 1 passo"), ("BMC K=2 (interp.)", 3.0, "#9fd3d4", ""), ("BMC K=4", 909, TEAL, "seguro até 4, 383 MB"),
            ("IC3 sintético", 0.37, GREEN, "provado p/ todo t"), ("IC3 ator real", 1802, RED, "não convergiu, 708 MB"), ("BMC ABC ator real", 901, AMBER, "5 frames, indeciso")]
    x0, wmax, lo, hi = 118, 200, math.log10(0.1), math.log10(3000)
    for i, (n, v, col, sub) in enumerate(data):
        y = 24 + i * pitch; bh = pitch - 5
        w = (math.log10(v) - lo) / (hi - lo) * wmax
        T(d, (x0 - 6, y + bh / 2), n, 11, anchor="rm")
        rbox(d, (x0, y, x0 + w, y + bh), col, r=4)
        lab = f"{v:g}".replace(".", ",") + " s"
        T(d, (x0 + w + 5, y + bh / 2), lab, 11.5, True, anchor="lm")
        if sub: T(d, (x0 + w + 5 + fnt(11.5 * S, True).getlength(lab) / S + 6, y + bh / 2), sub, 10, fill=GREY, anchor="lm")
    T(d, (4, H - 12), "K=2 interpolado só para leitura; medidos: 0,3 s (K=1) e 909 s (K=4), rede sintética de 24 neurônios.", 9, fill=GREY)
    im.save(path); return H

def fig_q88(path, W=COL_W, H=160):
    im, d = canvas(W, H)
    T(d, (4, 3), "Erro |Float32 − Q8.8| na saída (N), 10.000 estados", 12.5, True)
    pitch = (H - 26 - 34) / 4
    data = [("Erro médio", 0.0778, TEAL), ("Percentil 95", 0.2262, GREEN), ("Percentil 99", 1.0927, AMBER), ("Erro máximo", 7.7589, RED)]
    x0, wmax, lo, hi = 100, 250, math.log10(0.05), math.log10(10)
    for i, (n, v, col) in enumerate(data):
        y = 24 + i * pitch; bh = pitch - 6
        w = (math.log10(v) - lo) / (hi - lo) * wmax
        T(d, (x0 - 6, y + bh / 2), n, 11.5, anchor="rm")
        rbox(d, (x0, y, x0 + w, y + bh), col, r=4)
        T(d, (x0 + w + 6, y + bh / 2), f"{v:.4f}".replace(".", ",") + " N", 12, True, anchor="lm")
    T(d, (4, H - 28), "Pior estado: +6,4308 N (Float32) → −1,3281 N (Q8.8): inversão de sinal.", 11, True, fill=RED)
    T(d, (4, H - 13), "tanh aproximada: erro máx. 4,57 % do intervalo unitário.", 10, fill=GREY)
    im.save(path); return H

# ------------------------------------------------------------------ pptx helpers
def write(shape, pars, size, align=PP_ALIGN.JUSTIFY, left=None, top=None, width=COL_W, height=None, gap=4, anchor=None):
    tf = shape.text_frame; tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
    body = tf._txBody; ps = body.findall(A + "p")
    base = copy.deepcopy(ps[0])
    for ch in list(base):
        if not ch.tag.endswith("}pPr"): base.remove(ch)
    ppr = base.find(A + "pPr")
    if ppr is not None:
        for ch in list(ppr):
            if ch.tag.split("}")[1].startswith("bu"): ppr.remove(ch)
        for k in ("marL", "indent"): ppr.attrib.pop(k, None)
    for p in ps: body.remove(p)
    for t in pars:
        pn = copy.deepcopy(base); body.append(pn); para = _Paragraph(pn, tf)
        para.alignment = align; para.space_after = Pt(gap * 0.75); para.space_before = Pt(0); para.line_spacing = 1.0
        for txt, b in runs(t):
            r = para.add_run(); r.text = txt; r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = bool(b)
    if left is not None: shape.left = Emu(int(left) * PX)
    shape.width = Emu(int(width) * PX)
    if top is not None: shape.top = Emu(int(top) * PX)
    if height is not None: shape.height = Emu(int(height) * PX)
    if anchor is not None: tf.vertical_anchor = anchor

def place_header(group, x, y, w=COL_W):
    """Move e redimensiona um cabecalho (grupo barra turquesa + filete laranja) do modelo."""
    g = group._element; xfrm = g.find(".//" + A + "xfrm")
    ext, chext, choff = xfrm.find(A + "ext"), xfrm.find(A + "chExt"), xfrm.find(A + "chOff")
    chx = int(chext.get("cx")); scale = w * PX / chx
    choff.set("x", str(int(int(choff.get("x")) * scale)))
    for child in group.shapes:
        child.left = Emu(int(child.left * scale)); child.width = Emu(int(child.width * scale))
    ext.set("cx", str(w * PX)); chext.set("cx", str(w * PX))
    group.left, group.top = Emu(x * PX), Emu(int(y) * PX)

# ------------------------------------------------------------------ montagem
os.makedirs(TMP, exist_ok=True)
prs = Presentation(MODELO); s = prs.slides[0]
shapes = list(s.shapes)
def text_of(x): return x.text_frame.text if x.has_text_frame else ""
def find_text(prefix): return next(x for x in shapes if x.has_text_frame and text_of(x).strip().startswith(prefix))
G = {g.shapes[0].text_frame.text.strip(): g for g in shapes if g.shape_type == 6}
BOX = {"INTRODUÇÃO": find_text("A introdução"), "METODOLOGIA": find_text("Descrever como"), "RESULTADOS E DISCUSSÕES": find_text("Nesta seção"),
       "CONCLUSÃO": find_text("A conclusão"), "AGRADECIMENTOS": find_text("Aqui, você"), "REFERÊNCIAS": find_text("ALVES")}
extra_boxes = [find_text("ATENÇÃO: Na seção"), find_text("ATENÇÃO: Inclua")]   # reaproveitados como legendas
qr_ph = next(x for x in shapes if x.has_text_frame and "QR CODE" in text_of(x))
logo_unidade = find_text("LOGO DA UNIDADE"); logo_unidade._element.getparent().remove(logo_unidade._element)

def pic(path, x, y, w):
    return s.shapes.add_picture(path, Emu(int(x) * PX), Emu(int(y) * PX), width=Emu(int(w) * PX))

# titulo e autores
A_PT = 16
for T_PT in (36, 34, 32):          # modelo pede 36; cai ate caber em 3 linhas
    t_lines = nlines("**" + TITULO + "**", T_PT, 932 - 2 * INS)
    if t_lines <= 3: break
t_h = t_lines * T_PT * 4 / 3 * 1.18 + 8
a_h = len(AUTORES) * A_PT * 4 / 3 * 1.2 + 2 * INS + 6
T_TOP = 246; A_TOP = T_TOP + t_h + 2; Y0 = A_TOP + a_h + 8
write(find_text("TÍTULO"), ["**" + TITULO + "**"], T_PT, PP_ALIGN.CENTER, left=76, top=T_TOP, width=932, height=t_h, gap=0)
write(find_text("1Nome"), [("**" + t + "**") if b else t for t, b in AUTORES], A_PT, PP_ALIGN.CENTER, left=67, top=A_TOP, width=932, height=a_h, gap=0)

BODY, CAP = 12.5, 12   # pt (modelo: minimo absoluto 12 pt)

# ---- faixa A: introducao (esq) | metodologia (dir)
yA = Y0
place_header(G["INTRODUÇÃO"], XL, yA); place_header(G["METODOLOGIA"], XR, yA)
y_txt = yA + HDR_H + HDR_GAP
h_intro = box_h(INTRO, BODY, COL_W)
pipe_png = os.path.join(TMP, "fig1.png"); pipe_h = fig_pipeline(pipe_png)
pic(pipe_png, XR, y_txt, COL_W)
h_mcap = box_h(METOD_CAP, CAP, COL_W)
h_metod = pipe_h + 4 + h_mcap
write(BOX["INTRODUÇÃO"], INTRO, BODY, left=XL, top=y_txt, height=h_intro)
write(BOX["METODOLOGIA"], METOD_CAP, CAP, left=XR, top=y_txt + pipe_h + 4, height=h_mcap)
yB = y_txt + max(h_intro, h_metod) + GAP

# ---- faixa B: resultados (largura total)
place_header(G["RESULTADOS E DISCUSSÕES"], XL, yB, FULL_W)
y = yB + HDR_H + HDR_GAP
sc_png = os.path.join(TMP, "fig2.png"); sc_h = fig_scorecard(sc_png)
pic(sc_png, XL, y, FULL_W); y += sc_h + 2
h_capA = box_h([RES_CAP_A], CAP, FULL_W)
write(extra_boxes[0], [RES_CAP_A], CAP, left=XL, top=y, width=FULL_W, height=h_capA); y += h_capA + 4
wall_png = os.path.join(TMP, "fig3.png"); wall_h = fig_bmc_wall(wall_png)
q88_png = os.path.join(TMP, "fig4.png"); q88_h = fig_q88(q88_png)
pic(wall_png, XL, y, COL_W); pic(q88_png, XR, y, COL_W); y += max(wall_h, q88_h) + 2
h_capB, h_capC = box_h([RES_CAP_B], CAP, COL_W), box_h([RES_CAP_C], CAP, COL_W)
write(extra_boxes[1], [RES_CAP_B], CAP, left=XL, top=y, height=h_capB)
write(BOX["RESULTADOS E DISCUSSÕES"], [RES_CAP_C] + [""] * 0, CAP, left=XR, top=y, height=h_capC)
y += max(h_capB, h_capC) + 2
yC = y + GAP

# ---- faixa C: conclusao + agradecimentos (esq) | referencias + QR (dir)
place_header(G["CONCLUSÃO"], XL, yC)
y_txt = yC + HDR_H + HDR_GAP
h_concl = box_h(CONCL, BODY, COL_W)
write(BOX["CONCLUSÃO"], CONCL, BODY, left=XL, top=y_txt, height=h_concl)
y_ag = y_txt + h_concl + GAP
place_header(G["AGRADECIMENTOS"], XL, y_ag)
h_ag = box_h(AGRAD, CAP, COL_W)
write(BOX["AGRADECIMENTOS"], AGRAD, CAP, left=XL, top=y_ag + HDR_H + HDR_GAP, height=h_ag)
left_end = y_ag + HDR_H + HDR_GAP + h_ag

place_header(G["REFERÊNCIAS"], XR, yC)
h_refs = box_h(REFS, CAP, COL_W, gap=2)
write(BOX["REFERÊNCIAS"], REFS, CAP, PP_ALIGN.LEFT, left=XR, top=y_txt, height=h_refs, gap=2)
refs_end = y_txt + h_refs

# QR 250x250 no canto inferior direito, acima do limite das logos
QR_SZ = 250
qr_y = 1600; qr_x = XR + COL_W - QR_SZ   # ao lado das logos do rodape, como no modelo; base em 1850 (margem de 60 px)
if os.path.exists(QR):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    qr_ph._element.getparent().remove(qr_ph._element)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu((qr_x - 8) * PX), Emu((qr_y - 8) * PX), Emu((QR_SZ + 16) * PX), Emu((QR_SZ + 16) * PX))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(255, 255, 255); bg.line.fill.background()
    s.shapes.add_picture(QR, Emu(qr_x * PX), Emu(qr_y * PX), width=Emu(QR_SZ * PX), height=Emu(QR_SZ * PX))
else:
    write(qr_ph, ["**COLOQUE O QR CODE DO RESUMO EXPANDIDO AQUI**"], 13, PP_ALIGN.CENTER, left=qr_x, top=qr_y, width=QR_SZ, height=QR_SZ, anchor=MSO_ANCHOR.MIDDLE)
    qr_ph.line.width = Pt(2)

# ---- checagens de layout
overflow = max(left_end, refs_end) - BOTTOM
clash = refs_end - (qr_y - 8)
print(f"titulo {T_PT} pt x{t_lines} | corpo {BODY} pt | faixas: A={yA:.0f} B={yB:.0f} C={yC:.0f} | fim esq {left_end:.0f} / refs {refs_end:.0f} / QR {qr_y} | limite {BOTTOM}")
if overflow > 0: sys.exit(f"ESTOURO: conteudo passa {overflow:.0f}px do limite")
if clash > 0: sys.exit(f"CONFLITO: referencias invadem o QR por {clash:.0f}px")

prs.save(SAIDA_PPTX)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", TMP, SAIDA_PPTX], check=True, capture_output=True)
subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "1080", "-scale-to-y", "1920",
                os.path.join(TMP, "banner_conic_pibiti.pdf"), os.path.join(TMP, "p")], check=True)
Image.open(os.path.join(TMP, "p.png")).convert("RGB").save(SAIDA_PNG, optimize=True)
shutil.rmtree(TMP)
print("ok", SAIDA_PNG, Image.open(SAIDA_PNG).size, round(os.path.getsize(SAIDA_PNG) / 1024), "KB")
