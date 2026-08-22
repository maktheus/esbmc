(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,65018,e=>{"use strict";var a=e.i(43476);function r({vertical:e=!0}){return e?(0,a.jsx)("div",{className:"flex justify-center my-1",children:(0,a.jsxs)("div",{className:"flex flex-col items-center",children:[(0,a.jsx)("div",{className:"w-0.5 h-5 bg-gray-500"}),(0,a.jsx)("div",{className:"w-0 h-0 border-l-4 border-r-4 border-t-6 border-l-transparent border-r-transparent border-t-gray-500",style:{borderTopWidth:6}})]})}):(0,a.jsxs)("div",{className:"flex items-center mx-1",children:[(0,a.jsx)("div",{className:"h-0.5 w-5 bg-gray-500"}),(0,a.jsx)("div",{className:"w-0 h-0 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent border-l-gray-500",style:{borderLeftWidth:6}})]})}function s({label:e,sub:r,color:t="gray",small:o=!1}){let n={blue:"bg-blue-900/40 border-blue-600 text-blue-200",green:"bg-green-900/40 border-green-600 text-green-200",purple:"bg-purple-900/40 border-purple-600 text-purple-200",orange:"bg-orange-900/40 border-orange-600 text-orange-200",red:"bg-red-900/40 border-red-600 text-red-200",yellow:"bg-yellow-900/40 border-yellow-600 text-yellow-200",gray:"bg-gray-800 border-gray-600 text-gray-200",cyan:"bg-cyan-900/40 border-cyan-600 text-cyan-200"};return(0,a.jsxs)("div",{className:`rounded-xl border px-4 py-3 text-center ${n[t]??n.gray} ${o?"text-xs":"text-sm"}`,children:[(0,a.jsx)("div",{className:"font-bold",children:e}),r&&(0,a.jsx)("div",{className:"text-xs opacity-70 mt-0.5",children:r})]})}function t(){return(0,a.jsxs)("div",{className:"bg-gray-900 rounded-2xl border border-gray-700 p-6 overflow-x-auto",children:[(0,a.jsx)("p",{className:"text-gray-400 text-xs uppercase tracking-widest mb-6 text-center",children:"Pipeline Completo — DDPG Cart-Pole + Verificacao ESBMC"}),(0,a.jsxs)("div",{className:"flex items-center justify-center gap-0 flex-wrap md:flex-nowrap",children:[(0,a.jsx)(s,{label:"CartPole Env",sub:"cartpole_env.py",color:"gray"}),(0,a.jsx)(r,{vertical:!1}),(0,a.jsx)(s,{label:"Treinamento DDPG",sub:"train_ddpg.py",color:"blue"}),(0,a.jsx)(r,{vertical:!1}),(0,a.jsx)(s,{label:"ddpg_actor_best.pth",sub:"Actor: 4->24->24->1",color:"purple",small:!0})]}),(0,a.jsx)(r,{}),(0,a.jsxs)("div",{className:"flex items-center justify-center gap-0 flex-wrap md:flex-nowrap",children:[(0,a.jsx)(s,{label:"Extracao de Pesos",sub:"ddpg_weight_extractor.py",color:"cyan"}),(0,a.jsx)(r,{vertical:!1}),(0,a.jsx)(s,{label:"Quantizacao Q8.8",sub:"export_quantized_weights.py",color:"cyan"}),(0,a.jsx)(r,{vertical:!1}),(0,a.jsxs)("div",{className:"flex flex-col gap-2",children:[(0,a.jsx)(s,{label:"ddpg_weights_q88.json",sub:"pesos inteiros para browser",color:"purple",small:!0}),(0,a.jsx)(s,{label:"quantization_report.json",sub:"analise de erro",color:"gray",small:!0})]})]}),(0,a.jsx)(r,{}),(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:[(0,a.jsxs)("div",{className:"border border-orange-700/50 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-orange-300 font-bold text-sm text-center",children:"Dominio 1 — Modelo (NN)"}),(0,a.jsxs)("div",{className:"flex flex-col items-center gap-2",children:[(0,a.jsx)(s,{label:"Neuronios Mortos + Saturacao",sub:"verify_ddpg_dead_neurons.py + verify_ddpg_saturation.py",color:"orange",small:!0}),(0,a.jsx)(r,{}),(0,a.jsx)(s,{label:"48 harnesses C (L1+L2)",sub:"assert(h==0) / assert(pre<=0)",color:"gray",small:!0}),(0,a.jsx)(r,{}),(0,a.jsx)(s,{label:"ESBMC / Boolector",sub:"30s timeout por neuronio",color:"yellow",small:!0}),(0,a.jsx)(r,{}),(0,a.jsx)(s,{label:"0/48 mortos, 0 saturados",sub:"RESPONSIVE",color:"green",small:!0})]})]}),(0,a.jsxs)("div",{className:"border border-blue-700/50 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-blue-300 font-bold text-sm text-center",children:"Dominio 2 — Controlador (Malha Fechada)"}),(0,a.jsxs)("div",{className:"flex flex-col items-center gap-2",children:[(0,a.jsx)(s,{label:"Propriedades A, B, C",sub:"verify_ddpg_closed_loop.py",color:"blue",small:!0}),(0,a.jsx)(r,{}),(0,a.jsx)(s,{label:"Aritmetica de Intervalo",sub:"bounds pre-ativacao -> __ESBMC_assume",color:"gray",small:!0}),(0,a.jsx)(r,{}),(0,a.jsx)(s,{label:"ESBMC + Tanh Piecewise",sub:"Dinamica linearizada 1-step",color:"yellow",small:!0}),(0,a.jsx)(r,{}),(0,a.jsxs)("div",{className:"flex flex-col gap-1 w-full",children:[(0,a.jsx)(s,{label:"Prop C: SUCCESSFUL",sub:"|F| <= 10 N provado formalmente",color:"green",small:!0}),(0,a.jsx)(s,{label:"Prop B: FAILED",sub:"contraexemplo encontrado",color:"red",small:!0}),(0,a.jsx)(s,{label:"Prop A: TIMEOUT",sub:"120s insuficiente",color:"yellow",small:!0})]})]})]})]}),(0,a.jsx)(r,{}),(0,a.jsx)("div",{className:"flex justify-center",children:(0,a.jsx)(s,{label:"WebApp — Q8.8 em tempo real + contraexemplos visuais",sub:"Next.js / TypeScript / Canvas",color:"blue"})})]})}function o({children:e,title:r,lang:s}){return(0,a.jsxs)("div",{className:"space-y-0",children:[r&&(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 border-b-0 rounded-t-xl px-4 py-2 flex items-center gap-2",children:[(0,a.jsx)("span",{className:"text-gray-400 text-xs font-mono",children:r}),s&&(0,a.jsx)("span",{className:"text-gray-500 text-xs bg-gray-700 px-1.5 py-0.5 rounded",children:s})]}),(0,a.jsx)("pre",{className:`bg-gray-950 border border-gray-700 p-4 overflow-x-auto text-xs font-mono text-gray-300 leading-relaxed ${r?"rounded-b-xl":"rounded-xl"}`,children:(0,a.jsx)("code",{children:e.trim()})})]})}function n({n:e,title:r,color:s="blue",children:t}){let o={blue:"bg-blue-600",green:"bg-green-600",purple:"bg-purple-600",orange:"bg-orange-600",red:"bg-red-600",cyan:"bg-cyan-600"};return(0,a.jsxs)("div",{className:"space-y-4",children:[(0,a.jsxs)("div",{className:"flex items-center gap-3",children:[(0,a.jsx)("span",{className:`${o[s]??o.blue} text-white font-bold text-sm w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0`,children:e}),(0,a.jsx)("h2",{className:"text-xl font-bold text-white",children:r})]}),(0,a.jsx)("div",{className:"pl-10 space-y-4",children:t})]})}function i({label:e,color:r="gray"}){let s={green:"bg-green-900 text-green-300",red:"bg-red-900 text-red-300",yellow:"bg-yellow-900 text-yellow-300",blue:"bg-blue-900 text-blue-300",gray:"bg-gray-700 text-gray-300",orange:"bg-orange-900 text-orange-300"};return(0,a.jsx)("span",{className:`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold ${s[r]??s.gray}`,children:e})}e.s(["default",0,function(){return(0,a.jsxs)("div",{className:"space-y-12 pb-12",children:[(0,a.jsxs)("div",{children:[(0,a.jsx)("h1",{className:"text-3xl font-bold text-white mb-2",children:"Metodologia Tecnica"}),(0,a.jsx)("p",{className:"text-gray-400",children:"Pipeline completo de treinamento, quantizacao Q8.8 e verificacao formal do controlador DDPG continuo para Cart-Pole com ESBMC. Todos os codigos-fonte reais do projeto sao apresentados abaixo. O browser executa a mesma aritmetica inteira verificada pelo ESBMC."})]}),(0,a.jsx)(t,{}),(0,a.jsxs)(n,{n:1,title:"Sistema Fisico — Cart-Pole",color:"blue",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"Pendulo invertido sobre carro movel. Estado = vetor 4D continuo. Controlador DDPG aplica forca continua F in [-10, +10] N a cada 20 ms para manter equilibrio. Equacoes de movimento de Barto, Sutton & Anderson (1983), integracao de Euler."}),(0,a.jsx)("div",{className:"grid grid-cols-2 md:grid-cols-4 gap-3",children:[{v:"x",desc:"Posicao do carro",range:"-2.4 a 2.4 m",q88:"[-614, 614]"},{v:"x_dot",desc:"Velocidade do carro",range:"-5 a 5 m/s",q88:"[-1280, 1280]"},{v:"theta",desc:"Angulo do pendulo",range:"-12 a 12 graus",q88:"[-53, 53]"},{v:"theta_dot",desc:"Vel. angular",range:"-5 a 5 rad/s",q88:"[-1280, 1280]"}].map(({v:e,desc:r,range:s,q88:t})=>(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-xl p-3",children:[(0,a.jsx)("p",{className:"font-mono text-blue-400 text-lg font-bold",children:e}),(0,a.jsx)("p",{className:"text-gray-300 text-xs mt-1",children:r}),(0,a.jsx)("p",{className:"text-gray-500 text-xs font-mono mt-1",children:s}),(0,a.jsxs)("p",{className:"text-cyan-400 text-xs font-mono mt-0.5",children:["Q8.8: ",t]})]},e))}),(0,a.jsx)(o,{title:"lib/physics.ts — Motor de fisica no browser",lang:"TypeScript",children:`
export const GRAVITY  = 9.8;
export const M_CART   = 1.0;
export const M_POLE   = 0.1;
export const M_TOTAL  = M_CART + M_POLE;
export const L        = 0.5;
export const ML       = M_POLE * L;
export const DT       = 0.02;
export const FORCE_MAX = 10.0;

export const X_LIMIT     = 2.4;
export const THETA_LIMIT = 12.0 * Math.PI / 180;  // 0.2094 rad

export function physicsStep(s: CartPoleState, force: number): CartPoleState {
  const F = Math.max(-FORCE_MAX, Math.min(FORCE_MAX, force));
  const cosT = Math.cos(s.theta);
  const sinT = Math.sin(s.theta);

  const temp   = (F + ML * s.theta_dot ** 2 * sinT) / M_TOTAL;
  const th_acc = (GRAVITY * sinT - cosT * temp) /
                 (L * (4 / 3 - M_POLE * cosT ** 2 / M_TOTAL));
  const x_acc  = temp - ML * th_acc * cosT / M_TOTAL;

  return {
    x:         s.x         + DT * s.x_dot,
    x_dot:     s.x_dot     + DT * x_acc,
    theta:     s.theta     + DT * s.theta_dot,
    theta_dot: s.theta_dot + DT * th_acc,
  };
}`})]}),(0,a.jsxs)(n,{n:2,title:"Treinamento — DDPG (Continuo)",color:"purple",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"Deep Deterministic Policy Gradient: actor-critic off-policy para espacos de acao continuos. O Actor mapeia estado para forca via tanh x 10. O Critic avalia Q(s,a). Exploracao via ruido Ornstein-Uhlenbeck."}),(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-4",children:[(0,a.jsxs)("div",{className:"space-y-3",children:[(0,a.jsx)("h3",{className:"text-white font-semibold text-sm",children:"Arquitetura do Actor"}),(0,a.jsx)("div",{className:"flex items-center gap-2 flex-wrap",children:[{label:"Input\n4 vars",bg:"bg-blue-800"},{label:"Hidden 1\n24 ReLU",bg:"bg-purple-800"},{label:"Hidden 2\n24 ReLU",bg:"bg-purple-800"},{label:"Output\n1 Tanh x10",bg:"bg-green-800"}].map(({label:e,bg:r})=>(0,a.jsx)("div",{className:`${r} rounded-lg px-3 py-2 text-center text-xs text-white font-mono whitespace-pre-line`,children:e},e))}),(0,a.jsx)(o,{title:"ddpg_agent.py — Rede Actor (trecho)",lang:"Python",children:`
class DDPGActor(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * FORCE_MAX
        # tanh garante saida em [-1, 1]
        # multiplicado por 10 -> F in [-10, +10] N`})]}),(0,a.jsxs)("div",{className:"space-y-3",children:[(0,a.jsx)("h3",{className:"text-white font-semibold text-sm",children:"Extracao de Pesos"}),(0,a.jsx)(o,{title:"ddpg_weight_extractor.py — Codigo real",lang:"Python",children:`
import torch

def extract_ddpg_weights(pth_path: str) -> dict:
    sd = torch.load(pth_path,
                     map_location="cpu",
                     weights_only=True)

    return {
        "w1":    sd["net.0.weight"].tolist(),   # [24, 4]
        "b1":    sd["net.0.bias"].tolist(),      # [24]
        "w2":    sd["net.2.weight"].tolist(),   # [24, 24]
        "b2":    sd["net.2.bias"].tolist(),      # [24]
        "w_out": sd["net.4.weight"].tolist(),   # [1, 24]
        "b_out": sd["net.4.bias"].tolist(),      # [1]
    }`}),(0,a.jsx)("p",{className:"text-gray-400 text-xs",children:"Total de parametros: 4x24 + 24 + 24x24 + 24 + 24x1 + 1 = 721 pesos."})]})]})]}),(0,a.jsxs)(n,{n:3,title:"Quantizacao Q8.8 — Zero Gap de Fidelidade",color:"cyan",children:[(0,a.jsxs)("div",{className:"bg-cyan-900/20 border border-cyan-700 rounded-xl p-4 text-sm mb-4",children:[(0,a.jsx)("h3",{className:"text-cyan-300 font-semibold mb-2",children:"Principio Fundamental"}),(0,a.jsxs)("p",{className:"text-gray-300",children:["Em vez de verificar Q8.8 com ESBMC mas rodar float32 no browser (com gap de fidelidade), executamos a ",(0,a.jsx)("strong",{className:"text-cyan-300",children:"mesma aritmetica inteira Q8.8"})," no browser. Contraexemplos do ESBMC reproduzem ",(0,a.jsx)("em",{children:"exatamente"})," na simulacao."]})]}),(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-4",children:[(0,a.jsx)(o,{title:"export_quantized_weights.py — Quantizacao (codigo real)",lang:"Python",children:`
SCALE = 256

def q(v: float) -> int:
    return int(round(v * SCALE))

def cdiv(a: int, b: int) -> int:
    """Divisao C: trunca para zero (nao floor!)"""
    return int(a / b)

def tanh_q88(z: int) -> int:
    """Tanh piecewise linear em Q8.8 — 5 segmentos.
    Mesma funcao no C harness e no TypeScript."""
    z_abs = abs(z)
    if z_abs <= 64:
        t = cdiv(z_abs * 252, 256)
    elif z_abs <= 192:
        t = 62 + cdiv((z_abs - 64) * 200, 256)
    elif z_abs <= 384:
        t = 162 + cdiv((z_abs - 192) * 92, 256)
    elif z_abs <= 768:
        t = 231 + cdiv((z_abs - 384) * 16, 256)
    else:
        t = 255
    return t if z >= 0 else -t

def forward_q88(state_q, qw):
    h1 = []
    for i in range(len(qw["b1"])):
        pre = qw["b1"][i]
        for j in range(len(state_q)):
            pre += cdiv(state_q[j] * qw["w1"][i][j], SCALE)
        h1.append(max(0, pre))  # ReLU

    h2 = []
    for i in range(len(qw["b2"])):
        pre = qw["b2"][i]
        for j in range(len(h1)):
            pre += cdiv(h1[j] * qw["w2"][i][j], SCALE)
        h2.append(max(0, pre))

    z = qw["b_out"][0]
    for j in range(len(h2)):
        z += cdiv(h2[j] * qw["w_out"][0][j], SCALE)

    return cdiv(tanh_q88(z) * 10 * SCALE, SCALE)`}),(0,a.jsx)(o,{title:"lib/quantized-controller.ts — Mesmo Q8.8 no browser (codigo real)",lang:"TypeScript",children:`
const SCALE = 256;

function cdiv(a: number, b: number): number {
  return Math.trunc(a / b);  // mesmo que / em C
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

export function tanhQ88(z: number): number {
  const z_abs = Math.abs(z);
  let t: number;
  if (z_abs <= 64)
    t = cdiv(z_abs * 252, 256);
  else if (z_abs <= 192)
    t = 62 + cdiv((z_abs - 64) * 200, 256);
  else if (z_abs <= 384)
    t = 162 + cdiv((z_abs - 192) * 92, 256);
  else if (z_abs <= 768)
    t = 231 + cdiv((z_abs - 384) * 16, 256);
  else
    t = 255;
  return z >= 0 ? t : -t;
}

export function getForceQ88(
  state: [number, number, number, number],
  w: QuantizedWeights
): number {
  const h1: number[] = [];
  for (let i = 0; i < w.b1.length; i++) {
    let pre = w.b1[i];
    for (let j = 0; j < 4; j++)
      pre += cdiv(state[j] * w.w1[i][j], SCALE);
    h1.push(relu(pre));
  }

  const h2: number[] = [];
  for (let i = 0; i < w.b2.length; i++) {
    let pre = w.b2[i];
    for (let j = 0; j < h1.length; j++)
      pre += cdiv(h1[j] * w.w2[i][j], SCALE);
    h2.push(relu(pre));
  }

  let z = w.b_out[0];
  for (let j = 0; j < h2.length; j++)
    z += cdiv(h2[j] * w.w_out[0][j], SCALE);

  const tanh_z = tanhQ88(z);
  return cdiv(tanh_z * 10 * SCALE, SCALE);
}`})]}),(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-xl p-4 text-sm",children:[(0,a.jsx)("h3",{className:"text-white font-semibold mb-2",children:"Analise de Erro — Float32 vs Q8.8 (10.000 amostras)"}),(0,a.jsx)("div",{className:"grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs",children:[["Erro max","7.76 N","text-red-400"],["Erro medio","0.08 N","text-green-400"],["Erro p95","0.23 N","text-yellow-400"],["Erro p99","1.09 N","text-orange-400"]].map(([e,r,s])=>(0,a.jsxs)("div",{className:"bg-gray-900 rounded-lg p-3 text-center",children:[(0,a.jsx)("p",{className:"text-gray-500 text-xs",children:e}),(0,a.jsx)("p",{className:`text-lg font-bold ${s}`,children:r})]},e))}),(0,a.jsx)("p",{className:"text-gray-400 text-xs mt-3",children:"Erro maximo ocorre em estados extremos (bordas do dominio). Para 95% dos estados, erro menor que 0.23 N — negligivel para controle. A quantizacao e aceita porque o ESBMC verifica exatamente o que executa."})]}),(0,a.jsxs)("div",{className:"bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 text-xs text-gray-300",children:[(0,a.jsx)("p",{className:"text-yellow-300 font-semibold",children:"Soundness: Python // vs C /"}),(0,a.jsxs)("p",{className:"mt-1",children:["Python usa divisao de piso: ",(0,a.jsx)("code",{className:"text-yellow-200",children:"-300 // 256 = -2"}),". C trunca para zero: ",(0,a.jsx)("code",{className:"text-yellow-200",children:"-300 / 256 = -1"}),". Para bounds corretos, usa-se ",(0,a.jsx)("code",{className:"text-yellow-200",children:"cdiv(a,b) = int(a/b)"}),"em todas as computacoes de intervalo — replica exata da semantica C."]})]})]}),(0,a.jsxs)(n,{n:4,title:"Geracao de Harness C para ESBMC",color:"orange",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"Python gera harnesses C automaticamente com todos os 721 pesos expandidos inline. Sem loops C — o ESBMC precisa de codigo straight-line para eficiencia. Entradas simbolicas via nondet_int(). Bounds de pre-ativacao via __ESBMC_assume."}),(0,a.jsxs)("div",{className:"grid grid-cols-1 gap-4",children:[(0,a.jsx)(o,{title:"verify_ddpg_dead_neurons.py — Harness para neuronio morto (codigo real)",lang:"Python",children:`
def harness_layer1(idx, qw1, qb1, pre_bound):
    i = idx
    w = qw1[i]
    return f"""
void __ESBMC_assume(_Bool c);
void __ESBMC_assert(_Bool c, const char *m);
int nondet_int(void);

int main(void) {{
    int x1 = nondet_int();  // entradas simbolicas
    int x2 = nondet_int();
    int x3 = nondet_int();
    int x4 = nondet_int();

    // Restricoes de dominio (Q8.8)
    __ESBMC_assume(x1 >= -614  && x1 <= 614);   // x in [-2.4, 2.4]m
    __ESBMC_assume(x2 >= -1280 && x2 <= 1280);  // x_dot in [-5, 5]m/s
    __ESBMC_assume(x3 >= -53   && x3 <= 53);    // theta in [-12, 12]deg
    __ESBMC_assume(x4 >= -1280 && x4 <= 1280);  // theta_dot in [-5, 5]

    // Forward pass Q8.8 (pesos hardcoded)
    int pre = (x1*({w[0]}))/256
            + (x2*({w[1]}))/256
            + (x3*({w[2]}))/256
            + (x4*({w[3]}))/256
            + ({qb1[i]});

    __ESBMC_assume(pre >= -{pre_bound} && pre <= {pre_bound});
    int h = pre > 0 ? pre : 0;  // ReLU

    // Se ESBMC FAILED: existe entrada que ativa -> VIVO
    // Se ESBMC SUCCESS: nunca ativa -> MORTO
    __ESBMC_assert(h == 0, "neuronio L1[{i}] e sempre morto?");
    return 0;
}}
"""`}),(0,a.jsx)(o,{title:"verify_ddpg_dead_neurons.py — Aritmetica de intervalo (codigo real)",lang:"Python",children:`
def interval_propagate_layer(qw, qb, in_lo, in_hi):
    """Propaga bounds atraves de uma camada linear Q8.8.
    Resultado: [lo_pre, hi_pre] para cada neuronio.
    Usa cdiv (truncamento C) para soundness."""
    lo_pre, hi_pre = [], []
    for i in range(len(qb)):
        lo, hi = qb[i], qb[i]
        for k in range(len(in_lo)):
            w = qw[i][k]
            if w >= 0:
                lo += c_div(in_lo[k] * w, SCALE)
                hi += c_div(in_hi[k] * w, SCALE)
            else:
                lo += c_div(in_hi[k] * w, SCALE)
                hi += c_div(in_lo[k] * w, SCALE)
        lo_pre.append(lo)
        hi_pre.append(hi)
    return lo_pre, hi_pre

def relu_bounds(lo_pre, hi_pre):
    return [max(0, lo) for lo in lo_pre],\\
           [max(0, hi) for hi in hi_pre]`}),(0,a.jsx)(o,{title:"verify_ddpg_closed_loop.py — Expansao inline do controlador (codigo real)",lang:"Python",children:`
def generate_controller_body(qw1, qb1, qw2, qb2, qw_out, qb_out,
                             lo_pre1, hi_pre1, lo_pre2, hi_pre2):
    lines = []
    # Camada 1: 24 neuronios, 4 entradas cada
    for i in range(24):
        w0, w1, w2, w3 = qw1[i]
        b = qb1[i]
        lines.append(
            f"    int pre1_{i} = (x*({w0}))/256 + (xd*({w1}))/256"
            f" + (th*({w2}))/256 + (thd*({w3}))/256 + ({b});"
        )
        lines.append(
            f"    __ESBMC_assume(pre1_{i} >= {lo_pre1[i]}"
            f" && pre1_{i} <= {hi_pre1[i]});"
        )
        lines.append(f"    int h1_{i} = pre1_{i} > 0 ? pre1_{i} : 0;")

    # Camada 2: 24 neuronios, 24 entradas cada
    for j in range(24):
        terms = " + ".join(
            f"(h1_{k}*({qw2[j][k]}))/256" for k in range(24)
        )
        lines.append(f"    int pre2_{j} = {terms} + ({qb2[j]});")
        lines.append(
            f"    __ESBMC_assume(pre2_{j} >= {lo_pre2[j]}"
            f" && pre2_{j} <= {hi_pre2[j]});"
        )
        lines.append(f"    int h2_{j} = pre2_{j} > 0 ? pre2_{j} : 0;")

    # Saida: z = sum(h2 * w_out) + b_out
    out_terms = " + ".join(
        f"(h2_{k}*({qw_out[0][k]}))/256" for k in range(24)
    )
    lines.append(f"    int z = {out_terms} + ({qb_out[0]});")
    return "\\n".join(lines)`})]})]}),(0,a.jsxs)(n,{n:5,title:"Propriedades em Malha Fechada — Codigo Real",color:"blue",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"Tres propriedades verificadas sobre o sistema completo (controlador DDPG + dinamica Cart-Pole). Cada harness inclui o controlador expandido + a propriedade especifica."}),(0,a.jsx)(o,{title:"verify_ddpg_closed_loop.py — Property A: Direcao (codigo real)",lang:"Python",children:`
DANGER_TH = int(0.10 * 256)  # 25 em Q8.8 = 5.7 graus

def harness_prop_a_right(ctrl_body):
    return f"""
/* Property A: theta > 5.7 graus, theta_dot >= 0 -> F > 0
 * Usa monotonicidade: z > 0 <=> tanh(z) > 0 <=> F > 0
 * NAO precisa aproximar tanh! */

int main(void) {{
    int x   = nondet_int();
    int xd  = nondet_int();
    int th  = nondet_int();
    int thd = nondet_int();

    __ESBMC_assume(x   >= -614  && x   <= 614);
    __ESBMC_assume(xd  >= -1280 && xd  <= 1280);
    __ESBMC_assume(th  >  25    && th  <= 53);     // theta > limiar
    __ESBMC_assume(thd >= 0     && thd <= 1280);   // theta_dot >= 0

    {ctrl_body}  // 24+24 neuronios expandidos

    /* z > 0 => tanh(z) > 0 => F > 0 (monotonicidade) */
    __ESBMC_assert(z > 0,
        "PropA-right: controlador nao aplica forca positiva!");
    return 0;
}}
"""`}),(0,a.jsx)(o,{title:"verify_ddpg_closed_loop.py — Property B: Seguranca 1-step (codigo real)",lang:"Python",children:`
TANH_APPROX_C = """
    /* Tanh piecewise linear — MESMA do browser TypeScript */
    int z_abs = z >= 0 ? z : -z;
    int tanh_abs;
    if (z_abs <= 64)        tanh_abs = (z_abs * 252) / 256;
    else if (z_abs <= 192)  tanh_abs = 62 + ((z_abs - 64) * 200) / 256;
    else if (z_abs <= 384)  tanh_abs = 162 + ((z_abs - 192) * 92) / 256;
    else if (z_abs <= 768)  tanh_abs = 231 + ((z_abs - 384) * 16) / 256;
    else                    tanh_abs = 255;
    int tanh_z = z >= 0 ? tanh_abs : -tanh_abs;
    int F_Q = (tanh_z * 10 * 256) / 256;
"""

def harness_prop_b(ctrl_body):
    return f"""
/* Property B: estado seguro -> theta seguro apos 1 passo
 * Dinamica linearizada: sin(theta) ~ theta, cos(theta) ~ 1
 * th_acc = (4040 * th - 375 * F_Q) / 256
 * th_new = th + (5 * thd) / 256 */

int main(void) {{
    int x = nondet_int(), xd = nondet_int();
    int th = nondet_int(), thd = nondet_int();

    __ESBMC_assume(x >= -614 && x <= 614);
    __ESBMC_assume(xd >= -1280 && xd <= 1280);
    __ESBMC_assume(th >= -53 && th <= 53);        // estado seguro
    __ESBMC_assume(thd >= -1280 && thd <= 1280);

    {ctrl_body}       // forward pass expandido
    {TANH_APPROX_C}   // tanh + F_Q

    /* Dinamica linearizada Q8.8 */
    int th_acc  = (4040 * th - 375 * F_Q) / 256;
    int th_new  = th  + (5 * thd) / 256;
    int thd_new = thd + (5 * th_acc) / 256;

    __ESBMC_assert(th_new >= -53 && th_new <= 53,
        "PropB: theta sai da regiao segura apos 1 passo!");
    return 0;
}}
"""`}),(0,a.jsx)(o,{title:"verify_ddpg_closed_loop.py — Property C: Bounds de forca (codigo real)",lang:"Python",children:`
def harness_prop_c(ctrl_body):
    return f"""
/* Property C: |F| <= 10 N para todo estado
 * Deve ser SUCCESSFUL (tanh garante |output| <= 1) */

int main(void) {{
    int x = nondet_int(), xd = nondet_int();
    int th = nondet_int(), thd = nondet_int();

    __ESBMC_assume(x >= -614 && x <= 614);
    __ESBMC_assume(xd >= -1280 && xd <= 1280);
    __ESBMC_assume(th >= -53 && th <= 53);
    __ESBMC_assume(thd >= -1280 && thd <= 1280);

    {ctrl_body}
    {TANH_APPROX_C}

    /* F_Q em Q8.8: 10N = 2560, -10N = -2560 */
    __ESBMC_assert(F_Q >= -2560 && F_Q <= 2560,
        "PropC: forca excede limites [-10, +10] N!");
    return 0;
}}
"""`}),(0,a.jsx)(o,{title:"verify_ddpg_closed_loop.py — Runner ESBMC (codigo real)",lang:"Python",children:`
ESBMC = "esbmc-6.8.0/esbmc"

def run_esbmc(c_file, timeout=120):
    try:
        r = subprocess.run(
            [ESBMC, c_file,
             "--no-unwinding-assertions",
             "--boolector"],
            capture_output=True, text=True,
            timeout=timeout,
        )
        out = r.stdout + r.stderr

        if "VERIFICATION SUCCESSFUL" in out:
            return True, "", out
        elif "VERIFICATION FAILED" in out:
            # Extrair contraexemplo
            ce_parts = []
            for name in ["x", "xd", "th", "thd"]:
                m = re.search(rf'\\b{name}\\s*=\\s*(-?\\d+)', out)
                if m:
                    val = int(m.group(1))
                    ce_parts.append(f"{name}={val/256:.4f}")
            return False, "  ".join(ce_parts), out
        else:
            return None, "resultado desconhecido", out
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", ""`})]}),(0,a.jsx)(n,{n:6,title:"Resultados da Verificacao",color:"green",children:(0,a.jsxs)("div",{className:"space-y-4",children:[(0,a.jsxs)("div",{className:"bg-orange-900/10 border border-orange-700/50 rounded-xl p-4 space-y-3",children:[(0,a.jsx)("h3",{className:"text-orange-300 font-semibold",children:"Dominio 1 — Modelo (Rede Neural)"}),(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-3",children:[(0,a.jsxs)("div",{className:"space-y-2",children:[(0,a.jsxs)("div",{className:"flex items-center gap-2",children:[(0,a.jsx)(i,{label:"Neuronios Mortos",color:"orange"}),(0,a.jsx)(i,{label:"0/48 mortos",color:"green"})]}),(0,a.jsx)("p",{className:"text-gray-400 text-xs",children:"48 harnesses (24 L1 + 24 L2). Todos FAILED = todos os neuronios sao VIVOS. Nenhum candidato a poda."})]}),(0,a.jsxs)("div",{className:"space-y-2",children:[(0,a.jsxs)("div",{className:"flex items-center gap-2",children:[(0,a.jsx)(i,{label:"Saturacao",color:"orange"}),(0,a.jsx)(i,{label:"0 saturados + RESPONSIVE",color:"green"})]}),(0,a.jsxs)("p",{className:"text-gray-400 text-xs",children:["Nenhum neuronio sempre ativo. Saida z ",">"," 0 e z ","<"," 0 ambas FAILED — controlador produz forca positiva e negativa."]})]})]})]}),(0,a.jsxs)("div",{className:"bg-blue-900/10 border border-blue-700/50 rounded-xl p-4 space-y-4",children:[(0,a.jsx)("h3",{className:"text-blue-300 font-semibold",children:"Dominio 2 — Controlador (Malha Fechada)"}),(0,a.jsxs)("div",{className:"space-y-2",children:[(0,a.jsxs)("div",{className:"flex items-center gap-2",children:[(0,a.jsx)(i,{label:"Property A — Direcao",color:"yellow"}),(0,a.jsx)(i,{label:"TIMEOUT (120s)",color:"yellow"})]}),(0,a.jsxs)("p",{className:"text-gray-400 text-xs",children:["Espaco de estados muito grande para Boolector resolver em 120s. A monotonicidade de tanh simplifica (verifica z ",">"," 0 em vez de F ",">"," 0), mas o forward pass completo (48 neuronios) ainda excede o timeout."]})]}),(0,a.jsxs)("div",{className:"space-y-2",children:[(0,a.jsxs)("div",{className:"flex items-center gap-2",children:[(0,a.jsx)(i,{label:"Property B — Seguranca 1-step",color:"red"}),(0,a.jsx)(i,{label:"FAILED — contraexemplo",color:"red"})]}),(0,a.jsx)("div",{className:"bg-red-950/60 border border-red-800 rounded-lg p-3 font-mono text-xs text-red-200",children:"x = -0.7539 m | x_dot = -3.9219 m/s | theta = -0.1836 rad (-10.5 graus) | theta_dot = -1.5234 rad/s"}),(0,a.jsx)("p",{className:"text-gray-400 text-xs",children:"Estado no limite da regiao segura com velocidade angular significativa. Mesmo com forca maxima, a dinamica do pendulo nao permite manter theta dentro de [-12, 12] graus em 1 passo. Este contraexemplo e injetavel na simulacao em tempo real e reproduz exatamente no controlador Q8.8."})]}),(0,a.jsxs)("div",{className:"space-y-2",children:[(0,a.jsxs)("div",{className:"flex items-center gap-2",children:[(0,a.jsx)(i,{label:"Property C — Bounds de Forca",color:"green"}),(0,a.jsx)(i,{label:"SUCCESSFUL — prova formal",color:"green"})]}),(0,a.jsxs)("p",{className:"text-gray-400 text-xs",children:["ESBMC provou que |F| ","<="," 10 N para todo estado no dominio. A funcao tanh piecewise garante |output| ","<="," 255 em Q8.8, e (255 * 10 * 256) / 256 = 2550 ","<"," 2560. Propriedade satisfeita formalmente."]})]})]})]})}),(0,a.jsxs)(n,{n:7,title:"Tratamento de Tanh — Abordagem Hibrida",color:"red",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"A funcao tanh e nao-linear e nao pode ser representada diretamente em SMT. Usamos duas abordagens conforme a propriedade:"}),(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-4",children:[(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-blue-400 font-bold text-sm",children:"Property A — Monotonicidade"}),(0,a.jsxs)("p",{className:"text-gray-300 text-xs",children:["Tanh e estritamente monotonica: z ",">"," 0 se e somente se tanh(z) ",">"," 0 se e somente se F ",">"," 0. Basta verificar o sinal do valor pre-tanh z. Nao precisa computar tanh no harness C."]}),(0,a.jsx)(o,{children:`/* Property A: so verifica sinal de z */
// ... forward pass ate z ...
__ESBMC_assert(z > 0, "direcao");
// NAO precisa de tanh!`})]}),(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-orange-400 font-bold text-sm",children:"Properties B, C — Tanh Piecewise"}),(0,a.jsx)("p",{className:"text-gray-300 text-xs",children:"Precisa do valor numerico de F para dinamica (Prop B) e bounds (Prop C). Usa os mesmos 5 segmentos lineares implementados no browser TypeScript e no harness C. Mesma funcao, tres linguagens, resultado identico."}),(0,a.jsx)(o,{children:`/* Tanh piecewise em C (mesmo do TypeScript) */
if (z_abs <= 64)   t = (z_abs * 252) / 256;
else if (z_abs <= 192) t = 62 + ...;
else if (z_abs <= 384) t = 162 + ...;
else if (z_abs <= 768) t = 231 + ...;
else t = 255;
int F_Q = (tanh_z * 10 * 256) / 256;`})]})]})]}),(0,a.jsxs)(n,{n:8,title:"Arquitetura SOLID do WebApp",color:"purple",children:[(0,a.jsx)("p",{className:"text-gray-300 text-sm",children:"O webapp segue principios SOLID para separacao de responsabilidades e extensibilidade."}),(0,a.jsx)(o,{title:"lib/types.ts — Interface IController (LSP: Float e Q8.8 intercambiaveis)",lang:"TypeScript",children:`
export interface IController {
  getForce(state: [number, number, number, number]): number;
  readonly name: string;
  readonly isVerified: boolean;
}

// FloatDDPGController implements IController { isVerified = false }
// QuantizedDDPGController implements IController { isVerified = true }
// A simulacao nao sabe qual controlador esta rodando`}),(0,a.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-3 text-xs",children:[["SRP","Um modulo por responsabilidade","physics.ts = dinamica, quantized-controller.ts = inferencia, verification-data.ts = dados ESBMC"],["OCP","Aberto para extensao, fechado para modificacao","IController permite adicionar novos controladores (Q16.16, PID) sem alterar simulacao"],["LSP","Substituicao de Liskov","Float32 e Q8.8 sao intercambiaveis via IController. Simulacao funciona com qualquer um"],["ISP","Interfaces segregadas","NeuronVerification, ClosedLoopProperty, Counterexample — interfaces pequenas e focadas"]].map(([e,r,s])=>(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-lg p-3",children:[(0,a.jsx)("p",{className:"text-purple-400 font-bold",children:e}),(0,a.jsx)("p",{className:"text-gray-300 mt-1",children:r}),(0,a.jsx)("p",{className:"text-gray-500 mt-1",children:s})]},e))})]}),(0,a.jsxs)(n,{n:9,title:"Interpretacao dos Resultados ESBMC",color:"green",children:[(0,a.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-3 gap-4",children:[(0,a.jsxs)("div",{className:"bg-green-900/20 border border-green-700 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-green-400 font-bold text-sm",children:"VERIFICATION FAILED"}),(0,a.jsx)("p",{className:"text-gray-300 text-xs",children:"Solver encontrou entrada que viola a assercao. Para neuronios mortos: neuronio VIVO (existe entrada que o ativa). Para propriedades: contraexemplo reproduzivel no browser."})]}),(0,a.jsxs)("div",{className:"bg-blue-900/20 border border-blue-700 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-blue-400 font-bold text-sm",children:"VERIFICATION SUCCESSFUL"}),(0,a.jsxs)("p",{className:"text-gray-300 text-xs",children:["Prova formal: nenhuma entrada no dominio inteiro viola a propriedade. Ex: Property C — |F| ","<="," 10 N provado matematicamente para todo estado."]})]}),(0,a.jsxs)("div",{className:"bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 space-y-2",children:[(0,a.jsx)("p",{className:"text-yellow-400 font-bold text-sm",children:"TIMEOUT"}),(0,a.jsx)("p",{className:"text-gray-300 text-xs",children:"Solver nao concluiu em 120s. Resultado inconclusivo — a propriedade pode ser verdadeira ou falsa. Potencial: aumentar timeout ou usar abstractions."})]})]}),(0,a.jsxs)("div",{className:"bg-gray-800 border border-gray-700 rounded-xl p-4 text-sm text-gray-300",children:[(0,a.jsx)("p",{className:"font-semibold text-white mb-2",children:"Dominio verificado (Q8.8, scale=256)"}),(0,a.jsx)("div",{className:"grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs",children:[["x","-614 a +614","-2.4 a 2.4 m"],["x_dot","-1280 a +1280","-5 a 5 m/s"],["theta","-53 a +53","-12 a 12 graus"],["theta_dot","-1280 a +1280","-5 a 5 rad/s"]].map(([e,r,s])=>(0,a.jsxs)("div",{className:"bg-gray-900 rounded-lg p-2",children:[(0,a.jsx)("p",{className:"text-blue-400 font-bold",children:e}),(0,a.jsx)("p",{className:"text-gray-300",children:r}),(0,a.jsx)("p",{className:"text-gray-500",children:s})]},e))}),(0,a.jsx)("p",{className:"text-gray-500 text-xs mt-2",children:"Solver: Boolector (SMT bit-vector). Timeout: 30s por neuronio, 120s por propriedade. ESBMC v6.8.0 com flag --no-unwinding-assertions --boolector."})]})]})]})}])}]);