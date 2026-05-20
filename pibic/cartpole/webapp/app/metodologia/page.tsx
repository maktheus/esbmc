'use client';

/* ── Diagrama de fluxo ─────────────────────────────────────────────────────── */
function Arrow({ vertical = true }: { vertical?: boolean }) {
  if (vertical) {
    return (
      <div className="flex justify-center my-1">
        <div className="flex flex-col items-center">
          <div className="w-0.5 h-5 bg-gray-500" />
          <div className="w-0 h-0 border-l-4 border-r-4 border-t-6 border-l-transparent border-r-transparent border-t-gray-500"
               style={{ borderTopWidth: 6 }} />
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-center mx-1">
      <div className="h-0.5 w-5 bg-gray-500" />
      <div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent border-l-gray-500"
           style={{ borderLeftWidth: 6 }} />
    </div>
  );
}

function FlowBox({
  label, sub, color = 'gray', icon, small = false,
}: {
  label: string; sub?: string; color?: string; icon?: string; small?: boolean;
}) {
  const colors: Record<string, string> = {
    blue:   'bg-blue-900/40 border-blue-600 text-blue-200',
    green:  'bg-green-900/40 border-green-600 text-green-200',
    purple: 'bg-purple-900/40 border-purple-600 text-purple-200',
    orange: 'bg-orange-900/40 border-orange-600 text-orange-200',
    red:    'bg-red-900/40 border-red-600 text-red-200',
    yellow: 'bg-yellow-900/40 border-yellow-600 text-yellow-200',
    gray:   'bg-gray-800 border-gray-600 text-gray-200',
    cyan:   'bg-cyan-900/40 border-cyan-600 text-cyan-200',
  };
  return (
    <div className={`rounded-xl border px-4 py-3 text-center ${colors[color] ?? colors.gray} ${small ? 'text-xs' : 'text-sm'}`}>
      {icon && <div className="text-2xl mb-1">{icon}</div>}
      <div className="font-bold">{label}</div>
      {sub && <div className="text-xs opacity-70 mt-0.5">{sub}</div>}
    </div>
  );
}

function Pipeline() {
  return (
    <div className="bg-gray-900 rounded-2xl border border-gray-700 p-6 overflow-x-auto">
      <p className="text-gray-400 text-xs uppercase tracking-widest mb-6 text-center">
        Pipeline Completo — DQN Cart-Pole + Verificação ESBMC
      </p>

      {/* ── Linha 1: ambiente → treinamento → artefatos ── */}
      <div className="flex items-center justify-center gap-0 flex-wrap md:flex-nowrap">
        <FlowBox icon="🎮" label="CartPole Env" sub="cartpole_env.py" color="gray" />
        <Arrow vertical={false} />
        <FlowBox icon="🧠" label="Treinamento DQN" sub="train_dqn.py · 404 ep." color="blue" />
        <Arrow vertical={false} />
        <div className="flex flex-col gap-2">
          <FlowBox label="dqn_cartpole.pth" sub="pesos PyTorch" color="purple" small />
          <FlowBox label="dqn_cartpole.onnx" sub="grafo computacional" color="purple" small />
        </div>
      </div>

      <Arrow />

      {/* ── Linha 2: extração → quantização ── */}
      <div className="flex items-center justify-center gap-0 flex-wrap md:flex-nowrap">
        <FlowBox icon="🔬" label="Extração ONNX" sub="onnx_controller_extractor.py" color="cyan" />
        <Arrow vertical={false} />
        <FlowBox icon="⚖️" label="Quantização Q8.8" sub="scale = 256 · int(round(w×256))" color="cyan" />
        <Arrow vertical={false} />
        <div className="flex flex-col gap-2 text-xs text-center">
          <div className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-1 font-mono text-gray-300">w1[24×4], b1[24]</div>
          <div className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-1 font-mono text-gray-300">w2[24×24], b2[24]</div>
          <div className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-1 font-mono text-gray-300">wout[2×24], bout[2]</div>
        </div>
      </div>

      <Arrow />

      {/* ── Linha 3: três branches de verificação ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative">
        {/* Branch A */}
        <div className="flex flex-col items-center gap-2">
          <FlowBox icon="☠️" label="Neurônios Mortos" sub="verify_dead_neurons.py" color="orange" />
          <Arrow />
          <FlowBox label="Harness C" sub="assert(h == 0)" color="gray" small />
          <Arrow />
          <FlowBox icon="🔍" label="ESBMC / Boolector" sub="Domínio: Q8.8 inteiro" color="yellow" />
          <Arrow />
          <FlowBox label="24/24 VIVOS" sub="FAILED = neurônio ativo" color="green" small />
        </div>

        {/* Branch B */}
        <div className="flex flex-col items-center gap-2">
          <FlowBox icon="🔥" label="Saturação ReLU" sub="verify_saturation.py" color="orange" />
          <Arrow />
          <FlowBox label="Harness C" sub="assert(pre_i ≤ 0)" color="gray" small />
          <Arrow />
          <FlowBox icon="🔍" label="ESBMC / Boolector" sub="Domínio: Q8.8 inteiro" color="yellow" />
          <Arrow />
          <FlowBox label="0/24 saturados" sub="SUCCESSFUL = nunca ativa" color="green" small />
        </div>

        {/* Branch C */}
        <div className="flex flex-col items-center gap-2">
          <FlowBox icon="🔄" label="Malha Fechada" sub="verify_closed_loop.py" color="orange" />
          <Arrow />
          <FlowBox label="Aritmética de Intervalo" sub="bounds pré-ativação → __ESBMC_assume" color="gray" small />
          <Arrow />
          <FlowBox icon="🔍" label="ESBMC / Boolector" sub="Dinâmica linearizada 1 passo" color="yellow" />
          <Arrow />
          <div className="flex flex-col gap-1 w-full">
            <FlowBox label="Prop A-left: FAILED" sub="contraexemplo encontrado" color="red" small />
            <FlowBox label="Prop B: FAILED" sub="segurança violada" color="red" small />
          </div>
        </div>
      </div>

      <Arrow />

      {/* ── Linha 4: resultado ── */}
      <div className="flex justify-center">
        <FlowBox icon="📊" label="WebApp — Simulação + Verificação" sub="Next.js · contraexemplos visuais" color="blue" />
      </div>
    </div>
  );
}

/* ── Bloco de código ─────────────────────────────────────────────────────────── */
function Code({ children, lang = 'c' }: { children: string; lang?: string }) {
  return (
    <pre className="bg-gray-950 border border-gray-700 rounded-xl p-4 overflow-x-auto text-xs font-mono text-gray-300 leading-relaxed">
      <code>{children.trim()}</code>
    </pre>
  );
}

/* ── Seção numerada ──────────────────────────────────────────────────────────── */
function Section({ n, title, color = 'blue', children }: {
  n: number; title: string; color?: string; children: React.ReactNode;
}) {
  const colors: Record<string, string> = {
    blue:   'bg-blue-600',
    green:  'bg-green-600',
    purple: 'bg-purple-600',
    orange: 'bg-orange-600',
    red:    'bg-red-600',
    cyan:   'bg-cyan-600',
  };
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className={`${colors[color] ?? colors.blue} text-white font-bold text-sm w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0`}>
          {n}
        </span>
        <h2 className="text-xl font-bold text-white">{title}</h2>
      </div>
      <div className="pl-10 space-y-4">{children}</div>
    </div>
  );
}

function Pill({ label, color = 'gray' }: { label: string; color?: string }) {
  const colors: Record<string, string> = {
    green:  'bg-green-900 text-green-300',
    red:    'bg-red-900 text-red-300',
    yellow: 'bg-yellow-900 text-yellow-300',
    blue:   'bg-blue-900 text-blue-300',
    gray:   'bg-gray-700 text-gray-300',
    orange: 'bg-orange-900 text-orange-300',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold ${colors[color] ?? colors.gray}`}>
      {label}
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════ */
export default function MetodologiaPage() {
  return (
    <div className="space-y-12 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Metodologia Técnica</h1>
        <p className="text-gray-400">
          Pipeline completo de treinamento, extração, quantização e verificação formal
          do controlador DQN para o sistema Cart-Pole com ESBMC.
        </p>
      </div>

      {/* ── Diagrama ── */}
      <Pipeline />

      {/* ══════ Seção 1 — Cart-Pole ══════ */}
      <Section n={1} title="Sistema Físico — Cart-Pole" color="blue">
        <p className="text-gray-300 text-sm">
          O Cart-Pole é um sistema de controle clássico: um pêndulo invertido montado sobre um carro
          que se move horizontalmente. O estado é um vetor de 4 variáveis contínuas e o controlador
          deve aplicar uma força binária a cada passo de 20 ms para manter o pêndulo em pé.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { var: 'x', desc: 'Posição do carro', range: '−2.4 a 2.4 m' },
            { var: 'ẋ', desc: 'Velocidade do carro', range: '−5 a 5 m/s' },
            { var: 'θ', desc: 'Ângulo do pêndulo', range: '−12° a 12°' },
            { var: 'θ̇', desc: 'Vel. angular', range: '−5 a 5 rad/s' },
          ].map(({ var: v, desc, range }) => (
            <div key={v} className="bg-gray-800 border border-gray-700 rounded-xl p-3">
              <p className="font-mono text-blue-400 text-lg font-bold">{v}</p>
              <p className="text-gray-300 text-xs mt-1">{desc}</p>
              <p className="text-gray-500 text-xs font-mono mt-1">{range}</p>
            </div>
          ))}
        </div>
        <p className="text-gray-400 text-xs">
          Equações de movimento de Barto, Sutton & Anderson (1983), integração de Euler com
          dt=0.02s. Parâmetros: M_carro=1.0 kg, M_pêndulo=0.1 kg, L=0.5 m, F=±10 N.
          O episódio termina quando |x|{'>'}2.4 m ou |θ|{'>'}12°.
        </p>
      </Section>

      {/* ══════ Seção 2 — DQN ══════ */}
      <Section n={2} title="Treinamento — Deep Q-Network (DQN)" color="purple">
        <p className="text-gray-300 text-sm">
          O controlador é uma rede neural treinada com DQN: aprende a mapear estados do sistema
          para Q-values — estimativas do retorno acumulado futuro para cada ação. A ação escolhida
          é a de maior Q-value (<em>greedy</em>).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <h3 className="text-white font-semibold text-sm">Arquitetura da rede</h3>
            <div className="flex items-center gap-2 flex-wrap">
              {[
                { label: 'Input\n4 vars', bg: 'bg-blue-800' },
                { label: 'Hidden 1\n24 ReLU', bg: 'bg-purple-800' },
                { label: 'Hidden 2\n24 ReLU', bg: 'bg-purple-800' },
                { label: 'Output\n2 Q-vals', bg: 'bg-green-800' },
              ].map(({ label, bg }) => (
                <div key={label} className={`${bg} rounded-lg px-3 py-2 text-center text-xs text-white font-mono whitespace-pre-line`}>
                  {label}
                </div>
              ))}
            </div>
            <Code lang="python">{`
class QNetwork(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(4, 24),  # estado → camada 1
            nn.ReLU(),
            nn.Linear(24, 24), # camada 1 → camada 2
            nn.ReLU(),
            nn.Linear(24, 2),  # camada 2 → Q[esq, dir]
        )
    `}</Code>
          </div>
          <div className="space-y-3">
            <h3 className="text-white font-semibold text-sm">Hiperparâmetros</h3>
            <div className="space-y-1.5 text-xs font-mono">
              {[
                ['Replay Buffer', '10 000 transições'],
                ['Batch size', '64'],
                ['γ (desconto)', '0.99'],
                ['ε inicial', '1.0 → 0.01 (decay 0.995)'],
                ['Target update', 'a cada 200 steps'],
                ['Critério de parada', 'avg100 ≥ 470'],
                ['Convergência', '404 episódios'],
                ['Otimizador', 'Adam lr=1e-3'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-gray-700 pb-1">
                  <span className="text-gray-400">{k}</span>
                  <span className="text-gray-200">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 text-sm text-gray-300 space-y-2">
          <p className="font-semibold text-white text-sm">Como o DQN aprende</p>
          <p>
            A cada passo o agente observa o estado <code className="text-blue-300">s</code>, escolhe
            uma ação <code className="text-blue-300">a</code> (ε-greedy), recebe recompensa
            <code className="text-blue-300"> r=1</code> se não caiu, e armazena a transição
            <code className="text-blue-300"> (s, a, r, s′)</code> no replay buffer.
          </p>
          <p>
            A rede é atualizada minimizando o erro de Bellman:{' '}
            <code className="text-yellow-300 text-xs">
              L = (r + γ·max_a′ Q_target(s′,a′) − Q(s,a))²
            </code>
          </p>
        </div>
      </Section>

      {/* ══════ Seção 3 — Extração e Quantização ══════ */}
      <Section n={3} title="Extração ONNX e Quantização Q8.8" color="cyan">
        <p className="text-gray-300 text-sm">
          O ESBMC trabalha com aritmética inteira. Os pesos da rede (float32) são exportados
          para ONNX e então convertidos para <strong className="text-cyan-300">Q8.8</strong>:
          ponto fixo com 8 bits de parte inteira e 8 bits de fração (scale = 256).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Fórmula de quantização</h3>
            <Code>{`/* Float → Q8.8 */
q(v) = int(round(v × 256))

/* Multiplicação Q8.8 × Q8.8 → Q8.8 */
result = (a × b) / 256   /* divisão C: trunca p/ zero */

/* Exemplos */
w = 0.5    → q = 128
w = -0.25  → q = -64
b = 1.0    → q = 256`}</Code>
          </div>
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Forward pass quantizado</h3>
            <Code>{`/* Camada 1: 4 entradas → 24 neurônios */
for i in 0..23:
    pre = b1[i]
    for k in 0..3:
        pre += (x[k] * w1[i][k]) / 256
    h1[i] = max(0, pre)   /* ReLU */

/* Camada 2: 24 → 24 */
for i in 0..23:
    pre = b2[i]
    for k in 0..23:
        pre += (h1[k] * w2[i][k]) / 256
    h2[i] = max(0, pre)

/* Saída: 24 → 2 Q-values */
Q[0] = bout[0] + Σ h2[k]*wout[0][k] / 256
Q[1] = bout[1] + Σ h2[k]*wout[1][k] / 256`}</Code>
          </div>
        </div>

        <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 text-xs text-gray-300 space-y-1">
          <p className="text-yellow-300 font-semibold">⚠ Soundness: Python // vs C /</p>
          <p>
            Python usa divisão de piso (<code>-300 // 256 = -2</code>), enquanto C trunca para zero
            (<code>-300 / 256 = -1</code>). Para que os bounds de pré-ativação emitidos via{' '}
            <code>__ESBMC_assume</code> sejam corretos, usa-se{' '}
            <code className="text-yellow-300">c_div(a, b) = int(a / b)</code> em todas as
            computações de intervalo.
          </p>
        </div>
      </Section>

      {/* ══════ Seção 4 — Harness ESBMC ══════ */}
      <Section n={4} title="Geração de Harness C para o ESBMC" color="orange">
        <p className="text-gray-300 text-sm">
          O ESBMC verifica programas C. Para cada propriedade, um <em>harness</em> C é gerado
          automaticamente em Python, com todos os pesos hardcoded (sem loops) e entradas
          simbólicas via <code className="text-orange-300">nondet_int()</code>.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Estrutura do harness</h3>
            <Code>{`/* Primitivas ESBMC */
int  nondet_int(void);          // variável simbólica
void __ESBMC_assume(_Bool c);   // restringe domínio
void __ESBMC_assert(_Bool c, const char* msg);

int main(void) {
  /* 1. Entradas simbólicas */
  int x1 = nondet_int();
  __ESBMC_assume(x1 >= -614 && x1 <= 614);
  // ... x2, x3, x4

  /* 2. Forward pass Q8.8 (pesos hardcoded) */
  int pre0 = b1_0
    + (x1 * w1_0_0) / 256
    + (x2 * w1_0_1) / 256
    // ...
  int h0 = pre0 > 0 ? pre0 : 0;  // ReLU
  // ... todas as 24 saídas

  /* 3. Propriedade */
  __ESBMC_assert(h0 == 0,
    "neurônio 0 está morto?");
}`}</Code>
          </div>
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Aritmética de intervalo</h3>
            <p className="text-gray-400 text-xs mb-2">
              Para a verificação em malha fechada, bounds de pré-ativação são calculados
              analiticamente e usados em <code>__ESBMC_assume</code> para guiar o solver:
            </p>
            <Code lang="python">{`def compute_pre_bounds(qw, qb, in_lo, in_hi):
    lo, hi = qb[i], qb[i]
    for k in range(len(in_lo)):
        w = qw[i][k]
        if w >= 0:
            lo += c_div(in_lo[k] * w, SCALE)
            hi += c_div(in_hi[k] * w, SCALE)
        else:
            lo += c_div(in_hi[k] * w, SCALE)
            hi += c_div(in_lo[k] * w, SCALE)
    # → emite: __ESBMC_assume(pre >= lo && pre <= hi)`}</Code>
            <p className="text-gray-500 text-xs mt-2">
              Isso restringe as pré-ativações ao intervalo analiticamente correto, acelerando
              o solver sem excluir execuções reais.
            </p>
          </div>
        </div>
      </Section>

      {/* ══════ Seção 5 — Verificação ESBMC ══════ */}
      <Section n={5} title="Verificação Formal com ESBMC" color="green">
        <p className="text-gray-300 text-sm">
          O ESBMC (<em>Efficient SMT-Based Bounded Model Checker</em>) converte o programa C em
          fórmulas SMT e usa o solver <strong className="text-green-300">Boolector</strong> para
          determinar se existe uma atribuição de variáveis simbólicas que viola a propriedade.
        </p>

        <div className="space-y-4">
          {/* Prop 1 */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Pill label="Prop 1 — Neurônios Mortos" color="orange" />
              <Pill label="24/24 VIVOS" color="green" />
            </div>
            <p className="text-gray-300 text-sm">
              Para cada neurônio <em>i</em> da camada 1, verifica:{' '}
              <code className="text-yellow-300">assert(h1[i] == 0)</code> para todo estado no domínio.
            </p>
            <p className="text-gray-400 text-xs">
              <strong className="text-green-400">ESBMC FAILED</strong> = neurônio <strong>VIVO</strong> —
              o solver encontrou uma entrada que ativa o neurônio (ReLU {'>'} 0). Comportamento esperado.
              {' '}<strong className="text-red-400">ESBMC SUCCESS</strong> = neurônio <strong>MORTO</strong> —
              nunca ativa para nenhuma entrada válida; candidato a poda.
            </p>
          </div>

          {/* Prop 2 */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Pill label="Prop 2 — Saturação ReLU" color="orange" />
              <Pill label="0/24 saturados" color="green" />
            </div>
            <p className="text-gray-300 text-sm">
              Para cada neurônio <em>i</em>, verifica:{' '}
              <code className="text-yellow-300">assert(pre_i &lt;= 0)</code> — ou seja, tenta provar
              que a pré-ativação nunca é positiva (ReLU sempre cortaria). Se SUCCESS, o neurônio
              está sempre saturado.
            </p>
          </div>

          {/* Prop A */}
          <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Pill label="Prop A — Direção Errada" color="red" />
              <Pill label="FAILED (contraexemplo)" color="red" />
            </div>
            <p className="text-gray-300 text-sm">
              Quando θ {'<'} −0.10 rad e θ̇ ≤ 0 (pêndulo caindo à esquerda), o controlador
              <strong> deve</strong> empurrar à esquerda (ação=0). A propriedade verifica:{' '}
              <code className="text-yellow-300">assert(Q[0] {'>'} Q[1])</code> nessa região.
            </p>
            <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 font-mono text-xs text-red-200">
              Contraexemplo: x=−1.25m · ẋ=3.52m/s · θ=−0.19rad · θ̇=−0.42rad/s → ação=1 (ERRADO)
            </div>
            <p className="text-gray-400 text-xs">
              O ESBMC encontrou um estado real onde o controlador empurra na direção errada.
              Este estado é reproduzível e visualizável na página de Simulação (filtro "Falha ESBMC").
            </p>
          </div>

          {/* Prop B */}
          <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Pill label="Prop B — Segurança 1 Passo" color="red" />
              <Pill label="FAILED (contraexemplo)" color="red" />
            </div>
            <p className="text-gray-300 text-sm">
              Para qualquer estado inicial seguro (|θ|≤12°), aplica-se a ação do DQN e um passo
              de dinâmica linearizada (sin θ≈θ, cos θ≈1). A propriedade verifica:{' '}
              <code className="text-yellow-300">assert(|θ₁| ≤ 53)</code> (em Q8.8).
            </p>
            <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 font-mono text-xs text-red-200">
              Contraexemplo: x=−1.72m · ẋ=4.00m/s · θ=−0.14rad · θ̇=−4.59rad/s → |θ₁|{'>'}12°
            </div>
            <p className="text-gray-400 text-xs">
              Com velocidade angular extrema (θ̇=−4.59 rad/s), qualquer ação resulta em ângulo
              inseguro após 1 passo. O controlador não consegue garantir segurança nessa situação.
            </p>
          </div>
        </div>

        {/* Comando ESBMC */}
        <div>
          <h3 className="text-white font-semibold text-sm mb-2">Comando de verificação</h3>
          <Code>{`esbmc harness.c \\
  --no-div-by-zero-check \\
  --overflow-check \\
  --boolector \\
  --timeout 120`}</Code>
          <p className="text-gray-500 text-xs mt-2">
            O flag <code>--overflow-check</code> detecta overflows em aritmética Q8.8.
            Timeout de 120s por propriedade (Property A-right retornou TIMEOUT).
          </p>
        </div>
      </Section>

      {/* ══════ Seção 6 — Interpretação ══════ */}
      <Section n={6} title="Interpretação dos Resultados" color="red">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-900/20 border border-green-700 rounded-xl p-4 space-y-2">
            <p className="text-green-400 font-bold text-sm">✓ VERIFICATION FAILED</p>
            <p className="text-gray-300 text-xs">
              O solver encontrou uma entrada que <strong>viola</strong> a asserção.
              Para neurônios mortos: significa que o neurônio <strong>está vivo</strong> —
              existe uma entrada que o ativa.
            </p>
            <p className="text-gray-400 text-xs italic">
              "Falha" do ESBMC = boa notícia para neurônios mortos
            </p>
          </div>
          <div className="bg-blue-900/20 border border-blue-700 rounded-xl p-4 space-y-2">
            <p className="text-blue-400 font-bold text-sm">✓ VERIFICATION SUCCESSFUL</p>
            <p className="text-gray-300 text-xs">
              O solver <strong>provou</strong> que não existe nenhuma entrada que viola a asserção —
              a propriedade vale para todo o domínio verificado.
              Para neurônios mortos: o neurônio <strong>está morto</strong>.
            </p>
            <p className="text-gray-400 text-xs italic">
              Prova formal, não apenas testes
            </p>
          </div>
          <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 space-y-2">
            <p className="text-yellow-400 font-bold text-sm">⏱ TIMEOUT</p>
            <p className="text-gray-300 text-xs">
              O solver não concluiu dentro do limite de tempo (120s).
              O resultado é <strong>inconclusivo</strong> — a propriedade pode ser
              verdadeira ou falsa.
            </p>
            <p className="text-gray-400 text-xs italic">
              Property A-right: timeout (espaço de estados grande)
            </p>
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 text-sm text-gray-300 space-y-2">
          <p className="font-semibold text-white">Domínio verificado (Q8.8)</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
            {[
              ['x', '−614 a +614', '≙ −2.4 a 2.4 m'],
              ['ẋ', '−1280 a +1280', '≙ −5 a 5 m/s'],
              ['θ', '−53 a +53', '≙ −12° a 12°'],
              ['θ̇', '−1280 a +1280', '≙ −5 a 5 rad/s'],
            ].map(([v, q, f]) => (
              <div key={v} className="bg-gray-900 rounded-lg p-2">
                <p className="text-blue-400 font-bold">{v}</p>
                <p className="text-gray-300">{q}</p>
                <p className="text-gray-500">{f}</p>
              </div>
            ))}
          </div>
        </div>
      </Section>
    </div>
  );
}
