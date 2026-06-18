'use client';

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
  label, sub, color = 'gray', small = false,
}: {
  label: string; sub?: string; color?: string; small?: boolean;
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
      <div className="font-bold">{label}</div>
      {sub && <div className="text-xs opacity-70 mt-0.5">{sub}</div>}
    </div>
  );
}

function Pipeline() {
  return (
    <div className="bg-gray-900 rounded-2xl border border-gray-700 p-6 overflow-x-auto">
      <p className="text-gray-400 text-xs uppercase tracking-widest mb-6 text-center">
        Pipeline Completo — DDPG Cart-Pole + Verificacao ESBMC
      </p>

      {/* Row 1: Training */}
      <div className="flex items-center justify-center gap-0 flex-wrap md:flex-nowrap">
        <FlowBox label="CartPole Env" sub="cartpole_env.py" color="gray" />
        <Arrow vertical={false} />
        <FlowBox label="Treinamento DDPG" sub="train_ddpg.py" color="blue" />
        <Arrow vertical={false} />
        <FlowBox label="ddpg_actor_best.pth" sub="Actor: 4->24->24->1" color="purple" small />
      </div>

      <Arrow />

      {/* Row 2: Quantization */}
      <div className="flex items-center justify-center gap-0 flex-wrap md:flex-nowrap">
        <FlowBox label="Extracao de Pesos" sub="ddpg_weight_extractor.py" color="cyan" />
        <Arrow vertical={false} />
        <FlowBox label="Quantizacao Q8.8" sub="scale=256, round(w*256)" color="cyan" />
        <Arrow vertical={false} />
        <div className="flex flex-col gap-2">
          <FlowBox label="ddpg_weights_q88.json" sub="pesos inteiros para browser" color="purple" small />
          <FlowBox label="quantization_report.json" sub="analise de erro" color="gray" small />
        </div>
      </div>

      <Arrow />

      {/* Row 3: Two verification domains */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Domain 1: Model */}
        <div className="border border-orange-700/50 rounded-xl p-4 space-y-2">
          <p className="text-orange-300 font-bold text-sm text-center">Dominio 1 — Modelo (NN)</p>
          <div className="flex flex-col items-center gap-2">
            <FlowBox label="Neuronios Mortos" sub="verify_ddpg_dead_neurons.py" color="orange" small />
            <Arrow />
            <FlowBox label="48 harnesses (L1+L2)" sub="assert(h==0)" color="gray" small />
            <Arrow />
            <FlowBox label="ESBMC / Boolector" sub="Q8.8 inteiro" color="yellow" small />
            <Arrow />
            <FlowBox label="0/48 mortos, 0 saturados" sub="RESPONSIVE" color="green" small />
          </div>
        </div>

        {/* Domain 2: Controller */}
        <div className="border border-blue-700/50 rounded-xl p-4 space-y-2">
          <p className="text-blue-300 font-bold text-sm text-center">Dominio 2 — Controlador (Malha Fechada)</p>
          <div className="flex flex-col items-center gap-2">
            <FlowBox label="Propriedades A, B, C" sub="verify_ddpg_closed_loop.py" color="blue" small />
            <Arrow />
            <FlowBox label="Aritmetica de Intervalo" sub="bounds -> __ESBMC_assume" color="gray" small />
            <Arrow />
            <FlowBox label="ESBMC + Tanh Piecewise" sub="Dinamica linearizada" color="yellow" small />
            <Arrow />
            <div className="flex flex-col gap-1 w-full">
              <FlowBox label="Prop C: SUCCESSFUL" sub="|F| <= 10 N sempre" color="green" small />
              <FlowBox label="Prop B: FAILED" sub="contraexemplo encontrado" color="red" small />
              <FlowBox label="Prop A: TIMEOUT" sub="120s insuficiente" color="yellow" small />
            </div>
          </div>
        </div>
      </div>

      <Arrow />

      {/* Row 4: Webapp */}
      <div className="flex justify-center">
        <FlowBox label="WebApp — Q8.8 em tempo real" sub="Next.js + contraexemplos visuais" color="blue" />
      </div>
    </div>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="bg-gray-950 border border-gray-700 rounded-xl p-4 overflow-x-auto text-xs font-mono text-gray-300 leading-relaxed">
      <code>{children.trim()}</code>
    </pre>
  );
}

function Section({ n, title, color = 'blue', children }: {
  n: number; title: string; color?: string; children: React.ReactNode;
}) {
  const colors: Record<string, string> = {
    blue: 'bg-blue-600', green: 'bg-green-600', purple: 'bg-purple-600',
    orange: 'bg-orange-600', red: 'bg-red-600', cyan: 'bg-cyan-600',
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
    green: 'bg-green-900 text-green-300', red: 'bg-red-900 text-red-300',
    yellow: 'bg-yellow-900 text-yellow-300', blue: 'bg-blue-900 text-blue-300',
    gray: 'bg-gray-700 text-gray-300', orange: 'bg-orange-900 text-orange-300',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold ${colors[color] ?? colors.gray}`}>
      {label}
    </span>
  );
}

export default function MetodologiaPage() {
  return (
    <div className="space-y-12 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Metodologia Tecnica</h1>
        <p className="text-gray-400">
          Pipeline de treinamento, quantizacao Q8.8 e verificacao formal do controlador DDPG continuo
          para Cart-Pole com ESBMC. O browser executa a mesma aritmetica inteira verificada.
        </p>
      </div>

      <Pipeline />

      {/* Section 1: Cart-Pole */}
      <Section n={1} title="Sistema Fisico — Cart-Pole" color="blue">
        <p className="text-gray-300 text-sm">
          Pendulo invertido sobre carro movel. Estado = vetor 4D. Controlador DDPG aplica forca
          continua F in [-10, +10] N a cada 20 ms para manter equilibrio.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { v: 'x',         desc: 'Posicao do carro',    range: '-2.4 a 2.4 m' },
            { v: 'x_dot',     desc: 'Velocidade do carro',  range: '-5 a 5 m/s' },
            { v: 'theta',     desc: 'Angulo do pendulo',    range: '-12 a 12 graus' },
            { v: 'theta_dot', desc: 'Vel. angular',          range: '-5 a 5 rad/s' },
          ].map(({ v, desc, range }) => (
            <div key={v} className="bg-gray-800 border border-gray-700 rounded-xl p-3">
              <p className="font-mono text-blue-400 text-lg font-bold">{v}</p>
              <p className="text-gray-300 text-xs mt-1">{desc}</p>
              <p className="text-gray-500 text-xs font-mono mt-1">{range}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Section 2: DDPG */}
      <Section n={2} title="Treinamento — DDPG (Continuo)" color="purple">
        <p className="text-gray-300 text-sm">
          Deep Deterministic Policy Gradient — actor-critic para controle continuo.
          O Actor mapeia estado para forca continua via tanh x 10. O Critic avalia a qualidade.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <h3 className="text-white font-semibold text-sm">Arquitetura do Actor</h3>
            <div className="flex items-center gap-2 flex-wrap">
              {[
                { label: 'Input\n4 vars', bg: 'bg-blue-800' },
                { label: 'Hidden 1\n24 ReLU', bg: 'bg-purple-800' },
                { label: 'Hidden 2\n24 ReLU', bg: 'bg-purple-800' },
                { label: 'Output\n1 Tanh x10', bg: 'bg-green-800' },
              ].map(({ label, bg }) => (
                <div key={label} className={`${bg} rounded-lg px-3 py-2 text-center text-xs text-white font-mono whitespace-pre-line`}>
                  {label}
                </div>
              ))}
            </div>
            <Code>{`
class DDPGActor(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(4, 24),   # estado -> camada 1
            nn.ReLU(),
            nn.Linear(24, 24),  # camada 1 -> camada 2
            nn.ReLU(),
            nn.Linear(24, 1),   # camada 2 -> forca
        )
    def forward(self, x):
        return torch.tanh(self.net(x)) * 10
            `}</Code>
          </div>
          <div className="space-y-3">
            <h3 className="text-white font-semibold text-sm">Diferenca DQN vs DDPG</h3>
            <div className="space-y-2 text-xs">
              {[
                ['DQN (legado)', '2 acoes discretas: esquerda/direita', 'bg-gray-800'],
                ['DDPG (atual)', 'Forca continua: F in [-10, +10] N', 'bg-blue-900/40'],
              ].map(([title, desc, bg]) => (
                <div key={title} className={`${bg} border border-gray-700 rounded-lg p-3`}>
                  <p className="text-white font-semibold">{title}</p>
                  <p className="text-gray-400 mt-1">{desc}</p>
                </div>
              ))}
            </div>
            <p className="text-gray-400 text-xs">
              Saida tanh garante F in [-1, 1], multiplicado por 10 para F in [-10, +10] N.
              Isso e crucial para a Property C (bounds) — tanh garante limites fisicos.
            </p>
          </div>
        </div>
      </Section>

      {/* Section 3: Quantization */}
      <Section n={3} title="Quantizacao Q8.8 — Zero Gap de Fidelidade" color="cyan">
        <p className="text-gray-300 text-sm">
          O controlador no browser executa a <strong className="text-cyan-300">mesma aritmetica inteira</strong> verificada
          pelo ESBMC. Pesos float32 sao convertidos para Q8.8 (scale=256) e toda multiplicacao usa
          divisao C (truncamento para zero).
        </p>

        <div className="bg-cyan-900/20 border border-cyan-700 rounded-xl p-4 text-sm">
          <h3 className="text-cyan-300 font-semibold mb-2">Principio fundamental</h3>
          <p className="text-gray-300">
            Em vez de verificar Q8.8 mas rodar float32 (com gap de fidelidade), rodamos Q8.8
            no browser. Contraexemplos do ESBMC reproduzem <em>exatamente</em> no browser.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Quantizacao</h3>
            <Code>{`/* Float -> Q8.8 */
q(v) = int(round(v * 256))

/* Multiplicacao Q8.8 x Q8.8 -> Q8.8 */
result = (a * b) / 256   /* divisao C: trunca p/ zero */

/* Tanh piecewise linear (5 segmentos) */
if (z_abs <= 64)       t = (z_abs * 252) / 256
elif (z_abs <= 192)    t = 62 + ((z_abs-64)*200)/256
elif (z_abs <= 384)    t = 162 + ((z_abs-192)*92)/256
elif (z_abs <= 768)    t = 231 + ((z_abs-384)*16)/256
else                   t = 255`}</Code>
          </div>
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">Forward pass quantizado</h3>
            <Code>{`/* Camada 1: 4 -> 24 neuronios */
for i in 0..23:
    pre = b1[i]
    for k in 0..3:
        pre += (x[k] * w1[i][k]) / 256
    h1[i] = max(0, pre)   /* ReLU */

/* Camada 2: 24 -> 24 */
for i in 0..23:
    pre = b2[i]
    for k in 0..23:
        pre += (h1[k] * w2[i][k]) / 256
    h2[i] = max(0, pre)

/* Saida: 24 -> 1, tanh, x10 */
z = bout + sum(h2[k]*wout[k]/256)
F = tanh_q88(z) * 10 * 256 / 256`}</Code>
          </div>
        </div>

        <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 text-xs text-gray-300">
          <p className="text-yellow-300 font-semibold">Soundness: Python // vs C /</p>
          <p className="mt-1">
            Python usa divisao de piso (floor), C trunca para zero. Para bounds corretos,
            usa-se cdiv(a,b) = int(a/b) em todas as computacoes de intervalo.
          </p>
        </div>
      </Section>

      {/* Section 4: Harness */}
      <Section n={4} title="Harness C para ESBMC" color="orange">
        <p className="text-gray-300 text-sm">
          Harnesses C gerados automaticamente. Todos os 24+24 neuronios expandidos inline (sem loops).
          Entradas simbolicas via nondet_int(). Bounds de pre-ativacao via __ESBMC_assume.
        </p>

        <Code>{`/* Estrutura do harness DDPG */
int x = nondet_int(), xd = nondet_int();
int th = nondet_int(), thd = nondet_int();
__ESBMC_assume(x >= -614 && x <= 614);    // [-2.4, 2.4]m
__ESBMC_assume(xd >= -1280 && xd <= 1280); // [-5, 5]m/s
__ESBMC_assume(th >= -53 && th <= 53);      // [-12, 12]graus
__ESBMC_assume(thd >= -1280 && thd <= 1280);

/* Forward pass (24+24 neuronios, todos inline) */
int pre1_0 = (x*(-45))/256 + (xd*(112))/256 + ... + (-289);
__ESBMC_assume(pre1_0 >= -12345 && pre1_0 <= 6789);
int h1_0 = pre1_0 > 0 ? pre1_0 : 0;
/* ... 47 neuronios mais ... */

/* Saida z + tanh piecewise + F_Q */
int z = (h2_0*(w))/256 + ... + (bout);
int F_Q = tanh_q88(z) * 10;  /* em Q8.8 */

/* Propriedade */
__ESBMC_assert(F_Q >= -2560 && F_Q <= 2560, "bounds");`}</Code>
      </Section>

      {/* Section 5: Verification */}
      <Section n={5} title="Verificacao Formal com ESBMC" color="green">
        <div className="space-y-4">
          {/* Domain 1 */}
          <div className="bg-orange-900/10 border border-orange-700/50 rounded-xl p-4 space-y-3">
            <h3 className="text-orange-300 font-semibold">Dominio 1 — Modelo (Rede Neural)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Pill label="Neuronios Mortos" color="orange" />
                  <Pill label="0/48 mortos" color="green" />
                </div>
                <p className="text-gray-400 text-xs">
                  assert(h[i] == 0) para cada neuronio. FAILED = VIVO. 24 L1 + 24 L2.
                </p>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Pill label="Saturacao" color="orange" />
                  <Pill label="0/24 saturados" color="green" />
                </div>
                <p className="text-gray-400 text-xs">
                  assert(pre_i {'<='} 0) para verificar se ReLU nunca ativa. + saida responsiva.
                </p>
              </div>
            </div>
          </div>

          {/* Domain 2 */}
          <div className="bg-blue-900/10 border border-blue-700/50 rounded-xl p-4 space-y-3">
            <h3 className="text-blue-300 font-semibold">Dominio 2 — Controlador (Malha Fechada)</h3>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Pill label="Prop A — Direcao" color="yellow" />
                <Pill label="TIMEOUT" color="yellow" />
              </div>
              <p className="text-gray-400 text-xs">
                theta {'>'} 5.6 graus e theta_dot {'>'} 0 implica F {'>'} 0.
                Usa monotonicidade de tanh (z {'>'} 0 iff F {'>'} 0). Timeout 120s — espaco de estados grande.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Pill label="Prop B — Seguranca 1-step" color="red" />
                <Pill label="FAILED" color="red" />
              </div>
              <p className="text-gray-400 text-xs">
                Estado seguro sai da regiao segura em 1 passo de dinamica linearizada.
              </p>
              <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 font-mono text-xs text-red-200">
                Contraexemplo: x=-0.75m, x_dot=-3.92m/s, theta=-0.18rad, theta_dot=-1.52rad/s
              </div>
              <p className="text-gray-400 text-xs">
                Com velocidade angular significativa, o controlador nao consegue manter seguranca em 1 passo.
                Este estado e injetavel na simulacao em tempo real.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Pill label="Prop C — Bounds de Forca" color="green" />
                <Pill label="SUCCESSFUL" color="green" />
              </div>
              <p className="text-gray-400 text-xs">
                |F| {'<='} 10 N para todo estado. ESBMC provou formalmente — tanh garante |output| {'<='} 1.
              </p>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-white font-semibold text-sm mb-2">Comando de verificacao</h3>
          <Code>{`esbmc harness.c \\
  --no-unwinding-assertions \\
  --boolector \\
  --timeout 120`}</Code>
        </div>
      </Section>

      {/* Section 6: Tanh */}
      <Section n={6} title="Aproximacao de Tanh — Abordagem Hibrida" color="red">
        <p className="text-gray-300 text-sm">
          Tanh nao-linear e tratada de duas formas dependendo da propriedade:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2">
            <p className="text-blue-400 font-bold text-sm">Property A — Monotonicidade</p>
            <p className="text-gray-300 text-xs">
              z {'>'} 0 iff tanh(z) {'>'} 0 iff F {'>'} 0. Nao precisa computar tanh.
              Basta verificar o sinal de z (pre-tanh).
            </p>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2">
            <p className="text-orange-400 font-bold text-sm">Property B — Tanh Piecewise</p>
            <p className="text-gray-300 text-xs">
              Precisa do valor numerico de F para a dinamica. Usa os mesmos 5 segmentos
              lineares do browser TypeScript. Zero gap de fidelidade.
            </p>
          </div>
        </div>
      </Section>

      {/* Section 7: Interpretation */}
      <Section n={7} title="Interpretacao dos Resultados" color="green">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-900/20 border border-green-700 rounded-xl p-4 space-y-2">
            <p className="text-green-400 font-bold text-sm">VERIFICATION FAILED</p>
            <p className="text-gray-300 text-xs">
              Solver encontrou entrada que viola a assercao. Para neuronios: neuronio VIVO.
              Para propriedades: contraexemplo reproduzivel.
            </p>
          </div>
          <div className="bg-blue-900/20 border border-blue-700 rounded-xl p-4 space-y-2">
            <p className="text-blue-400 font-bold text-sm">VERIFICATION SUCCESSFUL</p>
            <p className="text-gray-300 text-xs">
              Prova formal: nenhuma entrada viola a propriedade em todo o dominio.
              Ex: Property C (|F| {'<='} 10 N) verificada para todo estado.
            </p>
          </div>
          <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 space-y-2">
            <p className="text-yellow-400 font-bold text-sm">TIMEOUT</p>
            <p className="text-gray-300 text-xs">
              Solver nao concluiu em 120s. Resultado inconclusivo.
              Property A com espaco de estados grande.
            </p>
          </div>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 text-sm text-gray-300">
          <p className="font-semibold text-white mb-2">Dominio verificado (Q8.8)</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
            {[
              ['x', '-614 a +614', '-2.4 a 2.4 m'],
              ['x_dot', '-1280 a +1280', '-5 a 5 m/s'],
              ['theta', '-53 a +53', '-12 a 12 graus'],
              ['theta_dot', '-1280 a +1280', '-5 a 5 rad/s'],
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
