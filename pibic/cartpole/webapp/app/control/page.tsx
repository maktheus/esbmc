'use client';

const CARD = 'bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-3';
const CODE = 'bg-gray-900 rounded-lg p-3 font-mono text-xs text-green-300 overflow-x-auto whitespace-pre';

function ClosedLoopDiagram() {
  const boxW = 110, boxH = 38;
  const nodes = [
    { x: 20,  y: 85,  label: 'Estado s_t',   sub: '[x, ẋ, θ, θ̇]', fill: '#1e3a5f', stroke: '#3b82f6' },
    { x: 162, y: 85,  label: 'DQN',           sub: '4→24→24→5',           fill: '#1a3a2a', stroke: '#22c55e' },
    { x: 304, y: 85,  label: 'Ação a_t',   sub: 'argmax Q(s,a)',               fill: '#3a2010', stroke: '#f59e0b' },
    { x: 446, y: 85,  label: 'Planta',         sub: 'Cart-Pole (Euler)',                  fill: '#2d1b69', stroke: '#a78bfa' },
  ];
  const paths = [
    'M 130 104 L 162 104',
    'M 272 104 L 304 104',
    'M 414 104 L 446 104',
    'M 556 104 L 580 104 L 580 185 L 20 185 L 20 123',
    'M 359 85 L 359 48 L 504 48 L 504 85',
  ];
  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 600 200" className="w-full" style={{ minWidth: 500, maxHeight: 200 }}>
        <defs>
          <marker id="ar" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6b7280" />
          </marker>
        </defs>
        {paths.map((d, i) => (
          <path key={i} d={d} fill="none" stroke="#6b7280" strokeWidth={1.5} markerEnd="url(#ar)" />
        ))}
        {nodes.map((n) => (
          <g key={n.label}>
            <rect x={n.x} y={n.y} width={boxW} height={boxH} rx={5}
              fill={n.fill} stroke={n.stroke} strokeWidth={1.5} />
            <text x={n.x + boxW / 2} y={n.y + 14} textAnchor="middle"
              fill="white" fontSize={10} fontWeight="bold">{n.label}</text>
            <text x={n.x + boxW / 2} y={n.y + 27} textAnchor="middle"
              fill="#9ca3af" fontSize={8}>{n.sub}</text>
          </g>
        ))}
        <text x={300} y={198} textAnchor="middle" fill="#4b5563" fontSize={9}>
          s_t+1 retroalimenta — loop fechado a cada dt = 0.02 s
        </text>
      </svg>
    </div>
  );
}

function PIDvsDQN() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
        <p className="text-blue-400 font-semibold text-sm mb-2">Controle PID Clássico</p>
        <div className={CODE}>
          {`e = theta_ref - theta  // erro\nF = Kp*e + Ki*int(e) + Kd*de/dt\n// Kp=25, Ki=0, Kd=5  (exemplo)`}
        </div>
        <ul className="mt-3 space-y-1 text-xs text-gray-400">
          <li>&#10003; Simples e interpretável</li>
          <li>&#10003; Garantias analíticas de estabilidade</li>
          <li>&#10007; Requer tunagem manual (Kp, Ki, Kd)</li>
          <li>&#10007; Linear — piora com não-linearidades</li>
          <li>&#10007; Não considera posição do carro (x)</li>
        </ul>
      </div>
      <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
        <p className="text-green-400 font-semibold text-sm mb-2">Controle DQN (este projeto)</p>
        <div className={CODE}>
          {`Q(s, a) = E[sum_k gamma^k * r_k]\na* = argmax_a Q(s, a)\n// s=[x,xd,theta,thd], gamma=0.99`}
        </div>
        <ul className="mt-3 space-y-1 text-xs text-gray-400">
          <li>&#10003; Aprende por trial-and-error automático</li>
          <li>&#10003; Considera todo o estado (x, ẋ, θ, θ̇)</li>
          <li>&#10003; 5 níveis de força (−5 a +10 N)</li>
          <li>&#10007; Caixa-preta — difícil de analisar</li>
          <li>&#10007; Requer verificação formal (ESBMC)</li>
        </ul>
      </div>
    </div>
  );
}

export default function ControlPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Teoria de Controle</h1>
        <span className="text-gray-400 text-sm">Como o DQN controla o Cart-Pole</span>
      </div>

      {/* 1. O problema */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">1. O Problema de Controle</p>
        <p className="text-gray-300 text-sm leading-relaxed">
          Um <strong className="text-white">pêndulo invertido sobre um carro</strong> é instável:
          qualquer perturbação em θ cresce sem intervenção. O objetivo é manter o pêndulo ereto
          (θ ≈ 0) e o carro dentro dos limites (|x| ≤ 2.4 m) aplicando forças horizontais.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { sym: 'x',   desc: 'Posição do carro',    lim: '±2.4 m',   col: 'border-blue-700 text-blue-400'   },
            { sym: 'ẋ',   desc: 'Vel. do carro',       lim: '±5 m/s',   col: 'border-blue-600 text-blue-300'   },
            { sym: 'θ',   desc: 'Ângulo do pêndulo',   lim: '±12°',     col: 'border-green-700 text-green-400' },
            { sym: 'θ̇',  desc: 'Vel. angular',         lim: '±5 rad/s', col: 'border-green-600 text-green-300' },
          ].map(s => (
            <div key={s.sym} className={`bg-gray-900 rounded-lg p-3 border ${s.col}`}>
              <p className={`text-xl font-bold font-mono ${s.col.split(' ')[1]}`}>{s.sym}</p>
              <p className="text-gray-400 text-xs mt-1">{s.desc}</p>
              <p className="text-red-400 text-xs mt-0.5">falha: {s.lim}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 2. Laço fechado */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">2. Laço de Controle Fechado</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          A cada 20 ms o estado é medido, passado ao DQN, que decide a força, que altera o estado
          — ciclo de retroalimentação contínuo.
        </p>
        <ClosedLoopDiagram />
        <div className={CODE}>
          {`# Loop de controle — 50 Hz\nwhile not done:\n    s_t  = env.state                   # [x, xd, theta, thd]\n    a_t  = argmax Q(s_t, a)            # DQN decide ação 0-4\n    F    = FORCE_LEVELS[a_t]           # [-10,-5,0,+5,+10] N\n    s_t1, r_t, done = env.step(a_t)   # avança física dt=0.02s`}
        </div>
      </div>

      {/* 3. Arquitetura QNetwork */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">3. Arquitetura do Controlador Neural</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          A rede tem 2 camadas ocultas de 24 neurônios. Cada saída é o <em>Q-value</em> —
          retorno esperado acumulado para aquela ação a partir do estado atual.
        </p>
        <div className="flex items-center justify-around gap-2 py-4 overflow-x-auto bg-gray-900 rounded-xl border border-gray-700">
          {[
            { title: 'Entrada (4)',    items: ['x', 'ẋ', 'θ', 'θ̇'],                    color: '#3b82f6' },
            { title: 'Oculta 1 (24)', items: ['●', '●', '●', '●', '●'],               color: '#22c55e' },
            { title: 'Oculta 2 (24)', items: ['●', '●', '●', '●', '●'],               color: '#22c55e' },
            { title: 'Saída Q (5)',   items: ['−10N', '−5N', '0N', '+5N', '+10N'],     color: '#f59e0b' },
          ].map((l, li) => (
            <div key={li} className="flex flex-col items-center gap-1.5 min-w-[72px]">
              <p className="text-gray-400 text-xs text-center">{l.title}</p>
              {l.items.map((n, ni) => (
                <div key={ni}
                  className="rounded-full flex items-center justify-center text-center leading-tight"
                  style={{
                    width: li === 0 || li === 3 ? 60 : 18,
                    height: li === 0 || li === 3 ? 22 : 18,
                    background: l.color + '22',
                    border: `1.5px solid ${l.color}`,
                    color: l.color,
                    fontSize: 9,
                  }}
                >
                  {n}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className={CODE}>
          {`# Equação de Bellman (alvo do treino)\nQ_target(s,a) = r + gamma * max_a' Q(s', a')   gamma=0.99\nLoss = MSE(Q_pred(s,a), Q_target(s,a))\n\n# Inferência\na* = argmax_{a} Q(s, a)   // 1 forward pass da rede`}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-400">
          {[
            { t: 'Replay Buffer', d: '50k transições (s, a, r, s\'). Amostragem aleatória quebra correlação temporal — necessário para convergência.' },
            { t: 'Target Network', d: 'Cópia congelada dos pesos. Atualizada a cada 100 passos. Estabiliza os alvos de treino.' },
            { t: 'ε-greedy', d: 'ε decai 1.0→0.01 (×0.998/ep). Garante exploração antes de explotar o aprendido.' },
          ].map(c => (
            <div key={c.t} className="bg-gray-900 rounded-lg p-3">
              <p className="text-white font-semibold mb-1">{c.t}</p>
              <p>{c.d}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. PID vs DQN */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">4. Controle Clássico vs DQN</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          O PID é o controlador industrial mais usado. Comparar com DQN revela as trocas entre
          interpretabilidade e capacidade de representação.
        </p>
        <PIDvsDQN />
        <p className="text-gray-500 text-xs mt-2">
          Um PID bem tunado estabiliza o pêndulo mas não considera x explicitamente.
          O DQN aprende implicitamente a balancear ambos os objetivos.
        </p>
      </div>

      {/* 5. Conexão com a simulação */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">5. Da Teoria à Simulação</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          Cada frame da animação na página de <strong className="text-white">Simulação</strong> corresponde a
          1 ciclo do laço (dt = 0.02 s). Os gráficos em tempo real mostram como o DQN reage
          ao estado — observe o controlador corrigindo θ enquanto mantém x nos limites.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="space-y-2">
            <p className="text-gray-300 font-semibold">Entradas (4 gráficos)</p>
            <ul className="space-y-1 text-gray-400">
              <li><span className="text-blue-400 font-mono">x, ẋ</span> — gráficos azuis: carro</li>
              <li><span className="text-green-400 font-mono">θ, θ̇</span> — gráficos verdes: pêndulo</li>
              <li>Linha vermelha = limite de falha</li>
            </ul>
          </div>
          <div className="space-y-2">
            <p className="text-gray-300 font-semibold">Saídas DQN (5 barras)</p>
            <ul className="space-y-1 text-gray-400">
              <li>Cada barra = Q-value de uma força</li>
              <li><span className="text-green-400">Verde</span> = ação escolhida (argmax)</li>
              <li>Diferença entre barras = confiança</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 6. Por que ESBMC */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">6. Por que Verificação Formal?</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          Simulação testa casos específicos. ESBMC prova propriedades para
          <strong className="text-white"> todos os estados possíveis</strong> no domínio.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {[
            { t: 'Simulação',      c: 'border-gray-600 text-gray-300',
              items: ['✓ Fácil de executar', '✓ Visualizável', '✗ Só seeds específicas', '✗ Sem garantia geral'] },
            { t: 'Testes',         c: 'border-blue-800 text-blue-300',
              items: ['✓ Automatizados', '✓ Rápidos', '✗ Cobertura limitada', '✗ Espaço contínuo'] },
            { t: 'ESBMC',          c: 'border-green-700 text-green-400',
              items: ['✓ Prova ∀ estados', '✓ Contraexemplo concreto', '✓ Sound (sem falsos neg.)', '✗ Mais lento'] },
          ].map(c => (
            <div key={c.t} className={`bg-gray-900 rounded-lg p-3 border ${c.c.split(' ')[0]}`}>
              <p className={`font-semibold mb-2 ${c.c.split(' ')[1]}`}>{c.t}</p>
              <ul className="space-y-1 text-gray-400">
                {c.items.map(it => <li key={it}>{it}</li>)}
              </ul>
            </div>
          ))}
        </div>
        <p className="text-gray-400 text-xs mt-2">
          Ver detalhes em <span className="text-blue-400">Verificação</span> e metodologia em{' '}
          <span className="text-blue-400">Metodologia</span>.
        </p>
      </div>
    </div>
  );
}
