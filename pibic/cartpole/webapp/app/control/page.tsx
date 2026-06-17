'use client';

const CARD = 'bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-3';
const CODE = 'bg-gray-900 rounded-lg p-3 font-mono text-xs text-green-300 overflow-x-auto whitespace-pre';

function ClosedLoopDiagram() {
  const boxW = 110, boxH = 38;
  const nodes = [
    { x: 20,  y: 85,  label: 'Estado s_t',   sub: '[x, ẋ, θ, θ̇]', fill: '#1e3a5f', stroke: '#3b82f6' },
    { x: 162, y: 85,  label: 'DDPG Actor',    sub: '4→24→24→tanh',           fill: '#1a3a2a', stroke: '#22c55e' },
    { x: 304, y: 85,  label: 'Forca F',       sub: 'F ∈ [-10,+10] N',         fill: '#3a2010', stroke: '#f59e0b' },
    { x: 446, y: 85,  label: 'Planta',        sub: 'Cart-Pole (Euler)',        fill: '#2d1b69', stroke: '#a78bfa' },
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

function PIDvsDDPG() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
        <p className="text-blue-400 font-semibold text-sm mb-2">Controle PID Classico</p>
        <div className={CODE}>
          {`e = theta_ref - theta  // erro\nF = Kp*e + Ki*int(e) + Kd*de/dt\n// Kp=25, Ki=0, Kd=5  (exemplo)\n// Saida: forca continua (analogico)`}
        </div>
        <ul className="mt-3 space-y-1 text-xs text-gray-400">
          <li>&#10003; Simples e interpretavel</li>
          <li>&#10003; Garantias analiticas de estabilidade</li>
          <li>&#10003; Saida continua (sem discretizacao)</li>
          <li>&#10007; Requer tunagem manual (Kp, Ki, Kd)</li>
          <li>&#10007; Linear — piora com nao-linearidades</li>
          <li>&#10007; Nao considera posicao do carro (x)</li>
        </ul>
      </div>
      <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
        <p className="text-green-400 font-semibold text-sm mb-2">Controle DDPG (este projeto)</p>
        <div className={CODE}>
          {`# Actor (politica deterministica)\nmu(s) = tanh(NN(s)) * 10  // N\n# s=[x,xd,theta,thd]\n# Saida: forca continua F ∈ [-10,+10]`}
        </div>
        <ul className="mt-3 space-y-1 text-xs text-gray-400">
          <li>&#10003; Aprende por trial-and-error automatico</li>
          <li>&#10003; Considera todo o estado (x, ẋ, θ, θ̇)</li>
          <li>&#10003; Forca continua — modulacao fina</li>
          <li>&#10003; Nao-linear — adapta-se a dinamica real</li>
          <li>&#10007; Caixa-preta — dificil de analisar</li>
          <li>&#10007; Requer verificacao formal (ESBMC)</li>
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
        <span className="text-gray-400 text-sm">Controlador DDPG continuo para Cart-Pole</span>
      </div>

      {/* 1. O problema */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">1. O Problema de Controle</p>
        <p className="text-gray-300 text-sm leading-relaxed">
          Um <strong className="text-white">pendulo invertido sobre um carro</strong> e instavel:
          qualquer perturbacao em θ cresce sem intervencao. O objetivo e manter o pendulo ereto
          (θ ≈ 0) e o carro dentro dos limites (|x| ≤ 2.4 m) aplicando forca horizontal
          <strong className="text-yellow-300"> continua</strong> F ∈ [-10, +10] N.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { sym: 'x',   desc: 'Posicao do carro',    lim: '±2.4 m',   col: 'border-blue-700 text-blue-400'   },
            { sym: 'ẋ',   desc: 'Vel. do carro',       lim: '±5 m/s',   col: 'border-blue-600 text-blue-300'   },
            { sym: 'θ',   desc: 'Angulo do pendulo',   lim: '±12°',     col: 'border-green-700 text-green-400' },
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

      {/* 2. Laco fechado */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">2. Laco de Controle Fechado</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          A cada 20 ms o estado e medido, passado ao actor DDPG, que decide a forca <strong className="text-yellow-300">continua</strong>,
          que altera o estado — ciclo de retroalimentacao continuo.
        </p>
        <ClosedLoopDiagram />
        <div className={CODE}>
          {`# Loop de controle — 50 Hz\nwhile not done:\n    s_t = env.state               # [x, xd, theta, thd]\n    F   = actor(s_t)              # tanh(NN(s)) * 10 N\n    s_t1, r, done = env.step(F)   # avanca fisica dt=0.02s`}
        </div>
      </div>

      {/* 3. Arquitetura DDPG */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">3. Arquitetura DDPG</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          O DDPG usa duas redes: o <strong className="text-green-400">Actor</strong> (politica) decide a forca,
          e o <strong className="text-blue-400">Critic</strong> avalia a qualidade da decisao. Ambas tem
          redes-alvo para estabilidade.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-900 rounded-xl p-4 border border-green-800">
            <p className="text-green-400 font-semibold text-sm mb-2">Actor (Politica)</p>
            <div className={CODE}>
              {`estado(4)\n  → Linear(4, 24) + ReLU\n  → Linear(24, 24) + ReLU\n  → Linear(24, 1) + Tanh\n  → × 10  →  F ∈ [-10, +10] N`}
            </div>
            <p className="text-gray-500 text-xs mt-2">
              A funcao tanh garante saida em [-1, 1]. Multiplicando por 10 obtemos a forca.
            </p>
          </div>
          <div className="bg-gray-900 rounded-xl p-4 border border-blue-800">
            <p className="text-blue-400 font-semibold text-sm mb-2">Critic (Q-value)</p>
            <div className={CODE}>
              {`(estado(4), acao(1)) → concat(5)\n  → Linear(5, 24) + ReLU\n  → Linear(24, 24) + ReLU\n  → Linear(24, 1)  →  Q(s, a)`}
            </div>
            <p className="text-gray-500 text-xs mt-2">
              Avalia: se estou no estado s e aplico forca a, qual o retorno total esperado?
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-400">
          {[
            { t: 'Replay Buffer', d: '100k transicoes (s, F, r, s\'). Amostragem aleatoria quebra correlacao temporal.' },
            { t: 'Target Networks', d: 'Copias do Actor e Critic. Soft-update: τ=0.005 a cada passo.' },
            { t: 'OU Noise', d: 'Ornstein-Uhlenbeck (σ=0.3). Ruido temporalmente correlacionado para exploracao suave.' },
          ].map(c => (
            <div key={c.t} className="bg-gray-900 rounded-lg p-3">
              <p className="text-white font-semibold mb-1">{c.t}</p>
              <p>{c.d}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. PID vs DDPG */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">4. Controle Classico vs DDPG</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          O PID e o controlador industrial mais usado. Comparar com DDPG revela as trocas entre
          interpretabilidade e capacidade de representacao. Ambos produzem <strong className="text-yellow-300">forca continua</strong>.
        </p>
        <PIDvsDDPG />
      </div>

      {/* 5. Simulacao interativa */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">5. Simulacao Interativa</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          Na pagina de <strong className="text-white">Simulacao</strong>, a fisica roda em tempo real no browser.
          Os pesos do actor sao carregados como JSON e a inferencia e feita em JavaScript puro —
          nenhum servidor necessario. Voce pode:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {[
            { t: 'Arrastar o carrinho', d: 'Clique e arraste para aplicar perturbacao. O DDPG reage instantaneamente.', c: 'border-blue-700 text-blue-300' },
            { t: 'Modo manual', d: 'Use as setas do teclado ou o slider para controlar voce mesmo. Compare com o DDPG.', c: 'border-purple-700 text-purple-300' },
            { t: 'Sem controle', d: 'Observe o pendulo cair livremente. Evidencia a instabilidade do sistema.', c: 'border-red-700 text-red-300' },
          ].map(c => (
            <div key={c.t} className={`bg-gray-900 rounded-lg p-3 border ${c.c.split(' ')[0]}`}>
              <p className={`font-semibold mb-1 ${c.c.split(' ')[1]}`}>{c.t}</p>
              <p className="text-gray-400">{c.d}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 6. Discreto vs Continuo */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">6. Por que Controle Continuo?</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
            <p className="text-red-400 font-semibold text-sm mb-2">DQN Discreto (antigo)</p>
            <div className={CODE}>
              {`F ∈ {-10, -5, 0, +5, +10}  // 5 niveis\na* = argmax Q(s, a)        // indice`}
            </div>
            <ul className="mt-3 space-y-1 text-gray-400">
              <li>&#10007; Forca quantizada — sem modulacao fina</li>
              <li>&#10007; Saltos bruscos de 5 N entre niveis</li>
              <li>&#10007; Irrealista — atuadores reais sao continuos</li>
            </ul>
          </div>
          <div className="bg-gray-900 rounded-xl p-4 border border-green-800">
            <p className="text-green-400 font-semibold text-sm mb-2">DDPG Continuo (atual)</p>
            <div className={CODE}>
              {`F ∈ [-10, +10]  // continuo\nF = tanh(NN(s)) * 10  // suave`}
            </div>
            <ul className="mt-3 space-y-1 text-gray-400">
              <li>&#10003; Forca continua — precisao arbitraria</li>
              <li>&#10003; Transicoes suaves — menos vibracao</li>
              <li>&#10003; Realista — modela atuadores reais</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 7. Por que ESBMC */}
      <div className={CARD}>
        <p className="text-white font-bold text-lg">7. Por que Verificacao Formal?</p>
        <p className="text-gray-300 text-sm leading-relaxed mb-3">
          Simulacao interativa testa casos especificos. ESBMC prova propriedades para
          <strong className="text-white"> todos os estados possiveis</strong> no dominio.
          Com saida continua, verificar que |F| ≤ 10 N e trivial (garantido por tanh),
          mas propriedades de seguranca (o pendulo nunca cai?) requerem analise formal.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {[
            { t: 'Simulacao',      c: 'border-gray-600 text-gray-300',
              items: ['✓ Interativa', '✓ Tempo real', '✗ So casos especificos', '✗ Sem garantia geral'] },
            { t: 'Testes',         c: 'border-blue-800 text-blue-300',
              items: ['✓ Automatizados', '✓ Rapidos', '✗ Cobertura limitada', '✗ Espaco continuo'] },
            { t: 'ESBMC',          c: 'border-green-700 text-green-400',
              items: ['✓ Prova para todos estados', '✓ Contraexemplo concreto', '✓ Sound (sem falsos neg.)', '✗ Mais lento'] },
          ].map(c => (
            <div key={c.t} className={`bg-gray-900 rounded-lg p-3 border ${c.c.split(' ')[0]}`}>
              <p className={`font-semibold mb-2 ${c.c.split(' ')[1]}`}>{c.t}</p>
              <ul className="space-y-1 text-gray-400">
                {c.items.map(it => <li key={it}>{it}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
