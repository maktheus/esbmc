'use client';

import Link from 'next/link';

interface CardProps {
  title:    string;
  value:    string;
  sub?:     string;
  color?:   string;
  href?:    string;
}

function InfoCard({ title, value, sub, color = 'blue', href }: CardProps) {
  const colorMap: Record<string, string> = {
    blue:   'border-blue-500  text-blue-400',
    green:  'border-green-500 text-green-400',
    purple: 'border-purple-500 text-purple-400',
    yellow: 'border-yellow-500 text-yellow-400',
  };
  const cls = colorMap[color] ?? colorMap.blue;

  const inner = (
    <div className={`bg-gray-800 rounded-xl border-l-4 p-5 ${cls.split(' ')[0]} h-full`}>
      <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">{title}</p>
      <p className={`text-2xl font-bold ${cls.split(' ')[1]}`}>{value}</p>
      {sub && <p className="text-gray-400 text-sm mt-1">{sub}</p>}
      {href && (
        <span className="inline-block mt-3 text-xs text-gray-500 hover:text-gray-300 underline underline-offset-2">
          Ver detalhes &rarr;
        </span>
      )}
    </div>
  );

  if (href) {
    return <Link href={href} className="block">{inner}</Link>;
  }
  return inner;
}

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center pt-4">
        <h1 className="text-3xl font-bold text-white mb-2">
          DQN Cart-Pole
        </h1>
        <p className="text-gray-400 text-lg">
          Verificacao Formal com ESBMC
        </p>
        <div className="mt-3 inline-block px-3 py-1 bg-green-900 text-green-300 rounded-full text-sm">
          Todos os 24 neuronios VIVOS &bull; 0 saturados &bull; Controlador responsivo
        </div>
      </div>

      {/* Cards principais */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard
          title="Treinamento"
          value="404 episodios"
          sub="Score medio final: 471 passos"
          color="blue"
        />
        <InfoCard
          title="Verificacao ESBMC"
          value="APROVADO"
          sub="0 neuronios mortos, 0 saturados"
          color="green"
          href="/verification"
        />
        <InfoCard
          title="Arquitetura"
          value="4 → 24 → 24 → 2"
          sub="Input ReLU ReLU Output"
          color="purple"
        />
      </div>

      {/* Grid de informacoes detalhadas */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* Descricao do sistema */}
        <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
          <h2 className="text-white font-semibold mb-3 text-lg">Sistema Cart-Pole</h2>
          <div className="space-y-2 text-sm text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Estado</span>
              <span className="font-mono">[x, x_dot, theta, theta_dot]</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Posicao carro</span>
              <span className="font-mono">x &#8712; [-2.4, 2.4] m</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Angulo pendulo</span>
              <span className="font-mono">&#952; &#8712; [-12&deg;, 12&deg;]</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Acoes</span>
              <span className="font-mono">0=esquerda / 1=direita</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Forca</span>
              <span className="font-mono">&#177;10 N</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Intervalo de tempo</span>
              <span className="font-mono">dt = 0.02 s</span>
            </div>
          </div>
        </div>

        {/* Resultados de verificacao */}
        <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
          <h2 className="text-white font-semibold mb-3 text-lg">Resultados ESBMC</h2>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0"></span>
              <div>
                <p className="text-white text-sm font-medium">Neuronios Mortos: 0 / 24</p>
                <p className="text-gray-400 text-xs">Todos os neuronios contribuem para a saida</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0"></span>
              <div>
                <p className="text-white text-sm font-medium">Neuronios Saturados: 0 / 24</p>
                <p className="text-gray-400 text-xs">Nenhum neuronio sempre ativo (ReLU nunca cortada)</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0"></span>
              <div>
                <p className="text-white text-sm font-medium">Acao: Responsiva</p>
                <p className="text-gray-400 text-xs">Controlador escolhe acoes diferentes para estados diferentes</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-blue-500 flex-shrink-0"></span>
              <div>
                <p className="text-white text-sm font-medium">Dominio: Q8.8 (scale=256)</p>
                <p className="text-gray-400 text-xs">Verificacao em aritmetica inteira de ponto fixo</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Links de navegacao */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        <Link href="/simulation" className="block">
          <div className="bg-blue-900/30 hover:bg-blue-900/50 border border-blue-700 rounded-xl p-5 transition-colors text-center">
            <p className="text-blue-300 font-semibold text-lg mb-1">Ver Simulacao</p>
            <p className="text-gray-400 text-sm">Animacao Cart-Pole com controle DQN em tempo real</p>
          </div>
        </Link>
        <Link href="/verification" className="block">
          <div className="bg-green-900/30 hover:bg-green-900/50 border border-green-700 rounded-xl p-5 transition-colors text-center">
            <p className="text-green-300 font-semibold text-lg mb-1">Ver Verificacao</p>
            <p className="text-gray-400 text-sm">Resultados detalhados da verificacao formal ESBMC</p>
          </div>
        </Link>
      </div>
    </div>
  );
}
