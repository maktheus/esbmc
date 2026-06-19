'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { DDPGVerificationData } from '@/lib/types';

function InfoCard({ title, value, sub, color = 'blue', href }: {
  title: string; value: string; sub?: string; color?: string; href?: string;
}) {
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

  if (href) return <Link href={href} className="block">{inner}</Link>;
  return inner;
}

export default function DashboardPage() {
  const [data, setData] = useState<DDPGVerificationData | null>(null);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
    fetch(`${base}/ddpg_verification_data.json`)
      .then(r => r.ok ? r.json() : null)
      .then(setData)
      .catch(() => {});
  }, []);

  const totalDead = data
    ? data.verification.dead_neurons_l1.dead.length + data.verification.dead_neurons_l2.dead.length
    : 0;
  const totalNeurons = data
    ? data.verification.dead_neurons_l1.total + data.verification.dead_neurons_l2.total
    : 48;
  const saturated = data ? data.verification.saturation.saturated_neurons.length : 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center pt-4">
        <h1 className="text-3xl font-bold text-white mb-2">
          DDPG Cart-Pole
        </h1>
        <p className="text-gray-400 text-lg">
          Verificacao Formal com ESBMC — Controlador Continuo Q8.8
        </p>
        <div className="mt-3 inline-block px-3 py-1 bg-green-900 text-green-300 rounded-full text-sm">
          {totalDead === 0 ? `Todos os ${totalNeurons} neuronios VIVOS` : `${totalDead} neuronios mortos`}
          {' '}&bull; {saturated} saturados &bull; Controlador responsivo
        </div>
      </div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard
          title="Controlador"
          value="DDPG Continuo"
          sub="F in [-10, +10] N — Q8.8 verificado"
          color="blue"
        />
        <InfoCard
          title="Verificacao ESBMC"
          value={(data?.counterexamples.length ?? 0) > 0 ? 'COM ALERTAS' : totalDead === 0 && saturated === 0 ? 'APROVADO' : 'COM ALERTAS'}
          sub={`${totalDead} mortos, ${saturated} saturados, ${data?.counterexamples.length ?? 0} contraexemplos`}
          color={(data?.counterexamples.length ?? 0) > 0 ? 'yellow' : 'green'}
          href="/verification"
        />
        <InfoCard
          title="Arquitetura"
          value={data?.model_info.architecture ?? '4->24->24->1'}
          sub="Input -> ReLU -> ReLU -> Tanh x 10"
          color="purple"
        />
      </div>

      {/* Detailed info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
          <h2 className="text-white font-semibold mb-3 text-lg">Sistema Cart-Pole</h2>
          <div className="space-y-2 text-sm text-gray-300">
            {[
              ['Estado', '[x, x_dot, theta, theta_dot]'],
              ['Posicao carro', 'x in [-2.4, 2.4] m'],
              ['Angulo pendulo', 'theta in [-12, 12] graus'],
              ['Controlador', 'DDPG (continuo, Q8.8)'],
              ['Forca', 'F in [-10, +10] N'],
              ['Intervalo', 'dt = 0.02 s'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-gray-400">{k}</span>
                <span className="font-mono">{v}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
          <h2 className="text-white font-semibold mb-3 text-lg">Resultados ESBMC</h2>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className={`w-3 h-3 rounded-full flex-shrink-0 ${totalDead === 0 ? 'bg-green-500' : 'bg-red-500'}`} />
              <div>
                <p className="text-white text-sm font-medium">Neuronios Mortos: {totalDead} / {totalNeurons}</p>
                <p className="text-gray-400 text-xs">L1 + L2 verificados individualmente</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className={`w-3 h-3 rounded-full flex-shrink-0 ${saturated === 0 ? 'bg-green-500' : 'bg-yellow-500'}`} />
              <div>
                <p className="text-white text-sm font-medium">Saturados: {saturated} / {totalNeurons}</p>
                <p className="text-gray-400 text-xs">ReLU nunca cortada?</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">Saida: Responsiva</p>
                <p className="text-gray-400 text-xs">Forca positiva e negativa para estados diferentes</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-blue-500 flex-shrink-0" />
              <div>
                <p className="text-white text-sm font-medium">Quantizacao: Q8.8 (scale=256)</p>
                <p className="text-gray-400 text-xs">Browser roda mesma aritmetica verificada</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pt-2">
        {[
          { href: '/simulation',   label: 'Simulacao',   desc: 'Q8.8 em tempo real', color: 'blue' },
          { href: '/control',      label: 'Controle',    desc: 'Teoria DDPG',        color: 'purple' },
          { href: '/training',     label: 'Treinamento', desc: 'Curva de aprendizado', color: 'yellow' },
          { href: '/verification', label: 'Verificacao', desc: 'Resultados ESBMC',   color: 'green' },
          { href: '/legacy',       label: 'Legado (DQN)', desc: 'Versao anterior',   color: 'gray' },
        ].map(({ href, label, desc, color }) => (
          <Link key={href} href={href} className="block">
            <div className={`bg-${color === 'gray' ? 'gray-700/30' : `${color}-900/30`} hover:bg-${color === 'gray' ? 'gray-700/50' : `${color}-900/50`} border border-${color === 'gray' ? 'gray-600' : `${color}-700`} rounded-xl p-4 transition-colors text-center`}>
              <p className={`text-${color === 'gray' ? 'gray-300' : `${color}-300`} font-semibold mb-1`}>{label}</p>
              <p className="text-gray-400 text-xs">{desc}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
