'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SimulationData, NeuronInfo, ClosedLoopProperty } from '@/lib/types';

function Badge({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${
      ok ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
    }`}>
      {ok ? '✓ VIVO' : '✗ MORTO'}
    </span>
  );
}

function SatBadge({ sat }: { sat: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${
      !sat ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
    }`}>
      {sat ? '⚠ SATURADO' : '✓ NORMAL'}
    </span>
  );
}

function ClosedLoopCard({ title, prop, description }: {
  title: string; prop: ClosedLoopProperty; description: string;
}) {
  const isFailed  = prop.result === 'FAILED';
  const isSuccess = prop.result === 'SUCCESSFUL';
  const isTimeout = prop.result === 'TIMEOUT';

  return (
    <div className={`rounded-xl border p-5 space-y-3 ${
      isFailed  ? 'bg-red-900/20 border-red-700'
    : isSuccess ? 'bg-green-900/20 border-green-700'
    : isTimeout ? 'bg-yellow-900/20 border-yellow-700'
    :             'bg-gray-800 border-gray-700'
    }`}>
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-white font-semibold text-sm">{title}</h3>
        <span className={`flex-shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${
          isFailed  ? 'bg-red-800 text-red-200'
        : isSuccess ? 'bg-green-800 text-green-200'
        : isTimeout ? 'bg-yellow-800 text-yellow-200'
        :             'bg-gray-700 text-gray-300'
        }`}>
          {isFailed ? '✗ FALHA' : isSuccess ? '✓ VERIFICADO' : isTimeout ? '⏱ TIMEOUT' : prop.result}
        </span>
      </div>
      <p className="text-gray-400 text-xs">{description}</p>
      {isFailed && prop.counterexample && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3">
          <p className="text-red-300 text-xs font-semibold mb-1">Contraexemplo:</p>
          <p className="text-red-200 text-xs font-mono break-all">{prop.counterexample}</p>
        </div>
      )}
      {isSuccess && <p className="text-green-400 text-xs">Propriedade satisfeita para todo estado.</p>}
      {isTimeout && <p className="text-yellow-400 text-xs">ESBMC excedeu o limite de tempo.</p>}
    </div>
  );
}

export default function LegacyPage() {
  const [data, setData]   = useState<SimulationData | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
    fetch(`${base}/simulation_data.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setError(String(e)));
  }, []);

  if (error) return (
    <div className="text-center py-20 text-red-400">
      <p className="text-xl mb-2">Erro ao carregar dados legado</p>
      <p className="text-sm font-mono">{error}</p>
    </div>
  );

  if (!data) return (
    <div className="text-center py-20 text-gray-400">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-400 mx-auto mb-4" />
      Carregando...
    </div>
  );

  const { dead_neurons, saturation } = data.verification;
  const saturatedSet = new Set(saturation.saturated_neurons);
  const deadSet      = new Set(dead_neurons.dead);
  const cl           = data.closed_loop_verification;

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-yellow-900/30 border border-yellow-700 rounded-xl p-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <p className="text-yellow-300 font-bold text-lg">Versao Anterior — Controlador DQN Discreto</p>
          <p className="text-yellow-400/70 text-sm">5 acoes discretas: [-10, -5, 0, +5, +10] N. Substituido pelo DDPG continuo.</p>
        </div>
        <Link href="/" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
          Voltar ao DDPG
        </Link>
      </div>

      <h1 className="text-2xl font-bold text-white">Verificacao Formal — DQN (Legado)</h1>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className={`rounded-xl p-5 border ${dead_neurons.dead.length === 0
          ? 'bg-green-900/20 border-green-700' : 'bg-red-900/20 border-red-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Neuronios Mortos</p>
          <p className={`text-3xl font-bold ${dead_neurons.dead.length === 0 ? 'text-green-400' : 'text-red-400'}`}>
            {dead_neurons.dead.length} / {dead_neurons.total}
          </p>
        </div>
        <div className={`rounded-xl p-5 border ${saturation.saturated_neurons.length === 0
          ? 'bg-green-900/20 border-green-700' : 'bg-yellow-900/20 border-yellow-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Neuronios Saturados</p>
          <p className={`text-3xl font-bold ${saturation.saturated_neurons.length === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
            {saturation.saturated_neurons.length} / {dead_neurons.total}
          </p>
        </div>
        <div className="bg-blue-900/20 border border-blue-700 rounded-xl p-5">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Controlador</p>
          <p className="text-2xl font-bold text-blue-400">DQN 5-acoes</p>
          <p className="text-gray-400 text-xs mt-1">{data.model_info.architecture}</p>
        </div>
      </div>

      {/* Neuron table */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-700">
          <h2 className="text-white font-semibold">Camada 1 — {dead_neurons.total} Neuronios</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                <th className="text-left px-4 py-2">Neuronio</th>
                <th className="text-right px-4 py-2">Vies (Q8.8)</th>
                <th className="text-right px-4 py-2">Vies (float)</th>
                <th className="text-center px-4 py-2">Morto?</th>
                <th className="text-center px-4 py-2">Saturacao?</th>
              </tr>
            </thead>
            <tbody>
              {dead_neurons.neurons.map((n: NeuronInfo) => {
                const isDead = deadSet.has(n.id);
                const isSat  = saturatedSet.has(n.id);
                return (
                  <tr key={n.id} className={`border-b border-gray-700/50 hover:bg-gray-700/30 ${isDead ? 'bg-red-900/10' : ''}`}>
                    <td className="px-4 py-2 font-mono text-gray-300">h1[{n.id}]</td>
                    <td className="px-4 py-2 text-right font-mono text-gray-300">{n.bias_q88}</td>
                    <td className="px-4 py-2 text-right font-mono text-gray-400">{(n.bias_q88 / 256).toFixed(4)}</td>
                    <td className="px-4 py-2 text-center"><Badge ok={!isDead} /></td>
                    <td className="px-4 py-2 text-center"><SatBadge sat={isSat} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Output saturation */}
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <h2 className="text-white font-semibold mb-3">Saturacao de Acao (Saida)</h2>
        <div className="flex items-start gap-3">
          <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0 mt-1" />
          <div>
            <p className="text-white font-medium">{saturation.output_status}</p>
            <p className="text-gray-400 text-sm mt-1">
              Q[0] {'>'} Q[1] e Q[1] {'>'} Q[0] ambos FAILED → controlador responsivo.
            </p>
          </div>
        </div>
      </div>

      {/* Closed-loop */}
      {cl && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-white">Verificacao Malha Fechada (DQN)</h2>
          <ClosedLoopCard
            title="Property A — Direcao (θ > 0.10 rad → push right)"
            prop={cl.property_a_right}
            description="Controlador DQN deve escolher acao=1 quando pendulo inclina para direita."
          />
          <ClosedLoopCard
            title="Property A — Direcao (θ < −0.10 rad → push left)"
            prop={cl.property_a_left}
            description="Simetrico: controlador deve escolher acao=0 quando pendulo inclina para esquerda."
          />
          <ClosedLoopCard
            title="Property B — Seguranca 1 Passo"
            prop={cl.property_b_safety}
            description="Apos 1 passo de dinamica linearizada, theta permanece na regiao segura?"
          />
        </div>
      )}

      {/* Methodology note */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-xs text-gray-400 space-y-1">
        <p className="text-gray-300 font-semibold text-sm mb-2">Notas</p>
        <p>Controlador DQN discreto com 5 niveis de forca: [-10, -5, 0, +5, +10] N</p>
        <p>Quantizacao Q8.8 (scale=256). Solver Boolector. Timeout 120s por propriedade.</p>
        <p>Harnesses C gerados sem loops — todos os pesos expandidos inline.</p>
      </div>
    </div>
  );
}
