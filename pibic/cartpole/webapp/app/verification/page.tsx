'use client';

import { useState, useEffect } from 'react';
import { DDPGVerificationData, NeuronInfo, NeuronVerification, ClosedLoopProperty } from '@/lib/types';

function Badge({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${
      ok ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
    }`}>
      {ok ? 'VIVO' : 'MORTO'}
    </span>
  );
}

function SatBadge({ sat }: { sat: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${
      !sat ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
    }`}>
      {sat ? 'SATURADO' : 'NORMAL'}
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
          {isFailed ? 'FALHA ENCONTRADA' : isSuccess ? 'VERIFICADO' : isTimeout ? 'TIMEOUT' : prop.result}
        </span>
      </div>
      <p className="text-gray-400 text-xs">{description}</p>
      {isFailed && prop.counterexample && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3">
          <p className="text-red-300 text-xs font-semibold mb-1">Contraexemplo:</p>
          <p className="text-red-200 text-xs font-mono break-all">{prop.counterexample}</p>
        </div>
      )}
      {isSuccess && <p className="text-green-400 text-xs">Propriedade satisfeita para todo estado no dominio verificado.</p>}
      {isTimeout && <p className="text-yellow-400 text-xs">ESBMC excedeu o limite de tempo (120s). Resultado inconclusivo.</p>}
    </div>
  );
}

function NeuronTable({ title, neurons, deadSet, saturatedSet }: {
  title: string; neurons: NeuronVerification; deadSet: Set<number>; saturatedSet: Set<number>;
}) {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-700">
        <h2 className="text-white font-semibold">{title} — {neurons.total} Neuronios</h2>
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
            {neurons.neurons.map((n: NeuronInfo) => {
              const isDead = deadSet.has(n.id);
              const isSat  = saturatedSet.has(n.id);
              return (
                <tr key={n.id} className={`border-b border-gray-700/50 hover:bg-gray-700/30 ${isDead ? 'bg-red-900/10' : ''}`}>
                  <td className="px-4 py-2 font-mono text-gray-300">[{n.id}]</td>
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
  );
}

export default function VerificationPage() {
  const [data, setData]   = useState<DDPGVerificationData | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
    fetch(`${base}/ddpg_verification_data.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setError(String(e)));
  }, []);

  if (error) return (
    <div className="text-center py-20 text-red-400">
      <p className="text-xl mb-2">Erro ao carregar dados</p>
      <p className="text-sm font-mono">{error}</p>
    </div>
  );

  if (!data) return (
    <div className="text-center py-20 text-gray-400">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-400 mx-auto mb-4" />
      Carregando...
    </div>
  );

  const { dead_neurons_l1, dead_neurons_l2, saturation } = data.verification;
  const saturatedSet = new Set(saturation.saturated_neurons);
  const deadSetL1    = new Set(dead_neurons_l1.dead);
  const deadSetL2    = new Set(dead_neurons_l2.dead);
  const cl           = data.closed_loop_verification;
  const totalDead    = dead_neurons_l1.dead.length + dead_neurons_l2.dead.length;
  const totalNeurons = dead_neurons_l1.total + dead_neurons_l2.total;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Verificacao Formal — DDPG Q8.8</h1>
        <p className="text-gray-400 text-sm mt-1">
          Controlador {data.model_info.controller_type} — {data.model_info.architecture} — {data.model_info.quantization}
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className={`rounded-xl p-5 border ${totalDead === 0
          ? 'bg-green-900/20 border-green-700' : 'bg-red-900/20 border-red-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Neuronios Mortos</p>
          <p className={`text-3xl font-bold ${totalDead === 0 ? 'text-green-400' : 'text-red-400'}`}>
            {totalDead} / {totalNeurons}
          </p>
          <p className="text-gray-400 text-xs mt-1">L1: {dead_neurons_l1.dead.length} | L2: {dead_neurons_l2.dead.length}</p>
        </div>

        <div className={`rounded-xl p-5 border ${saturation.saturated_neurons.length === 0
          ? 'bg-green-900/20 border-green-700' : 'bg-yellow-900/20 border-yellow-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Saturados</p>
          <p className={`text-3xl font-bold ${saturation.saturated_neurons.length === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
            {saturation.saturated_neurons.length} / {totalNeurons}
          </p>
          <p className="text-gray-400 text-xs mt-1">Nenhum sempre ativo</p>
        </div>

        <div className="bg-green-900/20 border border-green-700 rounded-xl p-5">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Saida</p>
          <p className="text-2xl font-bold text-green-400">{saturation.output_status}</p>
          <p className="text-gray-400 text-xs mt-1">Controlador responsivo</p>
        </div>

        <div className="bg-blue-900/20 border border-blue-700 rounded-xl p-5">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Quantizacao</p>
          <p className="text-2xl font-bold text-blue-400">Q8.8</p>
          <p className="text-gray-400 text-xs mt-1">scale=256, Boolector</p>
        </div>
      </div>

      {/* How to interpret */}
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 text-sm text-gray-300 space-y-2">
        <h2 className="text-white font-semibold text-base mb-3">Como interpretar</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <p><span className="text-green-400 font-bold">VERIFICATION FAILED</span> = neuronio <span className="text-green-300 font-bold">VIVO</span></p>
            <p className="text-gray-500 text-xs pl-3">ESBMC encontrou entrada que ativa o neuronio.</p>
          </div>
          <div className="space-y-2">
            <p><span className="text-red-400 font-bold">VERIFICATION SUCCESSFUL</span> = neuronio <span className="text-red-300 font-bold">MORTO</span></p>
            <p className="text-gray-500 text-xs pl-3">Neuronio nunca ativa — candidato a poda.</p>
          </div>
        </div>
        <div className="pt-2 border-t border-gray-700">
          <p className="text-gray-400 text-xs">
            <span className="text-blue-400 font-mono">Dominio:</span>{' '}
            x in [-2.4, 2.4] m | x_dot in [-5, 5] m/s | theta in [-12, 12] graus | theta_dot in [-5, 5] rad/s |
            Q8.8 (scale=256) | Solver: Boolector
          </p>
        </div>
      </div>

      {/* Neuron tables — Domain 1: Model */}
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-white">Dominio 1 — Modelo (Rede Neural)</h2>
        <p className="text-gray-400 text-sm">Propriedades estruturais da rede: neuronios mortos e saturacao.</p>
      </div>

      <NeuronTable title="Camada 1" neurons={dead_neurons_l1} deadSet={deadSetL1} saturatedSet={saturatedSet} />
      <NeuronTable title="Camada 2" neurons={dead_neurons_l2} deadSet={deadSetL2} saturatedSet={new Set()} />

      {/* Output saturation */}
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <h2 className="text-white font-semibold mb-3">Saturacao de Saida (Direcao)</h2>
        <div className="flex items-start gap-3">
          <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0 mt-1" />
          <div>
            <p className="text-white font-medium">{saturation.output_status}</p>
            <p className="text-gray-400 text-sm mt-1">
              Verificacao: z {'>'} 0 sempre e z {'<'} 0 sempre — ambas FAILED.
              O controlador produz forca positiva e negativa dependendo do estado.
            </p>
          </div>
        </div>
      </div>

      {/* Domain 2: Closed-loop */}
      <div className="space-y-2 pt-4 border-t border-gray-700">
        <h2 className="text-xl font-bold text-white">Dominio 2 — Controlador (Malha Fechada)</h2>
        <p className="text-gray-400 text-sm">
          Propriedades do sistema completo: DDPG + dinamica do Cart-Pole.
          Forward pass expandido inline (sem loops) com pesos Q8.8 hardcoded.
          Bounds de pre-ativacao derivados por aritmetica de intervalo.
        </p>
      </div>

      <ClosedLoopCard
        title="Property A — Direcao (theta > 5.6 graus, forca positiva)"
        prop={cl.property_a_right}
        description="Quando o pendulo inclina para a direita (theta > 0.10 rad) e theta_dot >= 0, o controlador deve aplicar forca positiva (F > 0). Usa monotonicidade de tanh: z > 0 implica F > 0."
      />
      <ClosedLoopCard
        title="Property A — Direcao (theta < -5.6 graus, forca negativa)"
        prop={cl.property_a_left}
        description="Simetrico: quando theta < -0.10 rad e theta_dot <= 0, o controlador deve aplicar forca negativa (F < 0)."
      />
      <ClosedLoopCard
        title="Property B — Seguranca 1 Passo (dinamica linearizada)"
        prop={cl.property_b_safety}
        description="Para todo estado inicial seguro, apos 1 passo de dinamica linearizada (sin(theta) aprox theta, cos(theta) aprox 1), theta permanece em [-12, 12] graus. Usa tanh piecewise linear identica ao browser."
      />
      <ClosedLoopCard
        title="Property C — Limites de Forca (|F| <= 10 N)"
        prop={cl.property_c_bounds}
        description="A saida quantizada nunca excede os limites fisicos de forca. Deve ser SUCCESSFUL (tanh garante |output| <= 1 => |F| <= 10)."
      />

      {/* Counterexamples */}
      {data.counterexamples.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-bold text-white">Contraexemplos Encontrados</h2>
          {data.counterexamples.map((ce, i) => (
            <div key={i} className="bg-red-900/20 border border-red-700 rounded-xl p-4 space-y-2">
              <p className="text-red-300 font-semibold text-sm">{ce.property}</p>
              <p className="text-gray-300 text-sm">{ce.description}</p>
              <p className="text-red-200 text-xs font-mono">{ce.state_str}</p>
              <p className="text-gray-400 text-xs">Esperado: {ce.expected_behavior}</p>
            </div>
          ))}
        </div>
      )}

      {/* Methodology notes */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-xs text-gray-400 space-y-1">
        <p className="text-gray-300 font-semibold text-sm mb-2">Notas</p>
        <p>Controlador DDPG continuo com saida tanh x 10 N.</p>
        <p>Quantizacao Q8.8 (scale=256). Solver Boolector. Timeout 120s por propriedade.</p>
        <p>Harnesses C gerados sem loops — todos os pesos expandidos inline.</p>
        <p>Browser roda a mesma aritmetica Q8.8 — contraexemplos reproduzem exatamente.</p>
      </div>
    </div>
  );
}
