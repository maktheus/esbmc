'use client';

import { useState, useEffect } from 'react';
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
  title: string;
  prop: ClosedLoopProperty;
  description: string;
}) {
  const isFailed    = prop.result === 'FAILED';
  const isSuccess   = prop.result === 'SUCCESSFUL';
  const isTimeout   = prop.result === 'TIMEOUT';
  const isUnknown   = !isFailed && !isSuccess && !isTimeout;

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
          {isFailed ? '✗ FALHA ENCONTRADA' : isSuccess ? '✓ VERIFICADO' : isTimeout ? '⏱ TIMEOUT' : prop.result}
        </span>
      </div>

      <p className="text-gray-400 text-xs">{description}</p>

      {isFailed && prop.counterexample && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3">
          <p className="text-red-300 text-xs font-semibold mb-1">
            FALHA ENCONTRADA — estado que causa colapso:
          </p>
          <p className="text-red-200 text-xs font-mono break-all">{prop.counterexample}</p>
        </div>
      )}

      {isSuccess && (
        <p className="text-green-400 text-xs">
          Propriedade satisfeita para todo estado no domínio verificado.
        </p>
      )}

      {isTimeout && (
        <p className="text-yellow-400 text-xs">
          ESBMC excedeu o limite de tempo. A propriedade pode ser verdadeira ou falsa — não conclusivo.
        </p>
      )}
    </div>
  );
}

export default function VerificationPage() {
  const [data,  setData]  = useState<SimulationData | null>(null);
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

  const { dead_neurons, saturation } = data.verification;
  const saturatedSet = new Set(saturation.saturated_neurons);
  const deadSet      = new Set(dead_neurons.dead);
  const cl           = data.closed_loop_verification;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Verificação Formal — ESBMC</h1>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className={`rounded-xl p-5 border ${dead_neurons.dead.length === 0
          ? 'bg-green-900/20 border-green-700'
          : 'bg-red-900/20 border-red-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Neurônios Mortos</p>
          <p className={`text-3xl font-bold ${dead_neurons.dead.length === 0 ? 'text-green-400' : 'text-red-400'}`}>
            {dead_neurons.dead.length} / {dead_neurons.total}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            {dead_neurons.dead.length === 0 ? 'Todos contribuem para a saída' : 'Candidatos a poda'}
          </p>
        </div>

        <div className={`rounded-xl p-5 border ${saturation.saturated_neurons.length === 0
          ? 'bg-green-900/20 border-green-700'
          : 'bg-yellow-900/20 border-yellow-700'}`}>
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Neurônios Saturados</p>
          <p className={`text-3xl font-bold ${saturation.saturated_neurons.length === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
            {saturation.saturated_neurons.length} / {dead_neurons.total}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            {saturation.saturated_neurons.length === 0 ? 'Nenhum sempre ativo' : 'ReLU nunca corta'}
          </p>
        </div>

        <div className="bg-green-900/20 border border-green-700 rounded-xl p-5">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Saturação de Ação</p>
          <p className="text-3xl font-bold text-green-400">NORMAL</p>
          <p className="text-gray-400 text-xs mt-1">Controlador responsivo</p>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 text-sm text-gray-300 space-y-2">
        <h2 className="text-white font-semibold text-base mb-3">Como interpretar os resultados</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <p><span className="text-green-400 font-bold">VERIFICATION FAILED</span> → neurônio <span className="text-green-300 font-bold">VIVO</span></p>
            <p className="text-gray-500 text-xs pl-3">ESBMC encontrou uma entrada no domínio que ativa o neurônio. Isso é o comportamento esperado — o neurônio contribui para a decisão.</p>
          </div>
          <div className="space-y-2">
            <p><span className="text-red-400 font-bold">VERIFICATION SUCCESSFUL</span> → neurônio <span className="text-red-300 font-bold">MORTO</span></p>
            <p className="text-gray-500 text-xs pl-3">ESBMC provou que o neurônio nunca ativa para nenhuma entrada válida. Indica problema no treinamento — candidato a poda.</p>
          </div>
        </div>
        <div className="pt-2 border-t border-gray-700">
          <p className="text-gray-400 text-xs">
            <span className="text-blue-400 font-mono">Domínio verificado:</span>{' '}
            x ∈ [−2.4, 2.4] m &bull; ẋ ∈ [−5, 5] m/s &bull; θ ∈ [−12°, 12°] &bull; θ̇ ∈ [−5, 5] rad/s &bull;
            Quantização: Q8.8 (scale=256) &bull; Solver: Boolector
          </p>
        </div>
      </div>

      {/* Neuron table */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-700">
          <h2 className="text-white font-semibold">Camada 1 — {dead_neurons.total} Neurônios</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                <th className="text-left px-4 py-2">Neurônio</th>
                <th className="text-right px-4 py-2">Viés (Q8.8)</th>
                <th className="text-right px-4 py-2">Viés (float)</th>
                <th className="text-center px-4 py-2">Neurônio Morto?</th>
                <th className="text-center px-4 py-2">Saturação?</th>
              </tr>
            </thead>
            <tbody>
              {dead_neurons.neurons.map((n: NeuronInfo) => {
                const isDead = deadSet.has(n.id);
                const isSat  = saturatedSet.has(n.id);
                return (
                  <tr key={n.id} className={`border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors ${isDead ? 'bg-red-900/10' : ''}`}>
                    <td className="px-4 py-2 font-mono text-gray-300">h₁[{n.id}]</td>
                    <td className="px-4 py-2 text-right font-mono text-gray-300">{n.bias_q88}</td>
                    <td className="px-4 py-2 text-right font-mono text-gray-400">
                      {(n.bias_q88 / 256).toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-center"><Badge ok={!isDead} /></td>
                    <td className="px-4 py-2 text-center"><SatBadge sat={isSat} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Saturation of output */}
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <h2 className="text-white font-semibold mb-3">Saturação de Ação (Saída)</h2>
        <div className="flex items-start gap-3">
          <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0 mt-1" />
          <div>
            <p className="text-white font-medium">{saturation.output_status}</p>
            <p className="text-gray-400 text-sm mt-1">
              Propriedade verificada: Q[0] {'>'} Q[1] para todo estado E Q[1] {'>'} Q[0] para todo estado.
              Ambas retornam VERIFICATION FAILED → o controlador escolhe ações diferentes
              dependendo do estado — comportamento correto para um controlador responsivo.
            </p>
          </div>
        </div>
      </div>

      {/* ── Closed-loop verification section ── */}
      {cl && (
        <div className="space-y-4">
          <div className="border-t border-gray-700 pt-4">
            <h2 className="text-xl font-bold text-white mb-1">Verificação em Malha Fechada</h2>
            <p className="text-gray-400 text-sm">
              O ESBMC verifica propriedades do sistema completo: controlador DQN + dinâmica do Cart-Pole.
              A passagem do DQN é completamente expandida (sem loops) com todos os pesos hardcoded
              em Q8.8 (scale=256). Bounds de pré-ativação são derivados por aritmética de intervalo.
            </p>
          </div>

          <ClosedLoopCard
            title="Property A — Direção Errada (θ > 0.10 rad → push right)"
            prop={cl.property_a_right}
            description={
              'Quando o pêndulo está inclinado para a direita (θ > 0.10 rad) e ainda acelerando nessa ' +
              'direção (θ̇ ≥ 0), o controlador DEVE aplicar força para a direita (ação=1) para restaurar ' +
              'o equilíbrio. Se aplicar força para a esquerda, é uma falha de controle.'
            }
          />

          <ClosedLoopCard
            title="Property A — Direção Errada (θ < −0.10 rad → push left)"
            prop={cl.property_a_left}
            description={
              'Versão simétrica: quando θ < −0.10 rad e θ̇ ≤ 0, o controlador DEVE ' +
              'aplicar força para a esquerda (ação=0). ESBMC FAILED significa que ' +
              'encontrou um estado onde o controlador empurra na direção errada.'
            }
          />

          <ClosedLoopCard
            title="Property B — Segurança em 1 Passo (dinâmica linearizada)"
            prop={cl.property_b_safety}
            description={
              'Para todo estado inicial s₀ ∈ S_safe, o controlador DQN gera uma ação e ' +
              'a dinâmica linearizada (sin≈θ, cos≈1) avança um passo. A propriedade garante ' +
              'que θ₁ ∈ [−12°, 12°] após o passo. ESBMC FAILED revela estados onde o ' +
              'sistema controlado não garante segurança em 1 passo.'
            }
          />

          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-xs text-gray-400 space-y-1">
            <p className="text-gray-300 font-semibold text-sm mb-2">Notas metodológicas</p>
            <p>• Harnesses C gerados automaticamente com todos os pesos expandidos (sem loops C) para evitar problemas de desdobramento no ESBMC.</p>
            <p>• <span className="font-mono text-blue-300">__ESBMC_assume</span> restringe pré-ativações ao intervalo derivado analiticamente (aritmética de intervalo), guiando o solver de forma eficiente.</p>
            <p>• Dinâmica de Property B usa linearização válida para |θ| ≤ 12°: sin(θ)≈θ, cos(θ)≈1.</p>
            <p>• Solver: Boolector. Domínio Q8.8 (scale=256). Limiar de perigo: |θ| &gt; 25/256 ≈ 5.6°.</p>
          </div>
        </div>
      )}
    </div>
  );
}
