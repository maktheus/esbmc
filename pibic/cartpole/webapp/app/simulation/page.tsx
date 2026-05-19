'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import CartPoleCanvas from '@/components/CartPoleCanvas';
import StatePlot      from '@/components/StatePlot';
import { SimulationData, Episode, TrajectoryFrame } from '@/lib/types';

const SPEEDS   = [0.5, 1, 2, 5, 10];
const BASE_FPS = 50;

type Filter = 'all' | 'controlled' | 'random' | 'counterexample';

function TypeBadge({ type }: { type: Episode['type'] }) {
  if (type === 'controlled') {
    return (
      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-bold bg-green-900 text-green-300">
        DQN
      </span>
    );
  }
  if (type === 'counterexample') {
    return (
      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-bold bg-orange-900 text-orange-300">
        ESBMC
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-bold bg-red-900 text-red-300">
      RND
    </span>
  );
}

function EsbmcAlert({ episode, frame }: { episode: Episode; frame: number }) {
  if (episode.type !== 'counterexample') return null;
  const isCritical = frame === (episode.critical_frame ?? 0);
  return (
    <div className={`rounded-xl border p-4 space-y-2 transition-colors ${
      isCritical
        ? 'bg-orange-900/40 border-orange-500 shadow-lg shadow-orange-900/30'
        : 'bg-orange-900/10 border-orange-800'
    }`}>
      <div className="flex items-center gap-2">
        <span className="text-orange-400 font-bold text-sm">
          {isCritical ? '⚠ FALHA ATIVA — Frame crítico' : 'Contraexemplo ESBMC'}
        </span>
        <span className="text-orange-600 text-xs font-mono">{episode.esbmc_property}</span>
      </div>
      <p className="text-orange-200 text-xs leading-relaxed">{episode.esbmc_note}</p>
      {isCritical && (
        <p className="text-orange-300 text-xs font-semibold">
          Este é o estado exato que o ESBMC encontrou. Observe a ação escolhida pelo controlador.
        </p>
      )}
    </div>
  );
}

export default function SimulationPage() {
  const [data,     setData]     = useState<SimulationData | null>(null);
  const [epIdx,    setEpIdx]    = useState(0);
  const [frame,    setFrame]    = useState(0);
  const [playing,  setPlaying]  = useState(false);
  const [speed,    setSpeed]    = useState(1);
  const [error,    setError]    = useState('');
  const [filter,   setFilter]   = useState<Filter>('all');
  const rafRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
    const url = process.env.NEXT_PUBLIC_API_URL
      ? `${process.env.NEXT_PUBLIC_API_URL}/api/simulate`
      : `${base}/simulation_data.json`;

    fetch(url)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setError(String(e)));
  }, []);

  const filteredEps: Episode[] = data?.episodes.filter(ep => {
    if (filter === 'all')            return true;
    if (filter === 'controlled')     return ep.type === 'controlled';
    if (filter === 'random')         return ep.type === 'random';
    if (filter === 'counterexample') return ep.type === 'counterexample';
    return true;
  }) ?? [];

  useEffect(() => {
    if (epIdx >= filteredEps.length && filteredEps.length > 0) {
      setEpIdx(0);
      setFrame(0);
    }
  }, [filter, filteredEps.length, epIdx]);

  const episode: Episode | null = filteredEps[epIdx] ?? null;
  const traj:    TrajectoryFrame[] = episode?.trajectory ?? [];
  const curFrame: TrajectoryFrame  = traj[frame] ?? { x: 0, x_dot: 0, theta: 0, theta_dot: 0, action: 0, q0: 0, q1: 0 };

  const stop = useCallback(() => {
    if (rafRef.current !== null) { clearTimeout(rafRef.current); rafRef.current = null; }
  }, []);

  const tick = useCallback(() => {
    setFrame(f => {
      if (f >= traj.length - 1) { setPlaying(false); return f; }
      return f + 1;
    });
  }, [traj.length]);

  useEffect(() => {
    if (!playing) { stop(); return; }
    const ms = 1000 / (BASE_FPS * speed);
    rafRef.current = setTimeout(function loop() {
      tick();
      rafRef.current = setTimeout(loop, ms);
    }, ms);
    return stop;
  }, [playing, speed, tick, stop]);

  const selectEpisode = (i: number) => {
    stop();
    setPlaying(false);
    setEpIdx(i);
    setFrame(0);
  };

  const reset = () => { stop(); setPlaying(false); setFrame(0); };

  const ctrlEps   = data?.episodes.filter(ep => ep.type === 'controlled')     ?? [];
  const randEps   = data?.episodes.filter(ep => ep.type === 'random')         ?? [];
  const ceEps     = data?.episodes.filter(ep => ep.type === 'counterexample') ?? [];
  const avgCtrl   = ctrlEps.length ? ctrlEps.reduce((s, e) => s + e.score, 0) / ctrlEps.length : 0;
  const avgRand   = randEps.length ? randEps.reduce((s, e) => s + e.score, 0) / randEps.length : 0;

  if (error) return (
    <div className="text-center py-20 text-red-400">
      <p className="text-xl mb-2">Erro ao carregar dados</p>
      <p className="text-sm font-mono">{error}</p>
      <p className="text-gray-500 text-sm mt-4">Execute <code className="bg-gray-800 px-1 rounded">python generate_webapp_data.py</code> para gerar os dados.</p>
    </div>
  );

  if (!data) return (
    <div className="text-center py-20 text-gray-400">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-400 mx-auto mb-4" />
      Carregando dados da simulação...
    </div>
  );

  const isCE = episode?.type === 'counterexample';
  const progressColor = isCE ? 'bg-orange-500' : episode?.type === 'controlled' ? 'bg-green-500' : 'bg-red-500';

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Simulação Cart-Pole</h1>
        <span className="text-gray-400 text-sm">
          {data.model_info.architecture} &bull; {data.model_info.training_episodes} ep. treinamento
        </span>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-green-900/20 border border-green-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Episódios DQN</p>
          <p className="text-2xl font-bold text-green-400">{ctrlEps.length}</p>
          <p className="text-gray-400 text-xs mt-1">Score médio: {avgCtrl.toFixed(0)} pts</p>
        </div>
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Episódios Aleatórios</p>
          <p className="text-2xl font-bold text-red-400">{randEps.length}</p>
          <p className="text-gray-400 text-xs mt-1">Score médio: {avgRand.toFixed(0)} pts</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Melhoria DQN</p>
          <p className="text-2xl font-bold text-blue-400">
            {avgRand > 0 ? `${(avgCtrl / avgRand).toFixed(0)}×` : '—'}
          </p>
          <p className="text-gray-400 text-xs mt-1">vs aleatório</p>
        </div>
        <div className="bg-orange-900/20 border border-orange-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Falhas ESBMC</p>
          <p className="text-2xl font-bold text-orange-400">{ceEps.length}</p>
          <p className="text-gray-400 text-xs mt-1">contraexemplos verificados</p>
        </div>
      </div>

      {/* Filter buttons */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-gray-400 text-sm mr-1">Filtrar:</span>
        {([
          { f: 'all',            label: 'Todos',          active: 'bg-blue-600'   },
          { f: 'controlled',     label: 'Controlado',     active: 'bg-green-700'  },
          { f: 'random',         label: 'Sem Controle',   active: 'bg-red-700'    },
          { f: 'counterexample', label: 'Falha ESBMC',    active: 'bg-orange-700' },
        ] as { f: Filter; label: string; active: string }[]).map(({ f, label, active }) => (
          <button
            key={f}
            onClick={() => { setFilter(f); setEpIdx(0); setFrame(0); stop(); setPlaying(false); }}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              filter === f ? `${active} text-white` : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
        <span className="text-gray-500 text-xs ml-2">
          {filteredEps.length} episódio{filteredEps.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Episode selector */}
      <div className="flex gap-2 flex-wrap">
        {filteredEps.map((ep, i) => (
          <button
            key={`${ep.seed}-${ep.type}`}
            onClick={() => selectEpisode(i)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              epIdx === i
                ? ep.type === 'controlled'
                  ? 'bg-green-700 text-white'
                  : ep.type === 'counterexample'
                  ? 'bg-orange-700 text-white'
                  : 'bg-red-700 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <TypeBadge type={ep.type} />
            {ep.type === 'counterexample'
              ? ep.esbmc_property
              : `Ep ${ep.seed} — ${ep.score} pts`}
          </button>
        ))}
      </div>

      {episode ? (
        <>
          {/* ESBMC alert panel */}
          <EsbmcAlert episode={episode} frame={frame} />

          <div className={`rounded-xl p-4 border ${
            isCE ? 'bg-orange-900/10 border-orange-800' : 'bg-gray-800 border-gray-700'
          }`}>
            {/* Episode type indicator */}
            <div className="flex items-center gap-2 mb-3">
              <TypeBadge type={episode.type} />
              <span className="text-gray-400 text-sm">
                {episode.type === 'controlled'
                  ? 'Controlador DQN — decisões baseadas em Q-values'
                  : episode.type === 'counterexample'
                  ? 'Estado inicial: contraexemplo formal provado pelo ESBMC'
                  : 'Política aleatória — ações uniformemente aleatórias'}
              </span>
            </div>

            <CartPoleCanvas frame={curFrame} width={500} />

            {/* Progress bar */}
            <div className="mt-3">
              <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-none ${progressColor}`}
                  style={{ width: `${traj.length > 1 ? (frame / (traj.length - 1)) * 100 : 0}%` }}
                />
              </div>
              {/* Critical frame marker */}
              {isCE && (
                <div className="relative h-3 mt-0.5">
                  <div
                    className="absolute top-0 w-0.5 h-3 bg-orange-400"
                    style={{ left: `${traj.length > 1 ? ((episode.critical_frame ?? 0) / (traj.length - 1)) * 100 : 0}%` }}
                    title="Frame crítico (contraexemplo)"
                  />
                </div>
              )}
              <div className="flex justify-between text-gray-500 text-xs mt-1">
                <span>t = {(frame * 0.02).toFixed(2)} s</span>
                <span>{frame} / {traj.length} frames</span>
                <span>t = {(traj.length * 0.02).toFixed(2)} s</span>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3 mt-3 flex-wrap">
              <button
                onClick={() => setPlaying(p => !p)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  playing ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-blue-600 hover:bg-blue-700'
                } text-white min-w-[80px]`}
              >
                {playing ? '⏸ Pausar' : '▶ Play'}
              </button>
              <button
                onClick={reset}
                className="px-4 py-1.5 rounded-lg text-sm bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
              >
                ⟳ Reset
              </button>
              {isCE && (
                <button
                  onClick={() => { stop(); setPlaying(false); setFrame(episode.critical_frame ?? 0); }}
                  className="px-4 py-1.5 rounded-lg text-sm bg-orange-700 hover:bg-orange-600 text-white transition-colors"
                >
                  ⚠ Frame crítico
                </button>
              )}
              <input
                type="range" min={0} max={traj.length - 1} value={frame}
                onChange={e => { stop(); setPlaying(false); setFrame(+e.target.value); }}
                className="flex-1 min-w-[100px] accent-blue-500"
              />
              <div className="flex items-center gap-1.5">
                <span className="text-gray-400 text-xs">Velocidade:</span>
                {SPEEDS.map(s => (
                  <button key={s} onClick={() => setSpeed(s)}
                    className={`px-2 py-0.5 rounded text-xs transition-colors ${
                      speed === s ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                    }`}
                  >{s}×</button>
                ))}
              </div>
            </div>
          </div>

          {/* State plots */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
              <p className="text-gray-400 text-xs uppercase tracking-wider mb-2">Posição do carro</p>
              <StatePlot
                data={traj} currentIdx={frame}
                field="x" label="x (posição)" unit="m"
                limit={2.4} color="#60A5FA"
              />
            </div>
            <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
              <p className="text-gray-400 text-xs uppercase tracking-wider mb-2">Ângulo do pêndulo</p>
              <StatePlot
                data={traj} currentIdx={frame}
                field="theta" label="θ (ângulo)" unit="rad"
                limit={0.2094} color="#34D399"
              />
            </div>
          </div>

          {/* Current state */}
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <p className="text-gray-400 text-xs uppercase tracking-wider mb-3">Estado atual</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-sm">
              {[
                { label: 'x',    val: curFrame.x.toFixed(4),    unit: 'm',     col: 'text-blue-300'  },
                { label: 'ẋ',    val: curFrame.x_dot.toFixed(4), unit: 'm/s',  col: 'text-blue-300'  },
                { label: 'θ',    val: (curFrame.theta*180/Math.PI).toFixed(3), unit: '°', col: 'text-green-300' },
                { label: 'θ̇',   val: curFrame.theta_dot.toFixed(4), unit: 'rad/s', col: 'text-green-300' },
              ].map(({ label, val, unit, col }) => (
                <div key={label} className="bg-gray-900 rounded-lg p-3 text-center">
                  <p className="text-gray-500 text-xs mb-1">{label}</p>
                  <p className={`text-lg font-bold ${col}`}>{val}</p>
                  <p className="text-gray-500 text-xs">{unit}</p>
                </div>
              ))}
            </div>
            {isCE && frame === (episode.critical_frame ?? 0) && (
              <div className="mt-3 p-3 bg-orange-900/20 border border-orange-700 rounded-lg">
                <p className="text-orange-300 text-xs font-semibold">
                  Ação escolhida: {curFrame.action === 1 ? 'Direita (+10N)' : 'Esquerda (−10N)'}
                  {' '}&bull; Q[0]={curFrame.q0.toFixed(3)} &bull; Q[1]={curFrame.q1.toFixed(3)}
                </p>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="text-center py-10 text-gray-500">
          Nenhum episódio para o filtro selecionado.
        </div>
      )}
    </div>
  );
}
