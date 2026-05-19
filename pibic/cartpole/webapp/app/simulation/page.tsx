'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import CartPoleCanvas from '@/components/CartPoleCanvas';
import StatePlot      from '@/components/StatePlot';
import { SimulationData, Episode, TrajectoryFrame } from '@/lib/types';

const SPEEDS   = [0.5, 1, 2, 5, 10];
const BASE_FPS = 50;  // matches 0.02s dt

export default function SimulationPage() {
  const [data,     setData]     = useState<SimulationData | null>(null);
  const [epIdx,    setEpIdx]    = useState(0);
  const [frame,    setFrame]    = useState(0);
  const [playing,  setPlaying]  = useState(false);
  const [speed,    setSpeed]    = useState(1);
  const [error,    setError]    = useState('');
  const rafRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load simulation data
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

  const episode: Episode | null = data?.episodes[epIdx] ?? null;
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

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Simulação Cart-Pole</h1>
        <span className="text-gray-400 text-sm">
          {data.model_info.architecture} &bull; {data.model_info.training_episodes} ep. treinamento
        </span>
      </div>

      {/* Episode selector */}
      <div className="flex gap-2 flex-wrap">
        {data.episodes.map((ep, i) => (
          <button
            key={i}
            onClick={() => selectEpisode(i)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              epIdx === i
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Ep {ep.seed} — {ep.score} pts
          </button>
        ))}
      </div>

      {/* Main animation */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <CartPoleCanvas frame={curFrame} width={500} />

        {/* Progress bar */}
        <div className="mt-3">
          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-none"
              style={{ width: `${traj.length > 1 ? (frame / (traj.length - 1)) * 100 : 0}%` }}
            />
          </div>
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

          {/* Scrubber */}
          <input
            type="range" min={0} max={traj.length - 1} value={frame}
            onChange={e => { stop(); setPlaying(false); setFrame(+e.target.value); }}
            className="flex-1 min-w-[100px] accent-blue-500"
          />

          {/* Speed */}
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

      {/* State plots + Q-values */}
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

      {/* Current state table */}
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
      </div>
    </div>
  );
}
