'use client';

import { useState, useEffect, useRef } from 'react';
import { SimulationData, TrainingPoint } from '@/lib/types';

function TrainingChart({ history }: { history: TrainingPoint[] }) {
  const refScore  = useRef<HTMLCanvasElement>(null);
  const refAvg    = useRef<HTMLCanvasElement>(null);
  const refEps    = useRef<HTMLCanvasElement>(null);

  function drawChart(
    canvas: HTMLCanvasElement,
    values: number[],
    color: string,
    label: string,
    yMin: number,
    yMax: number,
    limitLine?: number,
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    const PAD = { top: 12, bottom: 28, left: 48, right: 12 };
    const pw = W - PAD.left - PAD.right;
    const ph = H - PAD.top - PAD.bottom;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, W, H);

    const toX = (i: number) => PAD.left + (i / (values.length - 1)) * pw;
    const toY = (v: number) => PAD.top + ph * (1 - (v - yMin) / (yMax - yMin));

    // Grid lines
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    [0, 0.25, 0.5, 0.75, 1].forEach(f => {
      const y = PAD.top + ph * (1 - f);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
      ctx.fillStyle = '#6B7280';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(Math.round(yMin + f * (yMax - yMin))), PAD.left - 4, y + 3);
    });
    ctx.setLineDash([]);

    // Limit line
    if (limitLine !== undefined) {
      const ly = toY(limitLine);
      ctx.strokeStyle = '#22c55e88';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 3]);
      ctx.beginPath(); ctx.moveTo(PAD.left, ly); ctx.lineTo(PAD.left + pw, ly); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#22c55e';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`solved (${limitLine})`, PAD.left + 4, ly - 3);
    }

    // Trace
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = toX(i), y = toY(Math.max(yMin, Math.min(yMax, v)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // X axis labels
    ctx.fillStyle = '#6B7280';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    [0, 100, 200, 300, 404].forEach(ep => {
      const x = toX(Math.min(ep, values.length - 1));
      ctx.fillText(String(ep), x, H - 6);
    });

    // Label
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, PAD.left + 4, PAD.top + 10);
  }

  useEffect(() => {
    if (!history.length) return;
    const scores  = history.map(p => p.score);
    const avgs    = history.map(p => p.avg100);
    const epsilons = history.map(p => p.epsilon);

    if (refScore.current)
      drawChart(refScore.current, scores,   '#60A5FA', 'Score por episódio',   0, 500, 470);
    if (refAvg.current)
      drawChart(refAvg.current,   avgs,     '#34D399', 'Média 100 episódios',  0, 500, 470);
    if (refEps.current)
      drawChart(refEps.current,   epsilons, '#F59E0B', 'Epsilon (exploração)', 0, 1.05);
  }, [history]);

  if (!history.length) return (
    <p className="text-gray-500 text-sm">Sem dados de treinamento.</p>
  );

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-2">Score por Episódio</p>
        <canvas ref={refScore} width={700} height={140}
          className="rounded border border-gray-700" style={{ width: '100%', maxWidth: 700 }} />
        <p className="text-gray-500 text-xs mt-1">
          Cada ponto = duração de 1 episódio. Alta variância no início é normal — o agente ainda explora aleatoriamente.
        </p>
      </div>
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-2">Média Móvel (100 episódios)</p>
        <canvas ref={refAvg} width={700} height={140}
          className="rounded border border-gray-700" style={{ width: '100%', maxWidth: 700 }} />
        <p className="text-gray-500 text-xs mt-1">
          Critério de resolução: avg100 ≥ 470. Atingido no episódio 404.
        </p>
      </div>
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-2">Epsilon — Taxa de Exploração</p>
        <canvas ref={refEps} width={700} height={120}
          className="rounded border border-gray-700" style={{ width: '100%', maxWidth: 700 }} />
        <p className="text-gray-500 text-xs mt-1">
          ε decai ×0.995 por episódio (1.0 → 0.01). Acima de 0.5: ação aleatória. Abaixo: DQN decide.
        </p>
      </div>
    </div>
  );
}

function MilestoneTable({ history }: { history: TrainingPoint[] }) {
  const milestones = [1, 50, 100, 150, 200, 250, 300, 350, 404];
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-700">
        <h2 className="text-white font-semibold">Marcos do Treinamento</h2>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
            <th className="text-left px-4 py-2">Episódio</th>
            <th className="text-right px-4 py-2">Score</th>
            <th className="text-right px-4 py-2">Avg100</th>
            <th className="text-right px-4 py-2">Epsilon</th>
            <th className="text-left px-4 py-2">Fase</th>
          </tr>
        </thead>
        <tbody>
          {milestones.map(ep => {
            const pt = history[ep - 1];
            if (!pt) return null;
            const phase =
              ep <= 50  ? '🎲 Exploração pura'    :
              ep <= 150 ? '📈 Início do aprendizado' :
              ep <= 300 ? '🚀 Melhoria rápida'    :
                          '✓ Convergência';
            const solved = pt.avg100 >= 470;
            return (
              <tr key={ep} className={`border-b border-gray-700/50 hover:bg-gray-700/30 ${solved ? 'bg-green-900/10' : ''}`}>
                <td className="px-4 py-2 font-mono text-gray-300">{ep}</td>
                <td className="px-4 py-2 text-right font-mono text-blue-300">{pt.score.toFixed(0)}</td>
                <td className={`px-4 py-2 text-right font-mono ${solved ? 'text-green-400 font-bold' : 'text-gray-300'}`}>
                  {pt.avg100.toFixed(1)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-yellow-400">{pt.epsilon.toFixed(3)}</td>
                <td className="px-4 py-2 text-gray-400 text-xs">{phase}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function TrainingPage() {
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

  const history = data.training_history ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Curva de Aprendizado</h1>
        <span className="text-gray-400 text-sm">
          {data.model_info.architecture} &bull; {data.model_info.training_episodes} episódios &bull; score final: {data.model_info.final_avg_score}
        </span>
      </div>

      {/* Resumo */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-blue-900/20 border border-blue-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Episódios</p>
          <p className="text-2xl font-bold text-blue-400">{history.length}</p>
        </div>
        <div className="bg-green-900/20 border border-green-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Score Final</p>
          <p className="text-2xl font-bold text-green-400">{data.model_info.final_avg_score}</p>
        </div>
        <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Epsilon Final</p>
          <p className="text-2xl font-bold text-yellow-400">
            {history.length ? history[history.length - 1].epsilon.toFixed(3) : '—'}
          </p>
        </div>
        <div className="bg-purple-900/20 border border-purple-700 rounded-xl p-3 text-center">
          <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Arquitetura</p>
          <p className="text-lg font-bold text-purple-400 font-mono">{data.model_info.architecture}</p>
        </div>
      </div>

      {/* Explicação */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-sm text-gray-300 space-y-2">
        <h2 className="text-white font-semibold mb-2">Como o DQN aprende</h2>
        <p>
          O agente começa totalmente aleatório (ε=1.0). A cada episódio, ε decai e o DQN começa a
          usar o que aprendeu. O <span className="text-blue-300 font-mono">score</span> é o número
          de passos que o pêndulo ficou em pé. A <span className="text-green-300 font-mono">média 100</span> suaviza
          a variância e é o critério de parada (≥ 470).
        </p>
        <p className="text-gray-400 text-xs">
          Controlador: {data.model_info.architecture} (5 níveis de força: −10, −5, 0, +5, +10 N) &bull;
          γ = 0.99 &bull; lr = 1e-3 &bull; ReplayBuffer = 10 000 &bull; batch = 64 &bull; target update = 200 passos
        </p>
      </div>

      <TrainingChart history={history} />

      {history.length > 0 && <MilestoneTable history={history} />}
    </div>
  );
}
