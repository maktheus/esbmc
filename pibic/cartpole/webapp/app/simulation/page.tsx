'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { CartPoleState, physicsStep, isDone, resetState, FORCE_MAX, X_LIMIT, THETA_LIMIT, DT } from '@/lib/physics';
import { DDPGWeights, loadWeights, getForce } from '@/lib/controller';

const W = 600, H = 280;
const TRACK_Y = 200, TRACK_HW = 200;
const CART_W = 60, CART_H = 30;
const POLE_LEN = 120;
const SCALE = TRACK_HW / X_LIMIT;

const HISTORY_LEN = 250;

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: CartPoleState,
  force: number,
  failed: boolean,
  dragging: boolean,
) {
  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.strokeStyle = '#374151';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(W / 2 - TRACK_HW, TRACK_Y);
  ctx.lineTo(W / 2 + TRACK_HW, TRACK_Y);
  ctx.stroke();

  // Limit markers
  ctx.strokeStyle = '#EF444466';
  ctx.setLineDash([4, 4]);
  [-X_LIMIT, X_LIMIT].forEach(lim => {
    const lx = W / 2 + lim * SCALE;
    ctx.beginPath();
    ctx.moveTo(lx, TRACK_Y - 80);
    ctx.lineTo(lx, TRACK_Y + 10);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  const cx = W / 2 + state.x * SCALE;

  // Cart
  ctx.fillStyle = failed ? '#7F1D1D' : dragging ? '#1E40AF' : '#1F2937';
  ctx.strokeStyle = failed ? '#EF4444' : dragging ? '#3B82F6' : '#6B7280';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cx - CART_W / 2, TRACK_Y - CART_H, CART_W, CART_H, 4);
  ctx.fill();
  ctx.stroke();

  // Wheels
  ctx.fillStyle = '#4B5563';
  ctx.beginPath();
  ctx.arc(cx - 15, TRACK_Y, 5, 0, Math.PI * 2);
  ctx.arc(cx + 15, TRACK_Y, 5, 0, Math.PI * 2);
  ctx.fill();

  // Pole
  const poleEndX = cx + Math.sin(state.theta) * POLE_LEN;
  const poleEndY = TRACK_Y - CART_H - Math.cos(state.theta) * POLE_LEN;
  ctx.strokeStyle = failed ? '#EF4444' : '#F59E0B';
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx, TRACK_Y - CART_H);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();

  // Pole tip
  ctx.fillStyle = failed ? '#EF4444' : '#FBBF24';
  ctx.beginPath();
  ctx.arc(poleEndX, poleEndY, 6, 0, Math.PI * 2);
  ctx.fill();

  // Pivot
  ctx.fillStyle = '#9CA3AF';
  ctx.beginPath();
  ctx.arc(cx, TRACK_Y - CART_H, 4, 0, Math.PI * 2);
  ctx.fill();

  // Force arrow
  if (Math.abs(force) > 0.5) {
    const arrowX = cx + (force > 0 ? CART_W / 2 + 5 : -CART_W / 2 - 5);
    const arrowLen = Math.abs(force) / FORCE_MAX * 40;
    const dir = force > 0 ? 1 : -1;
    ctx.strokeStyle = force > 0 ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(arrowX, TRACK_Y - CART_H / 2);
    ctx.lineTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2);
    ctx.stroke();
    // arrowhead
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2 - 5);
    ctx.lineTo(arrowX + dir * (arrowLen + 8), TRACK_Y - CART_H / 2);
    ctx.lineTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2 + 5);
    ctx.fill();
  }

  // Force label
  ctx.fillStyle = '#9CA3AF';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`F = ${force.toFixed(2)} N`, W / 2, 20);

  if (dragging) {
    ctx.fillStyle = '#3B82F680';
    ctx.font = '12px sans-serif';
    ctx.fillText('ARRASTANDO', W / 2, 40);
  }
}

function MiniPlot({ history, label, limit, color, unit }: {
  history: number[]; label: string; limit: number; color: string; unit: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const cw = c.width, ch = c.height;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, cw, ch);

    const maxAbs = Math.max(limit * 1.2, ...history.map(Math.abs));
    const toY = (v: number) => ch / 2 - (v / maxAbs) * (ch / 2 - 4);

    // limit lines
    ctx.strokeStyle = '#EF444444';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    [limit, -limit].forEach(v => {
      const y = toY(v);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    });
    ctx.setLineDash([]);

    // zero line
    ctx.strokeStyle = '#374151';
    ctx.beginPath(); ctx.moveTo(0, ch / 2); ctx.lineTo(cw, ch / 2); ctx.stroke();

    // trace
    if (history.length > 1) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      history.forEach((v, i) => {
        const x = (i / (HISTORY_LEN - 1)) * cw;
        const y = toY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // current value
    if (history.length > 0) {
      const cur = history[history.length - 1];
      ctx.fillStyle = color;
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${cur.toFixed(3)} ${unit}`, cw - 4, 12);
    }

    // label
    ctx.fillStyle = '#6B7280';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, 4, 12);
  }, [history, label, limit, color, unit]);

  return (
    <canvas ref={ref} width={200} height={60}
      className="rounded border border-gray-700"
      style={{ width: '100%', height: 60 }} />
  );
}

export default function SimulationPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [weights, setWeights] = useState<DDPGWeights | null>(null);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<'ddpg' | 'manual' | 'off'>('ddpg');
  const [steps, setSteps] = useState(0);
  const [force, setForce] = useState(0);
  const [error, setError] = useState('');

  const stateRef = useRef<CartPoleState>(resetState());
  const forceRef = useRef(0);
  const runRef = useRef(false);
  const modeRef = useRef<'ddpg' | 'manual' | 'off'>('ddpg');
  const failRef = useRef(false);
  const dragRef = useRef(false);
  const dragXRef = useRef(0);
  const manualForceRef = useRef(0);
  const weightsRef = useRef<DDPGWeights | null>(null);

  // Histories for mini plots
  const [histX, setHistX] = useState<number[]>([]);
  const [histXd, setHistXd] = useState<number[]>([]);
  const [histTh, setHistTh] = useState<number[]>([]);
  const [histThd, setHistThd] = useState<number[]>([]);
  const [histF, setHistF] = useState<number[]>([]);

  const histXRef = useRef<number[]>([]);
  const histXdRef = useRef<number[]>([]);
  const histThRef = useRef<number[]>([]);
  const histThdRef = useRef<number[]>([]);
  const histFRef = useRef<number[]>([]);

  useEffect(() => {
    loadWeights()
      .then(w => { setWeights(w); weightsRef.current = w; })
      .catch(() => setError('Pesos DDPG não encontrados. Execute python train_ddpg.py'));
  }, []);

  const doReset = useCallback(() => {
    stateRef.current = resetState();
    forceRef.current = 0;
    failRef.current = false;
    setFailed(false);
    setSteps(0);
    setForce(0);
    histXRef.current = [];
    histXdRef.current = [];
    histThRef.current = [];
    histThdRef.current = [];
    histFRef.current = [];
    setHistX([]);
    setHistXd([]);
    setHistTh([]);
    setHistThd([]);
    setHistF([]);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) drawScene(ctx, stateRef.current, 0, false, false);
  }, []);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  // Simulation loop
  useEffect(() => {
    runRef.current = running;
    if (!running) return;

    let frameCount = 0;
    const interval = setInterval(() => {
      if (!runRef.current) return;

      const s = stateRef.current;
      let f = 0;

      if (dragRef.current) {
        f = manualForceRef.current;
      } else if (modeRef.current === 'ddpg' && weightsRef.current) {
        f = getForce([s.x, s.x_dot, s.theta, s.theta_dot], weightsRef.current);
      } else if (modeRef.current === 'manual') {
        f = manualForceRef.current;
      }

      const ns = physicsStep(s, f);
      const done = isDone(ns);

      stateRef.current = ns;
      forceRef.current = f;

      // Push history
      const push = (arr: number[], v: number) => {
        arr.push(v);
        if (arr.length > HISTORY_LEN) arr.shift();
        return arr;
      };
      push(histXRef.current, ns.x);
      push(histXdRef.current, ns.x_dot);
      push(histThRef.current, ns.theta);
      push(histThdRef.current, ns.theta_dot);
      push(histFRef.current, f);

      frameCount++;
      if (frameCount % 3 === 0) {
        setSteps(prev => prev + 3);
        setForce(f);
        setHistX([...histXRef.current]);
        setHistXd([...histXdRef.current]);
        setHistTh([...histThRef.current]);
        setHistThd([...histThdRef.current]);
        setHistF([...histFRef.current]);
      }

      if (done && !failRef.current) {
        failRef.current = true;
        setFailed(true);
      }

      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) drawScene(ctx, ns, f, done || failRef.current, dragRef.current);
    }, DT * 1000);

    return () => clearInterval(interval);
  }, [running]);

  // Draw initial state
  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) drawScene(ctx, stateRef.current, 0, false, false);
  }, []);

  // Mouse interaction for dragging
  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = (e.clientX - rect.left) / rect.width * W;
    const cx = W / 2 + stateRef.current.x * SCALE;
    if (Math.abs(mx - cx) < CART_W) {
      dragRef.current = true;
      dragXRef.current = mx;
    }
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = (e.clientX - rect.left) / rect.width * W;
    const dx = mx - dragXRef.current;
    manualForceRef.current = Math.max(-FORCE_MAX, Math.min(FORCE_MAX, dx * 0.5));
    dragXRef.current = mx;
  }, []);

  const onMouseUp = useCallback(() => {
    dragRef.current = false;
    manualForceRef.current = 0;
  }, []);

  // Touch support
  const onTouchStart = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !e.touches[0]) return;
    const mx = (e.touches[0].clientX - rect.left) / rect.width * W;
    const cx = W / 2 + stateRef.current.x * SCALE;
    if (Math.abs(mx - cx) < CART_W * 1.5) {
      dragRef.current = true;
      dragXRef.current = mx;
    }
  }, []);

  const onTouchMove = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || !e.touches[0]) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = (e.touches[0].clientX - rect.left) / rect.width * W;
    const dx = mx - dragXRef.current;
    manualForceRef.current = Math.max(-FORCE_MAX, Math.min(FORCE_MAX, dx * 0.5));
    dragXRef.current = mx;
  }, []);

  const onTouchEnd = useCallback(() => {
    dragRef.current = false;
    manualForceRef.current = 0;
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft')  manualForceRef.current = -FORCE_MAX;
      if (e.key === 'ArrowRight') manualForceRef.current = FORCE_MAX;
    };
    const upHandler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') manualForceRef.current = 0;
    };
    window.addEventListener('keydown', handler);
    window.addEventListener('keyup', upHandler);
    return () => {
      window.removeEventListener('keydown', handler);
      window.removeEventListener('keyup', upHandler);
    };
  }, []);

  const elapsed = (steps * DT).toFixed(2);

  if (error) return (
    <div className="text-center py-20 text-red-400">
      <p className="text-xl mb-2">Erro</p>
      <p className="text-sm font-mono">{error}</p>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Simulacao em Tempo Real</h1>
        <span className="text-gray-400 text-sm">
          DDPG continuo &bull; 4 → 24 → 24 → tanh &bull; F ∈ [-10, +10] N
        </span>
      </div>

      {/* Mode + controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-400 text-sm mr-1">Modo:</span>
          {([
            { m: 'ddpg' as const, label: 'DDPG', color: 'bg-green-700' },
            { m: 'manual' as const, label: 'Manual', color: 'bg-blue-700' },
            { m: 'off' as const, label: 'Sem controle', color: 'bg-red-700' },
          ]).map(({ m, label, color }) => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                mode === m ? `${color} text-white` : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}>
              {label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setRunning(r => !r)}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            running ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-blue-600 hover:bg-blue-700'
          } text-white min-w-[80px]`}>
          {running ? 'Pausar' : 'Iniciar'}
        </button>

        <button onClick={doReset}
          className="px-4 py-1.5 rounded-lg text-sm bg-gray-700 hover:bg-gray-600 text-gray-200">
          Reset
        </button>

        <div className="ml-auto flex items-center gap-4 text-xs font-mono text-gray-400">
          <span>t = {elapsed} s</span>
          <span>{steps} passos</span>
          <span className={failed ? 'text-red-400 font-bold' : 'text-green-400'}>
            {failed ? 'FALHOU' : 'ESTAVEL'}
          </span>
        </div>
      </div>

      {/* Instructions */}
      <div className="text-gray-500 text-xs flex gap-4 flex-wrap">
        <span>Clique e arraste o carrinho para perturbá-lo</span>
        <span>← → setas para aplicar força manual</span>
        {mode === 'ddpg' && <span className="text-green-500">DDPG responde em tempo real</span>}
      </div>

      {/* Canvas */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-3">
        <canvas
          ref={canvasRef}
          width={W} height={H}
          className="rounded-lg cursor-grab active:cursor-grabbing"
          style={{ width: '100%', maxWidth: W }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
        />
      </div>

      {/* State + Force plots */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-3">
          Estado do sistema — tempo real ({HISTORY_LEN} amostras = {(HISTORY_LEN * DT).toFixed(1)}s)
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <MiniPlot history={histX}   label="x (m)"      limit={X_LIMIT}     color="#60A5FA" unit="m" />
          <MiniPlot history={histXd}  label="ẋ (m/s)"    limit={5.0}         color="#93C5FD" unit="m/s" />
          <MiniPlot history={histTh}  label="θ (rad)"    limit={THETA_LIMIT} color="#34D399" unit="rad" />
          <MiniPlot history={histThd} label="θ̇ (rad/s)" limit={5.0}         color="#86EFAC" unit="rad/s" />
          <MiniPlot history={histF}   label="F (N)"      limit={FORCE_MAX}   color="#FBBF24" unit="N" />
          <div className="bg-gray-900 rounded border border-gray-700 p-3 flex flex-col items-center justify-center">
            <p className="text-gray-500 text-xs mb-2">Forca atual</p>
            <p className={`text-3xl font-bold font-mono ${
              force > 1 ? 'text-green-400' : force < -1 ? 'text-red-400' : 'text-gray-400'
            }`}>
              {force.toFixed(2)}
            </p>
            <p className="text-gray-500 text-xs mt-1">N</p>
            {/* Force bar */}
            <div className="w-full h-3 bg-gray-700 rounded-full mt-2 relative">
              <div className="absolute top-0 left-1/2 w-0.5 h-3 bg-gray-500" />
              <div
                className={`absolute top-0 h-3 rounded-full ${force > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                style={{
                  left: force > 0 ? '50%' : `${50 + (force / FORCE_MAX) * 50}%`,
                  width: `${Math.abs(force / FORCE_MAX) * 50}%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Current state numerical */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-3">Estado numerico</p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 font-mono text-sm">
          {[
            { label: 'x',  val: stateRef.current.x,         unit: 'm',     col: 'text-blue-300',  limit: X_LIMIT },
            { label: 'ẋ',  val: stateRef.current.x_dot,     unit: 'm/s',   col: 'text-blue-300',  limit: 5.0 },
            { label: 'θ',  val: stateRef.current.theta,     unit: 'rad',   col: 'text-green-300', limit: THETA_LIMIT },
            { label: 'θ̇', val: stateRef.current.theta_dot, unit: 'rad/s', col: 'text-green-300', limit: 5.0 },
            { label: 'F',  val: force,                       unit: 'N',     col: 'text-yellow-300', limit: FORCE_MAX },
          ].map(({ label, val, unit, col, limit }) => {
            const danger = Math.abs(val) > limit * 0.8;
            return (
              <div key={label} className={`bg-gray-900 rounded-lg p-3 text-center ${danger ? 'border border-red-700' : ''}`}>
                <p className="text-gray-500 text-xs mb-1">{label}</p>
                <p className={`text-lg font-bold ${danger ? 'text-red-400' : col}`}>{val.toFixed(4)}</p>
                <p className="text-gray-500 text-xs">{unit}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Manual force slider (for manual mode) */}
      {mode === 'manual' && (
        <div className="bg-gray-800 rounded-xl p-4 border border-blue-700">
          <p className="text-blue-400 text-xs uppercase tracking-wider mb-2">
            Controle manual — arraste o slider ou use ← →
          </p>
          <input
            type="range"
            min={-FORCE_MAX}
            max={FORCE_MAX}
            step={0.1}
            value={manualForceRef.current}
            onChange={e => { manualForceRef.current = parseFloat(e.target.value); }}
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between text-gray-500 text-xs mt-1">
            <span>-{FORCE_MAX} N</span>
            <span>0</span>
            <span>+{FORCE_MAX} N</span>
          </div>
        </div>
      )}
    </div>
  );
}
