'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { CartPoleState, physicsStep, isDone, resetState, FORCE_MAX, X_LIMIT, THETA_LIMIT, DT } from '@/lib/physics';
import { DDPGWeights, loadWeights, getForce } from '@/lib/controller';
import { loadQuantizedWeights, QuantizedWeights, stateToQ88, getForceQ88, forceFromQ88 } from '@/lib/quantized-controller';
import { DDPGVerificationData } from '@/lib/types';

const W = 700, H = 340;
const TRACK_Y = 240, TRACK_HW = 280;
const CART_W = 60, CART_H = 30;
const POLE_LEN = 120;
const SCALE_PX = TRACK_HW / X_LIMIT;

const HISTORY_LEN = 250;

type ControlMode = 'q88' | 'float' | 'manual' | 'off';

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: CartPoleState,
  force: number,
  failed: boolean,
  dragging: boolean,
  ceActive: boolean,
) {
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, W, H);

  // Ground
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, TRACK_Y, W, H - TRACK_Y);
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, TRACK_Y);
  ctx.lineTo(W, TRACK_Y);
  ctx.stroke();

  // Track rail (between walls)
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(W / 2 - X_LIMIT * SCALE_PX, TRACK_Y);
  ctx.lineTo(W / 2 + X_LIMIT * SCALE_PX, TRACK_Y);
  ctx.stroke();

  // Walls at x limits
  const wallH = 100;
  [-X_LIMIT, X_LIMIT].forEach(lim => {
    const lx = W / 2 + lim * SCALE_PX;
    ctx.fillStyle = '#7f1d1d88';
    ctx.fillRect(lim < 0 ? lx - 6 : lx, TRACK_Y - wallH, 6, wallH + 20);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lx, TRACK_Y - wallH);
    ctx.lineTo(lx, TRACK_Y + 20);
    ctx.stroke();
    // Wall label
    ctx.fillStyle = '#ef444488';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${lim > 0 ? '+' : ''}${lim.toFixed(1)}m`, lx, TRACK_Y - wallH - 4);
  });

  // Angle limit arcs (show ±12° danger zone)
  const cx = W / 2 + Math.max(-X_LIMIT, Math.min(X_LIMIT, state.x)) * SCALE_PX;

  ctx.strokeStyle = '#ef444433';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  [-THETA_LIMIT, THETA_LIMIT].forEach(angle => {
    const endX = cx + Math.sin(angle) * POLE_LEN;
    const endY = TRACK_Y - CART_H - Math.cos(angle) * POLE_LEN;
    ctx.beginPath();
    ctx.moveTo(cx, TRACK_Y - CART_H);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Danger zone label at top
  ctx.fillStyle = '#ef444444';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  const arcLabelX = cx + Math.sin(THETA_LIMIT) * (POLE_LEN + 12);
  const arcLabelY = TRACK_Y - CART_H - Math.cos(THETA_LIMIT) * (POLE_LEN + 12);
  if (arcLabelX > 20 && arcLabelX < W - 20) {
    ctx.fillText('12°', arcLabelX, arcLabelY);
  }

  // Cart body (clamp visual position to track)
  const atWall = Math.abs(state.x) >= X_LIMIT - 0.01;
  const cartVisualX = Math.max(W / 2 - X_LIMIT * SCALE_PX + CART_W / 2,
                      Math.min(W / 2 + X_LIMIT * SCALE_PX - CART_W / 2, cx));

  // Wall collision flash
  if (atWall) {
    const wallX = state.x > 0
      ? W / 2 + X_LIMIT * SCALE_PX
      : W / 2 - X_LIMIT * SCALE_PX;
    ctx.fillStyle = '#ef444444';
    ctx.beginPath();
    ctx.arc(wallX, TRACK_Y - CART_H / 2, 20, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = failed ? '#7F1D1D' : dragging ? '#1E40AF' : ceActive ? '#7C2D12' : '#1e293b';
  ctx.strokeStyle = failed ? '#EF4444' : dragging ? '#3B82F6' : ceActive ? '#F97316' : '#94a3b8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cartVisualX - CART_W / 2, TRACK_Y - CART_H, CART_W, CART_H, 6);
  ctx.fill();
  ctx.stroke();

  // Wheels
  ctx.fillStyle = '#64748b';
  ctx.beginPath();
  ctx.arc(cartVisualX - 15, TRACK_Y + 2, 6, 0, Math.PI * 2);
  ctx.arc(cartVisualX + 15, TRACK_Y + 2, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cartVisualX - 15, TRACK_Y + 2, 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cartVisualX + 15, TRACK_Y + 2, 6, 0, Math.PI * 2);
  ctx.stroke();

  // Pole
  const poleBaseX = cartVisualX;
  const poleEndX = poleBaseX + Math.sin(state.theta) * POLE_LEN;
  const poleEndY = TRACK_Y - CART_H - Math.cos(state.theta) * POLE_LEN;

  const absDeg = Math.abs(state.theta) * 180 / Math.PI;
  const poleColor = failed ? '#EF4444' : absDeg > 8 ? '#f97316' : absDeg > 5 ? '#eab308' : '#22c55e';

  ctx.strokeStyle = poleColor;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(poleBaseX, TRACK_Y - CART_H);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();

  // Pole tip
  ctx.fillStyle = poleColor;
  ctx.beginPath();
  ctx.arc(poleEndX, poleEndY, 7, 0, Math.PI * 2);
  ctx.fill();

  // Pivot point
  ctx.fillStyle = '#cbd5e1';
  ctx.beginPath();
  ctx.arc(poleBaseX, TRACK_Y - CART_H, 4, 0, Math.PI * 2);
  ctx.fill();

  // Force arrow
  if (Math.abs(force) > 0.3) {
    const arrowX = cartVisualX + (force > 0 ? CART_W / 2 + 8 : -CART_W / 2 - 8);
    const arrowLen = Math.abs(force) / FORCE_MAX * 50;
    const dir = force > 0 ? 1 : -1;
    ctx.strokeStyle = force > 0 ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(arrowX, TRACK_Y - CART_H / 2);
    ctx.lineTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2 - 6);
    ctx.lineTo(arrowX + dir * (arrowLen + 10), TRACK_Y - CART_H / 2);
    ctx.lineTo(arrowX + dir * arrowLen, TRACK_Y - CART_H / 2 + 6);
    ctx.fill();

    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${force.toFixed(1)}N`, arrowX + dir * arrowLen / 2, TRACK_Y - CART_H / 2 - 10);
  }

  // HUD top
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`x=${state.x.toFixed(3)}m  θ=${(state.theta*180/Math.PI).toFixed(2)}°  F=${force.toFixed(2)}N`, 10, 18);

  ctx.textAlign = 'right';
  const thetaDeg = (state.theta * 180 / Math.PI).toFixed(1);
  ctx.fillStyle = absDeg > 10 ? '#ef4444' : absDeg > 6 ? '#f97316' : '#22c55e';
  ctx.fillText(`${thetaDeg}°`, W - 10, 18);

  // Status banner
  if (ceActive) {
    ctx.fillStyle = '#f9731688';
    ctx.fillRect(0, 28, W, 22);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('CONTRAEXEMPLO ESBMC — observando comportamento do controlador', W / 2, 43);
  } else if (failed) {
    ctx.fillStyle = '#ef444488';
    ctx.fillRect(0, 28, W, 22);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('FALHOU — sistema saiu da regiao segura', W / 2, 43);
  } else if (dragging) {
    ctx.fillStyle = '#3B82F644';
    ctx.fillRect(0, 28, W, 22);
    ctx.fillStyle = '#93c5fd';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Arrastando — solte para liberar', W / 2, 43);
  }

  // Ground texture (rails)
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let gx = W / 2 - TRACK_HW; gx <= W / 2 + TRACK_HW; gx += 20) {
    ctx.beginPath();
    ctx.moveTo(gx, TRACK_Y + 8);
    ctx.lineTo(gx, TRACK_Y + 16);
    ctx.stroke();
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

    ctx.strokeStyle = '#EF444444';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    [limit, -limit].forEach(v => {
      const y = toY(v);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    });
    ctx.setLineDash([]);

    ctx.strokeStyle = '#374151';
    ctx.beginPath(); ctx.moveTo(0, ch / 2); ctx.lineTo(cw, ch / 2); ctx.stroke();

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

    if (history.length > 0) {
      const cur = history[history.length - 1];
      ctx.fillStyle = color;
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${cur.toFixed(3)} ${unit}`, cw - 4, 12);
    }

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

function parseCounterexampleState(str: string): CartPoleState | null {
  const m = str.match(/x=(-?[\d.]+)\s+xd=(-?[\d.]+)\s+th=(-?[\d.]+)\s+thd=(-?[\d.]+)/);
  if (m) {
    return {
      x: parseFloat(m[1]),
      x_dot: parseFloat(m[2]),
      theta: parseFloat(m[3]),
      theta_dot: parseFloat(m[4]),
    };
  }
  return null;
}

export default function SimulationPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [floatWeights, setFloatWeights] = useState<DDPGWeights | null>(null);
  const [qWeights, setQWeights] = useState<QuantizedWeights | null>(null);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<ControlMode>('q88');
  const [steps, setSteps] = useState(0);
  const [force, setForce] = useState(0);
  const [error, setError] = useState('');
  const [ceActive, setCeActive] = useState(false);
  const [verData, setVerData] = useState<DDPGVerificationData | null>(null);

  const stateRef = useRef<CartPoleState>(resetState());
  const forceRef = useRef(0);
  const runRef = useRef(false);
  const modeRef = useRef<ControlMode>('q88');
  const failRef = useRef(false);
  const dragRef = useRef(false);
  const dragXRef = useRef(0);
  const manualForceRef = useRef(0);
  const floatWeightsRef = useRef<DDPGWeights | null>(null);
  const qWeightsRef = useRef<QuantizedWeights | null>(null);
  const ceActiveRef = useRef(false);

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
    const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
    Promise.all([
      loadWeights().then(w => { setFloatWeights(w); floatWeightsRef.current = w; }),
      loadQuantizedWeights().then(w => { setQWeights(w); qWeightsRef.current = w; }),
      fetch(`${base}/ddpg_verification_data.json`).then(r => r.ok ? r.json() : null).then(setVerData),
    ]).catch(() => setError('Falha ao carregar pesos'));
  }, []);

  const doReset = useCallback(() => {
    stateRef.current = resetState();
    forceRef.current = 0;
    failRef.current = false;
    ceActiveRef.current = false;
    setFailed(false);
    setCeActive(false);
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
    if (ctx) drawScene(ctx, stateRef.current, 0, false, false, false);
  }, []);

  const injectCounterexample = useCallback((stateStr: string) => {
    const parsed = parseCounterexampleState(stateStr);
    if (!parsed) return;
    stateRef.current = parsed;
    forceRef.current = 0;
    failRef.current = false;
    ceActiveRef.current = true;
    setFailed(false);
    setCeActive(true);
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
    setRunning(true);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) drawScene(ctx, stateRef.current, 0, false, false, true);
  }, []);

  useEffect(() => { modeRef.current = mode; }, [mode]);

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
      } else if (modeRef.current === 'q88' && qWeightsRef.current) {
        const sq = stateToQ88([s.x, s.x_dot, s.theta, s.theta_dot]);
        const fq = getForceQ88(sq, qWeightsRef.current);
        f = forceFromQ88(fq);
      } else if (modeRef.current === 'float' && floatWeightsRef.current) {
        f = getForce([s.x, s.x_dot, s.theta, s.theta_dot], floatWeightsRef.current);
      } else if (modeRef.current === 'manual') {
        f = manualForceRef.current;
      }

      const ns = physicsStep(s, f);
      const done = isDone(ns);

      stateRef.current = ns;
      forceRef.current = f;

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
        setRunning(true);
      }

      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) drawScene(ctx, ns, f, done || failRef.current, dragRef.current, ceActiveRef.current);
    }, DT * 1000);

    return () => clearInterval(interval);
  }, [running]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) drawScene(ctx, stateRef.current, 0, false, false, false);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = (e.clientX - rect.left) / rect.width * W;
    const cx = W / 2 + stateRef.current.x * SCALE_PX;
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

  const onTouchStart = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || !e.touches[0]) return;
    const mx = (e.touches[0].clientX - rect.left) / rect.width * W;
    const cx = W / 2 + stateRef.current.x * SCALE_PX;
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
  const counterexamples = verData?.counterexamples ?? [];

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
          {mode === 'q88' ? 'Q8.8 verificado' : mode === 'float' ? 'Float32 referencia' : mode === 'manual' ? 'Manual' : 'Sem controle'}
          {' | 4→24→24→tanh×10'}
        </span>
      </div>

      {/* Mode + controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-400 text-sm mr-1">Modo:</span>
          {([
            { m: 'q88' as ControlMode,    label: 'Q8.8 (verificado)', color: 'bg-green-700' },
            { m: 'float' as ControlMode,  label: 'Float32',           color: 'bg-blue-700' },
            { m: 'manual' as ControlMode, label: 'Manual',            color: 'bg-purple-700' },
            { m: 'off' as ControlMode,    label: 'Sem controle',      color: 'bg-red-700' },
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
          <span className={failed ? 'text-red-400 font-bold animate-pulse' : 'text-green-400'}>
            {failed ? 'FORA DA REGIAO SEGURA' : 'ESTAVEL'}
          </span>
        </div>
      </div>

      {/* Instructions */}
      <div className="text-gray-500 text-xs flex gap-4 flex-wrap">
        <span>Clique e arraste o carrinho para perturba-lo</span>
        <span>Setas para forca manual</span>
        {mode === 'q88' && <span className="text-green-500">Q8.8 — mesma aritmetica verificada pelo ESBMC</span>}
        {mode === 'float' && <span className="text-blue-500">Float32 — referencia nao-verificada</span>}
      </div>

      {/* Counterexample injection */}
      {counterexamples.length > 0 && (
        <div className="bg-orange-900/20 border border-orange-700 rounded-xl p-4 space-y-3">
          <h3 className="text-orange-300 font-semibold text-sm">Contraexemplos ESBMC</h3>
          <p className="text-gray-400 text-xs">
            Injete estados encontrados pelo ESBMC para ver a falha em tempo real.
            O controlador Q8.8 reproduz exatamente o comportamento verificado.
          </p>
          <div className="space-y-2">
            {counterexamples.map((ce, i) => {
              const canInject = parseCounterexampleState(ce.state_str) !== null;
              return (
                <div key={i} className="flex items-start gap-3 bg-orange-950/40 rounded-lg p-3 border border-orange-800/50">
                  <div className="flex-1 min-w-0">
                    <p className="text-orange-200 text-xs font-semibold">{ce.property}</p>
                    <p className="text-gray-400 text-xs mt-0.5">{ce.description}</p>
                    <p className="text-orange-300 text-xs font-mono mt-1 break-all">{ce.state_str}</p>
                    <p className="text-gray-500 text-xs mt-0.5">Esperado: {ce.expected_behavior}</p>
                  </div>
                  {canInject ? (
                    <button
                      onClick={() => injectCounterexample(ce.state_str)}
                      className="flex-shrink-0 px-3 py-1.5 bg-orange-700 hover:bg-orange-600 text-white rounded-lg text-xs font-medium transition-colors"
                    >
                      Injetar
                    </button>
                  ) : (
                    <span className="flex-shrink-0 px-3 py-1.5 bg-gray-700 text-gray-400 rounded-lg text-xs">
                      Sem estado
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          {ceActive && (
            <div className="bg-orange-950/60 border border-orange-800 rounded-lg p-2 text-orange-200 text-xs font-mono">
              Estado injetado — observando comportamento do controlador Q8.8
            </div>
          )}
        </div>
      )}

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
          Estado — tempo real ({HISTORY_LEN} amostras = {(HISTORY_LEN * DT).toFixed(1)}s)
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <MiniPlot history={histX}   label="x (m)"      limit={X_LIMIT}     color="#60A5FA" unit="m" />
          <MiniPlot history={histXd}  label="x_dot (m/s)" limit={5.0}         color="#93C5FD" unit="m/s" />
          <MiniPlot history={histTh}  label="theta (rad)" limit={THETA_LIMIT} color="#34D399" unit="rad" />
          <MiniPlot history={histThd} label="theta_dot"   limit={5.0}         color="#86EFAC" unit="rad/s" />
          <MiniPlot history={histF}   label="F (N)"       limit={FORCE_MAX}   color="#FBBF24" unit="N" />
          <div className="bg-gray-900 rounded border border-gray-700 p-3 flex flex-col items-center justify-center">
            <p className="text-gray-500 text-xs mb-2">Forca atual</p>
            <p className={`text-3xl font-bold font-mono ${
              force > 1 ? 'text-green-400' : force < -1 ? 'text-red-400' : 'text-gray-400'
            }`}>
              {force.toFixed(2)}
            </p>
            <p className="text-gray-500 text-xs mt-1">N</p>
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

      {/* State numerical */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p className="text-gray-400 text-xs uppercase tracking-wider mb-3">Estado numerico</p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 font-mono text-sm">
          {[
            { label: 'x',         val: stateRef.current.x,         unit: 'm',     col: 'text-blue-300',  limit: X_LIMIT },
            { label: 'x_dot',     val: stateRef.current.x_dot,     unit: 'm/s',   col: 'text-blue-300',  limit: 5.0 },
            { label: 'theta',     val: stateRef.current.theta,     unit: 'rad',   col: 'text-green-300', limit: THETA_LIMIT },
            { label: 'theta_dot', val: stateRef.current.theta_dot, unit: 'rad/s', col: 'text-green-300', limit: 5.0 },
            { label: 'F',         val: force,                       unit: 'N',     col: 'text-yellow-300', limit: FORCE_MAX },
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

      {/* Manual force slider */}
      {mode === 'manual' && (
        <div className="bg-gray-800 rounded-xl p-4 border border-purple-700">
          <p className="text-purple-400 text-xs uppercase tracking-wider mb-2">
            Controle manual — arraste o slider ou use setas
          </p>
          <input
            type="range"
            min={-FORCE_MAX}
            max={FORCE_MAX}
            step={0.1}
            value={manualForceRef.current}
            onChange={e => { manualForceRef.current = parseFloat(e.target.value); }}
            className="w-full accent-purple-500"
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
