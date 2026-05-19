'use client';

import { useRef, useEffect } from 'react';
import { TrajectoryFrame } from '@/lib/types';

interface Props {
  data:       TrajectoryFrame[];
  currentIdx: number;
  field:      'x' | 'theta';
  label:      string;
  unit:       string;
  limit:      number;          // safety limit (|value| > limit = failed)
  color:      string;          // CSS color for the trace
}

export default function StatePlot({ data, currentIdx, field, label, unit, limit, color }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const PAD = { top: 8, bottom: 20, left: 40, right: 8 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, W, H);

    // Determine value range
    const values   = data.map(f => field === 'x' ? f.x : f.theta);
    const maxAbs   = Math.max(Math.abs(limit) * 1.1, ...values.map(Math.abs));
    const toY      = (v: number) => PAD.top + plotH * (1 - (v + maxAbs) / (2 * maxAbs));
    const toX      = (i: number) => PAD.left + (i / (data.length - 1)) * plotW;
    const zeroY    = toY(0);

    // Grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth   = 1;
    for (const v of [-limit, 0, limit]) {
      const y = toY(v);
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + plotW, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Zero line
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(PAD.left, zeroY); ctx.lineTo(PAD.left + plotW, zeroY); ctx.stroke();

    // Limit labels
    ctx.fillStyle  = '#EF4444';
    ctx.font       = '9px monospace';
    ctx.textAlign  = 'right';
    const lim = field === 'theta' ? (limit * 180 / Math.PI).toFixed(0) + '°' : limit.toFixed(1);
    ctx.fillText(`+${lim}`, PAD.left - 2, toY(limit) + 3);
    ctx.fillText(`-${lim}`, PAD.left - 2, toY(-limit) + 3);

    // Limit lines
    ctx.strokeStyle = '#EF444466';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([3, 3]);
    [limit, -limit].forEach(v => {
      ctx.beginPath(); ctx.moveTo(PAD.left, toY(v)); ctx.lineTo(PAD.left + plotW, toY(v)); ctx.stroke();
    });
    ctx.setLineDash([]);

    // Trace
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const px = toX(i), py = toY(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Current position marker
    if (currentIdx < data.length) {
      const cx = toX(currentIdx);
      const cy = toY(values[currentIdx]);
      // vertical cursor
      ctx.strokeStyle = '#FFFFFF44';
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, PAD.top + plotH); ctx.stroke();
      // dot
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    }

    // Y axis label
    ctx.fillStyle  = '#9CA3AF';
    ctx.font       = '9px monospace';
    ctx.textAlign  = 'left';
    const curVal   = currentIdx < data.length ? values[currentIdx] : values[values.length - 1];
    const dispVal  = field === 'theta'
      ? (curVal * 180 / Math.PI).toFixed(1) + '°'
      : curVal.toFixed(3) + ' ' + unit;
    ctx.fillText(dispVal, PAD.left + 3, PAD.top + 10);

    // Label bottom
    ctx.textAlign  = 'center';
    ctx.fillStyle  = '#6B7280';
    ctx.font       = '9px monospace';
    ctx.fillText(`${label}  (t = ${(currentIdx * 0.02).toFixed(2)} s)`, PAD.left + plotW / 2, H - 4);

  }, [data, currentIdx, field, limit, color, label, unit]);

  return (
    <canvas
      ref={ref}
      width={400}
      height={110}
      className="rounded-lg border border-gray-700"
      style={{ width: '100%', maxWidth: 400 }}
    />
  );
}
