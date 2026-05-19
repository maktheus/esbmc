'use client';

import React from 'react';
import { TrajectoryFrame } from '@/lib/types';

interface CartPoleCanvasProps {
  frame: TrajectoryFrame;
  width?: number;
}

// Physics constants matching cartpole_env.py
const CART_W = 60;   // px
const CART_H = 30;   // px
const POLE_LEN = 100; // px (1m = 100px, pole half-length = 0.5m)
const WHEEL_R = 8;

// SVG canvas dimensions
const SVG_W = 500;
const SVG_H = 250;

// Center of track in SVG coords
const CX = SVG_W / 2;
const TRACK_Y = SVG_H - 60; // y of the track line

// Scale: 1 meter = 100px
const SCALE = 100;
const X_LIMIT = 2.4;

export default function CartPoleCanvas({ frame, width }: CartPoleCanvasProps) {
  const { x, theta, action } = frame;

  // Cart center in SVG coords
  const cartX = CX + x * SCALE;
  const cartY = TRACK_Y - CART_H / 2;

  // Pole tip position (theta=0 is upright; positive theta = clockwise in physics)
  const poleTopX = cartX + Math.sin(theta) * POLE_LEN;
  const poleTopY = cartY - Math.cos(theta) * POLE_LEN;

  // Limit lines (±2.4m from center)
  const leftLimit  = CX - X_LIMIT * SCALE;
  const rightLimit = CX + X_LIMIT * SCALE;

  // Clamp cart visually within SVG
  const cartXClamped = Math.max(CART_W / 2 + 2, Math.min(SVG_W - CART_W / 2 - 2, cartX));

  // Arrow direction
  const arrowColor = action === 1 ? '#22c55e' : '#ef4444';
  const arrowDx = action === 1 ? 20 : -20;

  const svgStyle = width ? { width: '100%', maxWidth: width } : { width: '100%' };

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={svgStyle}
      className="block"
      aria-label="Cart-Pole animation"
    >
      {/* Background */}
      <rect x={0} y={0} width={SVG_W} height={SVG_H} fill="#111827" rx={8} />

      {/* Track */}
      <line x1={20} y1={TRACK_Y} x2={SVG_W - 20} y2={TRACK_Y} stroke="#6b7280" strokeWidth={3} />

      {/* Limit lines */}
      <line
        x1={leftLimit} y1={TRACK_Y - 80}
        x2={leftLimit} y2={TRACK_Y + 10}
        stroke="#ef4444" strokeWidth={1.5}
        strokeDasharray="6,4" opacity={0.7}
      />
      <line
        x1={rightLimit} y1={TRACK_Y - 80}
        x2={rightLimit} y2={TRACK_Y + 10}
        stroke="#ef4444" strokeWidth={1.5}
        strokeDasharray="6,4" opacity={0.7}
      />
      <text x={leftLimit + 3} y={TRACK_Y - 84} fill="#ef4444" fontSize={9} opacity={0.7}>-2.4m</text>
      <text x={rightLimit - 26} y={TRACK_Y - 84} fill="#ef4444" fontSize={9} opacity={0.7}>+2.4m</text>

      {/* Center mark */}
      <line
        x1={CX} y1={TRACK_Y - 8}
        x2={CX} y2={TRACK_Y + 8}
        stroke="#4b5563" strokeWidth={1}
      />

      {/* Pole (draw before cart so cart is on top) */}
      <line
        x1={cartXClamped}
        y1={cartY}
        x2={cartXClamped + (poleTopX - cartX)}
        y2={poleTopY}
        stroke="#ef4444"
        strokeWidth={6}
        strokeLinecap="round"
      />

      {/* Pole tip circle */}
      <circle
        cx={cartXClamped + (poleTopX - cartX)}
        cy={poleTopY}
        r={5}
        fill="#fca5a5"
      />

      {/* Cart body */}
      <rect
        x={cartXClamped - CART_W / 2}
        y={cartY - CART_H / 2}
        width={CART_W}
        height={CART_H}
        fill="#3b82f6"
        rx={4}
      />

      {/* Wheels */}
      <circle cx={cartXClamped - CART_W / 4} cy={cartY + CART_H / 2} r={WHEEL_R} fill="#1d4ed8" />
      <circle cx={cartXClamped + CART_W / 4} cy={cartY + CART_H / 2} r={WHEEL_R} fill="#1d4ed8" />
      <circle cx={cartXClamped - CART_W / 4} cy={cartY + CART_H / 2} r={3} fill="#93c5fd" />
      <circle cx={cartXClamped + CART_W / 4} cy={cartY + CART_H / 2} r={3} fill="#93c5fd" />

      {/* Action arrow */}
      <line
        x1={cartXClamped}
        y1={cartY + 8}
        x2={cartXClamped + arrowDx}
        y2={cartY + 8}
        stroke={arrowColor}
        strokeWidth={3}
        markerEnd={`url(#arrowhead-${action === 1 ? 'right' : 'left'})`}
      />

      {/* Arrowhead markers */}
      <defs>
        <marker id="arrowhead-right" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <polygon points="0 0, 6 3, 0 6" fill="#22c55e" />
        </marker>
        <marker id="arrowhead-left" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto">
          <polygon points="6 0, 0 3, 6 6" fill="#ef4444" />
        </marker>
      </defs>

      {/* State info overlay */}
      <text x={8} y={16} fill="#9ca3af" fontSize={9} fontFamily="monospace">
        x={frame.x.toFixed(3)}m
      </text>
      <text x={8} y={27} fill="#9ca3af" fontSize={9} fontFamily="monospace">
        {'θ'}={( frame.theta * 180 / Math.PI).toFixed(1)}{'°'}
      </text>

      {/* Action label */}
      <text
        x={SVG_W - 8}
        y={16}
        fill={arrowColor}
        fontSize={9}
        fontFamily="monospace"
        textAnchor="end"
      >
        {action === 1 ? 'PUSH RIGHT' : 'PUSH LEFT'}
      </text>
    </svg>
  );
}
