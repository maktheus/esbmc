'use client';

import React from 'react';

interface QValueBarProps {
  q0:     number;
  q1:     number;
  action: number;
}

export default function QValueBar({ q0, q1, action }: QValueBarProps) {
  const maxAbs = Math.max(Math.abs(q0), Math.abs(q1), 0.001);

  function barWidth(q: number) {
    return Math.abs(q) / maxAbs * 100;
  }

  const labels = ['Q0 (←)', 'Q1 (→)'];
  const values = [q0, q1];
  const colors = ['#ef4444', '#22c55e'];

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-400 uppercase tracking-wider">Q-Values</p>
      {values.map((q, i) => (
        <div key={i}>
          <div className="flex justify-between text-xs mb-1">
            <span className={action === i ? 'text-white font-bold' : 'text-gray-400'}>
              {labels[i]}
              {action === i && (
                <span className="ml-1 text-yellow-400">&#9654;</span>
              )}
            </span>
            <span className="font-mono text-gray-300">{q.toFixed(4)}</span>
          </div>
          <div className="bg-gray-700 rounded h-4 overflow-hidden">
            <div
              className="h-full rounded transition-all duration-100"
              style={{
                width: `${barWidth(q)}%`,
                background: colors[i],
                opacity: action === i ? 1 : 0.4,
              }}
            />
          </div>
        </div>
      ))}
      <div className="pt-1 border-t border-gray-700">
        <p className="text-xs text-gray-500">
          Ação selecionada:{' '}
          <span className={action === 1 ? 'text-green-400' : 'text-red-400'} style={{fontWeight:'bold'}}>
            {action === 1 ? '1 (direita)' : '0 (esquerda)'}
          </span>
        </p>
      </div>
    </div>
  );
}
