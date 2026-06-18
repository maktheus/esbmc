/**
 * Controlador DDPG quantizado Q8.8 — mesma aritmética que o ESBMC verifica.
 *
 * Divisão por truncação em direção a zero (cdiv) = operador '/' do C.
 * Piecewise linear tanh com segmentos idênticos ao harness C.
 * Contraexemplos do ESBMC reproduzem exatamente neste controlador.
 */

import { IController } from './types';

const SCALE = 256;

export interface QuantizedWeights {
  w1: number[][]; b1: number[];
  w2: number[][]; b2: number[];
  w_out: number[][]; b_out: number[];
}

function cdiv(a: number, b: number): number {
  return Math.trunc(a / b);
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

export function tanhQ88(z: number): number {
  const z_abs = Math.abs(z);
  let t: number;
  if (z_abs <= 64)        t = cdiv(z_abs * 252, 256);
  else if (z_abs <= 192)  t = 62 + cdiv((z_abs - 64) * 200, 256);
  else if (z_abs <= 384)  t = 162 + cdiv((z_abs - 192) * 92, 256);
  else if (z_abs <= 768)  t = 231 + cdiv((z_abs - 384) * 16, 256);
  else                    t = 255;
  return z >= 0 ? t : -t;
}

export function getForceQ88(
  state: [number, number, number, number],
  w: QuantizedWeights
): number {
  const h1: number[] = [];
  for (let i = 0; i < w.b1.length; i++) {
    let pre = w.b1[i];
    for (let j = 0; j < 4; j++) pre += cdiv(state[j] * w.w1[i][j], SCALE);
    h1.push(relu(pre));
  }

  const h2: number[] = [];
  for (let i = 0; i < w.b2.length; i++) {
    let pre = w.b2[i];
    for (let j = 0; j < h1.length; j++) pre += cdiv(h1[j] * w.w2[i][j], SCALE);
    h2.push(relu(pre));
  }

  let z = w.b_out[0];
  for (let j = 0; j < h2.length; j++) z += cdiv(h2[j] * w.w_out[0][j], SCALE);

  const tanh_z = tanhQ88(z);
  return cdiv(tanh_z * 10 * SCALE, SCALE);
}

export function stateToQ88(s: [number, number, number, number]): [number, number, number, number] {
  return [
    Math.round(s[0] * SCALE),
    Math.round(s[1] * SCALE),
    Math.round(s[2] * SCALE),
    Math.round(s[3] * SCALE),
  ];
}

export function forceFromQ88(f_q: number): number {
  return f_q / SCALE;
}

let _qWeights: QuantizedWeights | null = null;

export async function loadQuantizedWeights(): Promise<QuantizedWeights> {
  if (_qWeights) return _qWeights;
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  const resp = await fetch(`${base}/ddpg_weights_q88.json`);
  _qWeights = await resp.json() as QuantizedWeights;
  return _qWeights;
}

export class QuantizedDDPGController implements IController {
  readonly name = 'DDPG Q8.8 (verificado)';
  readonly isVerified = true;
  private w: QuantizedWeights;

  constructor(w: QuantizedWeights) { this.w = w; }

  getForce(state: [number, number, number, number]): number {
    const sq = stateToQ88(state);
    const fq = getForceQ88(sq, this.w);
    return forceFromQ88(fq);
  }
}
