/**
 * Float32 DDPG inference — referência não-verificada para comparação.
 * Arquitetura: state(4) → Linear(24) → ReLU → Linear(24) → ReLU → Linear(1) → Tanh × 10
 */

import { FORCE_MAX } from './physics';
import { IController } from './types';

export interface DDPGWeights {
  'net.0.weight': number[][];
  'net.0.bias':   number[];
  'net.2.weight': number[][];
  'net.2.bias':   number[];
  'net.4.weight': number[][];
  'net.4.bias':   number[];
}

function relu(x: number): number { return x > 0 ? x : 0; }

function linearForward(
  W: number[][], b: number[], input: number[], activation: 'relu' | 'tanh' | 'none'
): number[] {
  const out: number[] = [];
  for (let i = 0; i < W.length; i++) {
    let sum = b[i];
    for (let j = 0; j < input.length; j++) {
      sum += W[i][j] * input[j];
    }
    if (activation === 'relu')      out.push(relu(sum));
    else if (activation === 'tanh') out.push(Math.tanh(sum));
    else                            out.push(sum);
  }
  return out;
}

let _weights: DDPGWeights | null = null;

export async function loadWeights(): Promise<DDPGWeights> {
  if (_weights) return _weights;
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  const resp = await fetch(`${base}/ddpg_weights.json`);
  _weights = await resp.json() as DDPGWeights;
  return _weights;
}

export function getForce(state: [number, number, number, number], w: DDPGWeights): number {
  const h1 = linearForward(w['net.0.weight'], w['net.0.bias'], state, 'relu');
  const h2 = linearForward(w['net.2.weight'], w['net.2.bias'], h1,    'relu');
  const out = linearForward(w['net.4.weight'], w['net.4.bias'], h2,    'tanh');
  return out[0] * FORCE_MAX;
}

export class FloatDDPGController implements IController {
  readonly name = 'DDPG Float32 (referência)';
  readonly isVerified = false;
  private w: DDPGWeights;

  constructor(w: DDPGWeights) { this.w = w; }

  getForce(state: [number, number, number, number]): number {
    return getForce(state, this.w);
  }
}
