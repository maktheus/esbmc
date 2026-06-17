/**
 * Neural network inference — executa o actor DDPG no browser.
 * Arquitetura: state(4) → Linear(24) → ReLU → Linear(24) → ReLU → Linear(1) → Tanh × 10
 */

import { FORCE_MAX } from './physics';

export interface DDPGWeights {
  'net.0.weight': number[][];  // 24×4
  'net.0.bias':   number[];    // 24
  'net.2.weight': number[][];  // 24×24
  'net.2.bias':   number[];    // 24
  'net.4.weight': number[][];  // 1×24
  'net.4.bias':   number[];    // 1
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
  const resp = await fetch('/ddpg_weights.json');
  _weights = await resp.json() as DDPGWeights;
  return _weights;
}

export function getForce(state: [number, number, number, number], w: DDPGWeights): number {
  const h1 = linearForward(w['net.0.weight'], w['net.0.bias'], state, 'relu');
  const h2 = linearForward(w['net.2.weight'], w['net.2.bias'], h1,    'relu');
  const out = linearForward(w['net.4.weight'], w['net.4.bias'], h2,    'tanh');
  return out[0] * FORCE_MAX;
}
