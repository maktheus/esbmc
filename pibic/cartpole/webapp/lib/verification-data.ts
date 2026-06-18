/**
 * Carrega e parseia resultados de verificação ESBMC (SRP).
 */

import { DDPGVerificationData } from './types';

let _cache: DDPGVerificationData | null = null;

export async function loadVerificationData(): Promise<DDPGVerificationData> {
  if (_cache) return _cache;
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  const resp = await fetch(`${base}/ddpg_verification_data.json`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  _cache = await resp.json() as DDPGVerificationData;
  return _cache;
}
