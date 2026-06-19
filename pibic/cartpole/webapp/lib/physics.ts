/**
 * Cart-Pole physics — port exato de cartpole_env.py para execução no browser.
 * Euler integration, mesmos parâmetros do MathWorks / OpenAI Gym.
 */

export const GRAVITY  = 9.8;
export const M_CART   = 1.0;
export const M_POLE   = 0.1;
export const M_TOTAL  = M_CART + M_POLE;
export const L        = 0.5;
export const ML       = M_POLE * L;
export const DT       = 0.02;
export const FORCE_MAX = 10.0;

export const X_LIMIT     = 2.4;
export const THETA_LIMIT = 12.0 * Math.PI / 180; // ≈ 0.2094

export interface CartPoleState {
  x:         number;
  x_dot:     number;
  theta:     number;
  theta_dot: number;
}

export function physicsStep(s: CartPoleState, force: number): CartPoleState {
  const F = Math.max(-FORCE_MAX, Math.min(FORCE_MAX, force));
  const cosT = Math.cos(s.theta);
  const sinT = Math.sin(s.theta);

  const temp   = (F + ML * s.theta_dot ** 2 * sinT) / M_TOTAL;
  const th_acc = (GRAVITY * sinT - cosT * temp) /
                 (L * (4 / 3 - M_POLE * cosT ** 2 / M_TOTAL));
  const x_acc  = temp - ML * th_acc * cosT / M_TOTAL;

  let new_x     = s.x         + DT * s.x_dot;
  let new_x_dot = s.x_dot     + DT * x_acc;

  if (new_x > X_LIMIT) {
    new_x = X_LIMIT;
    new_x_dot = Math.min(0, -new_x_dot * 0.3);
  } else if (new_x < -X_LIMIT) {
    new_x = -X_LIMIT;
    new_x_dot = Math.max(0, -new_x_dot * 0.3);
  }

  return {
    x:         new_x,
    x_dot:     new_x_dot,
    theta:     s.theta     + DT * s.theta_dot,
    theta_dot: s.theta_dot + DT * th_acc,
  };
}

export function isDone(s: CartPoleState): boolean {
  return Math.abs(s.theta) > THETA_LIMIT;
}

export function resetState(): CartPoleState {
  const r = () => (Math.random() - 0.5) * 0.1;
  return { x: r(), x_dot: r(), theta: r(), theta_dot: r() };
}
