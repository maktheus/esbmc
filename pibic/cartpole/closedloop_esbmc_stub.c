/*
 * closedloop_esbmc_stub.c — Esboço de verificação em malha fechada Cart-Pole + DQN
 *
 * TRABALHO FUTURO (Etapa 4)
 *
 * Problema: as equações de movimento do Cart-Pole contêm sin() e cos(),
 * que são transcendentais e não representáveis diretamente em aritmética
 * inteira Q8.8 para o ESBMC.
 *
 * Abordagens possíveis:
 *
 *  (A) Modelo linearizado (válido perto do equilíbrio θ ≈ 0):
 *       sin(θ) ≈ θ,  cos(θ) ≈ 1
 *       → Sistema linear → verificação exata com ESBMC/Boolector
 *
 *  (B) Aproximação polinomial de Taylor (ordem 3):
 *       sin(θ) ≈ θ - θ³/6,  cos(θ) ≈ 1 - θ²/2
 *       → Produto de inteiros → verificável mas mais lento
 *
 *  (C) Intervalo analítico (k passos de simulação):
 *       Propaga bounds do estado por k=5..10 passos usando aritmética
 *       de intervalos; verifica se o estado permanece seguro.
 *
 * Propriedade de segurança a verificar:
 *   ∀ s₀ ∈ S_init . ∀ k ≤ K . s_k ∈ S_safe
 *
 *   S_init = {|x|≤0.5, |ẋ|≤0.1, |θ|≤6°, |θ̇|≤0.1}   (perturbação inicial)
 *   S_safe = {|x|≤2.4, |θ|≤12°}                        (não falha)
 *   K = 50 passos (1 segundo de simulação, DT=0.02)
 */

#include <stdint.h>

/* ── Declarações ESBMC ───────────────────────────────────────────────────── */
void __ESBMC_assume(_Bool cond);
void __ESBMC_assert(_Bool cond, const char *msg);
int  nondet_int(void);

/* ── Escala Q8.8 ─────────────────────────────────────────────────────────── */
#define SCALE    256
#define DT_Q     5       /* DT=0.02 → 0.02×256 = 5.12 ≈ 5 */
#define GRAVITY  2509    /* 9.8 × 256 */
#define ML_Q     13      /* 0.05 × 256 (M_POLE × L = 0.1 × 0.5) */
#define MTOT_Q   282     /* 1.1 × 256 */
#define L_Q      128     /* 0.5 × 256 */
#define FORCE_Q  2560    /* 10.0 × 256 */

/* ── Pesos do controlador (gerados por quantize_controller.py) ───────────── */
/* TODO: preencher com pesos reais após treinamento                           */
/* static int qw1[24][4] = { ... };                                           */
/* static int qb1[24]    = { ... };                                           */
/* ... camada 2, saída ...                                                    */

/* ── Aproximação linear de sin/cos (Abordagem A) ─────────────────────────── */
/* sin(θ) ≈ θ  (válido para |θ| ≤ 12° ≈ 0.21 rad, erro < 0.15%) */
static int sin_approx(int theta_q) { return theta_q; }
static int cos_approx(int theta_q) { (void)theta_q; return SCALE; }  /* cos ≈ 1 */

/* ── Passo de dinâmica linearizada ─────────────────────────────────────────
 *
 * Equações linearizadas (sin θ ≈ θ, cos θ ≈ 1):
 *
 *   θ̈ = (g·θ - F/M_total) / (L·(4/3 - m/M_total))
 *   ẍ = F/M_total - (m·L·θ̈) / M_total
 *
 * Integração de Euler em Q8.8:
 *   x    ← x    + DT × ẋ
 *   ẋ    ← ẋ    + DT × ẍ
 *   θ    ← θ    + DT × θ̇
 *   θ̇   ← θ̇   + DT × θ̈
 */
typedef struct { int x, xd, th, thd; } State;

static State step_linearized(State s, int action) {
    int F = (action == 1) ? FORCE_Q : -FORCE_Q;

    /* θ̈ linearizado (denominador ≈ L × 4/3 × M_total × 256²) */
    int th_acc = (GRAVITY * s.th / SCALE - F * SCALE / MTOT_Q);
    int x_acc  = F / MTOT_Q - (ML_Q * th_acc / SCALE) / MTOT_Q;

    State ns;
    ns.x   = s.x   + (DT_Q * s.xd)  / SCALE;
    ns.xd  = s.xd  + (DT_Q * x_acc) / SCALE;
    ns.th  = s.th  + (DT_Q * s.thd) / SCALE;
    ns.thd = s.thd + (DT_Q * th_acc)/ SCALE;
    return ns;
}

/* ── Inferência do controlador (stub — pesos a preencher) ──────────────────  */
static int controller(State s) {
    /* TODO: implementar forward pass Q8.8 com pesos reais */
    (void)s;
    return nondet_int();   /* placeholder simbólico */
}

/* ── Verificação em malha fechada (K passos) ────────────────────────────── */
#define K_STEPS  50
#define X_SAFE   614    /* 2.4 × 256 */
#define TH_SAFE   53    /* 0.209 × 256 */
#define X_INIT   128    /* 0.5 × 256 */
#define TH_INIT   15    /* 6° × 256/180*π ≈ 15 */

int main(void) {
    /* Estado inicial simbólico (perturbação pequena) */
    State s;
    s.x   = nondet_int(); __ESBMC_assume(s.x   >= -X_INIT  && s.x   <= X_INIT);
    s.xd  = nondet_int(); __ESBMC_assume(s.xd  >= -26      && s.xd  <= 26);  /* ±0.1 m/s */
    s.th  = nondet_int(); __ESBMC_assume(s.th  >= -TH_INIT && s.th  <= TH_INIT);
    s.thd = nondet_int(); __ESBMC_assume(s.thd >= -26      && s.thd <= 26);

    /* Simulação em malha fechada por K passos */
    for (int k = 0; k < K_STEPS; k++) {
        /* Propriedade de segurança a cada passo */
        __ESBMC_assert(s.x  >= -X_SAFE  && s.x  <= X_SAFE,
                       "segurança: carro dentro dos limites");
        __ESBMC_assert(s.th >= -TH_SAFE && s.th <= TH_SAFE,
                       "segurança: angulo dentro dos limites");

        int action = controller(s);
        __ESBMC_assume(action == 0 || action == 1);

        s = step_linearized(s, action);
    }
    return 0;
}
