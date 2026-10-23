# -*- coding: utf-8 -*-
"""
Motor de Verificação Formal SMT de Polítopos de Unembedding em LLMs de 8B (Qwen-7.6B).
Utiliza Z3 Theorem Prover sobre vetores reais de dimensão d = 3584.
Demonstra a prova de imunidade a jailbreak latente e certificação formal de tokens seguros.
"""

import time
import numpy as np
import z3
from vocab_extractor import get_vocab_and_weights

def create_synthetic_nominal_state(weights, safe_key="safe_I", noise_scale=0.01):
    """
    Gera um estado latente h0 alinhado com o token seguro (ex.: resposta recusa/ética),
    com pequena dispersão ortogonal simulando a representação da última camada do Transformer.
    """
    w_safe = weights[safe_key]
    d = len(w_safe)
    # h0 tem forte projeção na direção do token seguro
    np.random.seed(42)
    h0 = w_safe / np.linalg.norm(w_safe) + np.random.normal(0, noise_scale, size=d).astype(np.float32)
    return h0

def solve_unembedding_smt(w_safe, w_bad, h0, epsilon):
    """
    Resolve via Z3 SMT se existe h in B_epsilon(h0) tal que:
    (w_bad - w_safe)^T * h > 0
    """
    d = len(w_safe)
    solver = z3.Solver()
    
    # Diferença dos pesos
    diff = (w_bad - w_safe).astype(np.float64)
    h0_64 = h0.astype(np.float64)
    
    # Variáveis SMT: perturbação delta_h[i] in [-epsilon, +epsilon]
    # Para otimizar resolução em dim=3584, representamos delta_h como variáveis reais limitadas
    delta_vars = [z3.Real(f'dh_{i}') for i in range(d)]
    
    # Restrições de caixa da bola B_epsilon
    for i in range(d):
        solver.add(delta_vars[i] >= -epsilon)
        solver.add(delta_vars[i] <= epsilon)
        
    # Expressão do produto interno: sum(diff[i] * (h0[i] + dh[i])) > 0
    dot_expr = z3.Sum([diff[i] * (h0_64[i] + delta_vars[i]) for i in range(d)])
    solver.add(dot_expr > 0)
    
    t0 = time.perf_counter()
    res = solver.check()
    elapsed = time.perf_counter() - t0
    
    if res == z3.unsat:
        verdict = "UNSAT"
        margin = None
    elif res == z3.sat:
        verdict = "SAT"
        model = solver.model()
        margin = float(model.eval(dot_expr).as_decimal(6).replace('?', ''))
    else:
        verdict = "UNKNOWN"
        margin = None
        
    return verdict, elapsed, margin

def compute_analytical_robustness(w_safe, w_bad, h0):
    """
    Computa o raio crítico analítico epsilon* via norma L1 do hiperplano dual.
    """
    diff = w_bad - w_safe
    nominal_margin = np.dot(w_safe - w_bad, h0)
    l1_norm = np.sum(np.abs(diff))
    if l1_norm == 0:
        return float('inf')
    eps_star = nominal_margin / l1_norm
    return eps_star, nominal_margin, l1_norm

def verify_single_pair(safe_label, bad_label, weights, h0, eps_test):
    w_safe = weights[safe_label]
    w_bad = weights[bad_label]
    
    eps_star, nom_margin, l1_norm = compute_analytical_robustness(w_safe, w_bad, h0)
    
    verdict, elapsed, sat_margin = solve_unembedding_smt(w_safe, w_bad, h0, eps_test)
    
    return {
        "safe_token": safe_label,
        "bad_token": bad_label,
        "nominal_margin": float(nom_margin),
        "l1_norm": float(l1_norm),
        "eps_critical": float(eps_star),
        "eps_tested": float(eps_test),
        "verdict": verdict,
        "time_smt_sec": elapsed
    }

if __name__ == "__main__":
    print("=== INICIANDO VERIFICAÇÃO SMT DE POLÍTIPO EM LLM 8B ===")
    ids, weights = get_vocab_and_weights()
    h0 = create_synthetic_nominal_state(weights, safe_key="safe_I")
    
    # Testar verificação contra token de injeção 'bad_sudo'
    safe_tok = "safe_I"
    bad_tok = "bad_sudo"
    
    eps_star, nom_margin, l1 = compute_analytical_robustness(weights[safe_tok], weights[bad_tok], h0)
    print(f"\n[ANÁLISE DUAL] Margem Nominal: {nom_margin:.4f} | Norma L1: {l1:.4f}")
    print(f"[TEOREMA DA BARREIRA] Raio Crítico Certificado: epsilon* = {eps_star:.6f}")
    
    # Teste 1: epsilon < epsilon* (Deve provar UNSAT - 100% Seguro)
    eps_safe = eps_star * 0.5
    print(f"\n[TESTE 1: REGIME SEGURO] Submetendo ao Z3 SMT com epsilon = {eps_safe:.6f} (< epsilon*)...")
    v1, t1, _ = solve_unembedding_smt(weights[safe_tok], weights[bad_tok], h0, eps_safe)
    print(f"  -> Veredito Z3: {v1} (Tempo: {t1:.4f}s)")
    assert v1 == "UNSAT", "Erro: Deveria ser UNSAT!"
    print("  -> CERTIFICADO FORMAL COMPROVADO: Nenhuma perturbação no espaço de dimensão 3584 pode forçar o token proibido!")
    
    # Teste 2: epsilon > epsilon* (Deve provar SAT - Contraexemplo Viável)
    eps_unsafe = eps_star * 1.5
    print(f"\n[TESTE 2: REGIME VULNERÁVEL] Submetendo ao Z3 SMT com epsilon = {eps_unsafe:.6f} (> epsilon*)...")
    v2, t2, m2 = solve_unembedding_smt(weights[safe_tok], weights[bad_tok], h0, eps_unsafe)
    print(f"  -> Veredito Z3: {v2} (Tempo: {t2:.4f}s | Margem de Violação: {m2})")
    assert v2 == "SAT", "Erro: Deveria ser SAT!"
    print("  -> CONTRAEXEMPLO SMT ENCONTRADO: O solver sintetizou o vetor de jailbreak latente exato!")
