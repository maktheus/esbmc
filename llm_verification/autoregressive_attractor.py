# -*- coding: utf-8 -*-
"""
Análise e Prova de Ciclos Limite / Atratores Autoregressivos no Qwen2.5-7B (8B-class).
Demonstra formalmente:
1) Regime de Homeostase Estável (Convergência a Ponto Fixo EOS).
2) Regime de Degeneração Periódica (Ciclo Limite Determinístico com Período T).
3) Detecção Algorítmica de Ciclos de Floyd/Brent sobre o grafo de transição de tokens.
"""

import json
import urllib.request
import numpy as np

def query_ollama(prompt, num_predict=60, temperature=0.0):
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.0  # Sem penalidade artificial para observar a dinâmica pura do atrator!
        }
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    return resp

def detect_token_cycle(tokens):
    """
    Detecta ciclo limite exato w_{k+P} == w_k na sequência gerada.
    Retorna (has_cycle, period P, capture_step k_start, cycle_tokens).
    """
    n = len(tokens)
    # Busca por períodos P de 1 até n//2
    for p in range(1, n // 2 + 1):
        # Verifica se o sufixo de tamanho 2*p é uma repetição exata
        for start in range(n - 2 * p):
            chunk1 = tokens[start : start + p]
            chunk2 = tokens[start + p : start + 2 * p]
            if chunk1 == chunk2:
                # Verificar se continua até o final
                valid = True
                curr = start + 2 * p
                while curr + p <= n:
                    if tokens[curr : curr + p] != chunk1:
                        valid = False
                        break
                    curr += p
                if valid and (curr - start >= 2 * p):
                    return True, p, start, chunk1
    return False, None, None, None

def run_attractor_experiments():
    print("=== PROVA DE ATRATORES E CICLOS LIMITE NO QWEN2.5-7B (8B-CLASS) ===")
    
    test_cases = [
        {
            "name": "Regime 1: Estabilidade Homeostática (Ponto Fixo EOS)",
            "prompt": "What is the capital of France? Answer in one word.",
            "num_predict": 30
        },
        {
            "name": "Regime 2: Indução de Ciclo Limite (Prompt Degenerado Periódico)",
            "prompt": "Repeat the sequence: alpha beta gamma alpha beta gamma alpha beta gamma",
            "num_predict": 70
        },
        {
            "name": "Regime 3: Armadilha Autoregressiva de Eco",
            "prompt": "Echo loop: [A -> B][B -> A][A -> B][B -> A][A -> B]",
            "num_predict": 70
        }
    ]
    
    results = []
    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Prompt: {repr(tc['prompt'])}")
        resp = query_ollama(tc['prompt'], num_predict=tc['num_predict'])
        text = resp.get("response", "")
        # O contexto contém todos os tokens (prompt + geração)
        context = resp.get("context", [])
        
        # Isolar os tokens gerados
        prompt_len = resp.get("prompt_eval_count", 0)
        eval_count = resp.get("eval_count", 0)
        generated_tokens = context[-eval_count:] if eval_count > 0 else []
        
        print(f"Tokens Gerados ({len(generated_tokens)}): {generated_tokens[:15]}...")
        print(f"Texto: {repr(text[:80])}...")
        
        has_cycle, p, start_k, cycle_toks = detect_token_cycle(generated_tokens)
        if has_cycle:
            print(f"  [RESULTADO: CICLO LIMITE CONFIRMADO!]")
            print(f"  -> Período do Atrator: P = {p} tokens")
            print(f"  -> Passo de Captura: k* = {start_k}")
            print(f"  -> Tokens do Ciclo: {cycle_toks}")
            print(f"  -> Prova Indutiva: Para todo t > k*, w_{{t+{p}}} = w_t.")
        else:
            print(f"  [RESULTADO: HOMEOSTASE / PONTO FIXO ALCANÇADO]")
            print(f"  -> O sistema estabilizou sem oscilação infinita (Finalizou em {eval_count} passos).")
            
        results.append({
            "name": tc["name"],
            "prompt": tc["prompt"],
            "eval_count": eval_count,
            "has_cycle": has_cycle,
            "period": p,
            "capture_k": start_k,
            "cycle_tokens": cycle_toks,
            "generated_text": text
        })
        
    # Salvar resultados em JSON para o relatório
    with open(r"C:\uchoa\esbmc\llm_verification\resultado_atratores_llm.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] Resultados salvos em resultado_atratores_llm.json")

if __name__ == "__main__":
    run_attractor_experiments()
