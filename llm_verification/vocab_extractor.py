# -*- coding: utf-8 -*-
"""
Extrator de Vocabulário e Pesos de Saída (Unembedding) do Qwen2.5-7B (8B-class)
Mapeia tokens de segurança (Safe vs Unsafe / Exploits) diretamente do arquivo GGUF.
"""

import os
import gguf
import numpy as np
from gguf.quants import dequantize

GGUF_PATH = r"C:\Users\Administrador\.ollama\models\blobs\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"

def get_vocab_and_weights():
    reader = gguf.GGUFReader(GGUF_PATH)
    tokens_field = reader.fields.get('tokenizer.ggml.tokens')
    if not tokens_field:
        raise ValueError("tokenizer.ggml.tokens não encontrado no GGUF")
    
    total_tokens = len(tokens_field.parts)
    print(f"[VOCAB] Total de tokens no modelo: {total_tokens}")
    
    # Mapear strings dos tokens
    vocab_list = [p.tobytes().decode('utf-8', errors='replace') for p in tokens_field.parts]
    token_to_id = {t: idx for idx, t in enumerate(vocab_list)}
    
    # Encontrar output.weight
    output_tensor = None
    for t in reader.tensors:
        if t.name == 'output.weight':
            output_tensor = t
            break
            
    if output_tensor is None:
        raise ValueError("output.weight não encontrado")
        
    print(f"[TENSOR] output.weight: shape={output_tensor.shape}, type={output_tensor.tensor_type}")
    
    # Categorias de Tokens para Verificação Formal de Segurança
    target_tokens = {
        # Safe Tokens (Respostas alinhadas e de recusa ética)
        "safe_I": "I",
        "safe_Sorry": "Sorry",
        "safe_Sure": "Sure",
        "safe_The": "The",
        "safe_As": "As",
        "safe_Certainly": "Certainly",
        "safe_cannot": "cannot",
        
        # Unsafe / Jailbreak / Command Injection Tokens
        "bad_sudo": "sudo",
        "bad_rm": "rm",
        "bad_DROP": "DROP",
        "bad_password": "password",
        "bad_eval": "eval",
        "bad_sh": "sh",
        "bad_bash": "bash",
        "bad_chmod": "chmod",
        "bad_kill": "kill",
        "bad_root": "root"
    }
    
    resolved_ids = {}
    for label, tok in target_tokens.items():
        if tok in token_to_id:
            resolved_ids[label] = token_to_id[tok]
            print(f"  [MAP] {label} ('{tok}') -> ID {token_to_id[tok]}")
        else:
            # Tentar token com espaço na frente
            tok_space = " " + tok
            if tok_space in token_to_id:
                resolved_ids[label] = token_to_id[tok_space]
                print(f"  [MAP] {label} ('{tok_space}') -> ID {token_to_id[tok_space]}")
            else:
                print(f"  [AVISO] '{tok}' não encontrado diretamente.")

    # Dequantizar apenas as linhas selecionadas com copy() para garantir contiguidade
    weights = {}
    for label, tid in resolved_ids.items():
        if tid < output_tensor.data.shape[0]:
            raw_row = output_tensor.data[tid:tid+1].copy()
            deq = dequantize(raw_row, output_tensor.tensor_type) # shape: (1, 3584)
            weights[label] = deq[0].astype(np.float32)
        else:
            print(f"  [AVISO] Token ID {tid} excede linhas de output.weight ({output_tensor.data.shape[0]})")
        
    print(f"[EXTRACAO] {len(weights)} vetores de unembedding (dim=3584) extraídos com sucesso.")
    return resolved_ids, weights

if __name__ == "__main__":
    ids, w = get_vocab_and_weights()
    print("Sucesso! Vetor sample safe_I:", w["safe_I"][:5])
