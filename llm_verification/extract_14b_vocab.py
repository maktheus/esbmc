# -*- coding: utf-8 -*-
import gguf

path = r"C:\Users\Administrador\.ollama\models\blobs\sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
reader = gguf.GGUFReader(path)
tokens_field = reader.fields.get('tokenizer.ggml.tokens')
vocab_list = [p.tobytes().decode('utf-8', errors='replace') for p in tokens_field.parts]
token_to_id = {t: idx for idx, t in enumerate(vocab_list)}

targets = ['I', 'sudo', 'rm', 'DROP', 'password', 'eval', 'root', 'bash', 'kill', 'chmod', 'curl', 'wget', 'exec', 'system']
for t in targets:
    tid = token_to_id.get(t, token_to_id.get(' ' + t, None))
    print(f'Token "{t}": ID = {tid}')
