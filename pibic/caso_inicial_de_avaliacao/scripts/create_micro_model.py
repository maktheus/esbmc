import struct
import random

dim = 2
hidden_dim = 2
n_layers = 1
n_heads = 1
n_kv_heads = 1
vocab_size = 64
seq_len = 2
head_size = dim // n_heads

total_floats = 0
total_floats += vocab_size * dim
total_floats += n_layers * dim
total_floats += n_layers * dim * (n_heads * head_size)
total_floats += n_layers * dim * (n_kv_heads * head_size)
total_floats += n_layers * dim * (n_kv_heads * head_size)
total_floats += n_layers * (n_heads * head_size) * dim
total_floats += n_layers * dim
total_floats += n_layers * dim * hidden_dim
total_floats += n_layers * hidden_dim * dim
total_floats += n_layers * dim * hidden_dim
total_floats += dim
total_floats += seq_len * head_size

with open('src/dummy_model.bin', 'wb') as f:
    f.write(struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))
    for _ in range(total_floats):
        f.write(struct.pack('f', random.uniform(-0.01, 0.01)))
        
print(f"Generated Micro Model with {total_floats} floats")
