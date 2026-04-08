import struct
import random

# Llama2.c Config Header struct:
# int dim; int hidden_dim; int n_layers; int n_heads; int n_kv_heads; int vocab_size; int seq_len;
dim = 16
hidden_dim = 32
n_layers = 1
n_heads = 2
n_kv_heads = 2
vocab_size = 32000
seq_len = 16

head_size = dim // n_heads

# Calculate total floats needed for the weights mapping
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
    # write config header
    f.write(struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))
    
    # generate random floats to simulate weights
    for _ in range(total_floats):
        # writing tiny floats around 0
        v = random.uniform(-0.01, 0.01)
        f.write(struct.pack('f', v))
        
print(f"Generated dummy_model.bin with {total_floats} floats ({(total_floats*4 + 28)/1024/1024:.2f} MB)")
