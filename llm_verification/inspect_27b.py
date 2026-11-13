import sys
import time
import gguf

GGUF_PATH = r"C:\Users\Administrador\.ollama\models\blobs\sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"

print(f"Reading GGUF reader from {GGUF_PATH}...")
t0 = time.perf_counter()
reader = gguf.GGUFReader(GGUF_PATH)
print(f"Header loaded in {time.perf_counter() - t0:.2f}s.")
print(f"Number of tensors: {len(reader.tensors)}")

tensor_names = [t.name for t in reader.tensors]
print(f"Sample tensor names (first 10): {tensor_names[:10]}")
print(f"Sample tensor names (last 10): {tensor_names[-10:]}")

# Check output weight and norms
for key in ['output.weight', 'output_norm.weight', 'token_embd.weight']:
    if key in tensor_names:
        t = [t for t in reader.tensors if t.name == key][0]
        print(f"Tensor {key}: shape={t.shape}, type={t.tensor_type}")

# Check block count
block_ids = set()
for name in tensor_names:
    if name.startswith('blk.'):
        b_id = int(name.split('.')[1])
        block_ids.add(b_id)
print(f"Detected {len(block_ids)} transformer blocks: 0 to {max(block_ids) if block_ids else 'none'}")
