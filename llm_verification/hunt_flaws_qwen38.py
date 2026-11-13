import os, sys, struct, json, mmap
import numpy as np
import z3

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print("Python executable (Isolated VENV):", sys.executable)

GGUF_PATH = r"C:\Users\Administrador\.ollama\models\blobs\sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"

GGUF_TYPE_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
def read_val(f, vtype):
    if vtype == 8:
        ln = struct.unpack("<Q", f.read(8))[0]
        return f.read(ln).decode("utf-8", errors="replace")
    elif vtype == 9:
        atype, alen = struct.unpack("<IQ", f.read(12))
        if atype in GGUF_TYPE_SIZES:
            b = f.read(alen * GGUF_TYPE_SIZES[atype])
            if atype in (4,5,6) and alen <= 256:
                fmt = {4:"<I", 5:"<i", 6:"<f"}[atype]
                return [struct.unpack_from(fmt, b, i*4)[0] for i in range(alen)]
            return f"<array type={atype} len={alen}>"
        out = [read_val(f, atype) for _ in range(alen)]
        return out if alen <= 300000 else f"<array len={alen}>"
    else:
        fmt = {0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f", 7:"<?", 10:"<Q", 11:"<q", 12:"<d"}[vtype]
        return struct.unpack(fmt, f.read(GGUF_TYPE_SIZES[vtype]))[0]

with open(GGUF_PATH, "rb") as f:
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    meta = {}
    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        k = f.read(klen).decode("utf-8", errors="replace")
        vtype = struct.unpack("<I", f.read(4))[0]
        meta[k] = read_val(f, vtype)
    tensors = {}
    for _ in range(n_tensors):
        nlen = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nlen).decode("utf-8", errors="replace")
        ndims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = {"dims": dims, "type": ttype, "offset": offset}
    pos = f.tell()
    data_offset = (pos + 31) & ~31

print("=== 1. METADATA & TOKENIZER ANOMALIES ===")
for k, v in sorted(meta.items()):
    if not isinstance(v, list):
        print(f"  META {k}: {v}")
tokens = meta.get("tokenizer.ggml.tokens", [])
print("Total tokens in tokenizer:", len(tokens))
print("Token 246203:", repr(tokens[246203]))
ctrl_tokens = [(i, repr(t)) for i, t in enumerate(tokens) if any(ord(c) < 32 for c in t) or t == ""]
print(f"Control / Empty tokens count: {len(ctrl_tokens)} | Sample: {ctrl_tokens[:15]} ... {ctrl_tokens[-10:]}")

print("\n=== 2. F32 TENSOR ANOMALIES ACROSS ALL 65 LAYERS ===")
f32_tensors = {k: v for k, v in tensors.items() if v["type"] == 0}
print(f"Total F32 tensors: {len(f32_tensors)}")

anomalies = []
with open(GGUF_PATH, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for name, info in sorted(f32_tensors.items()):
        n_el = int(np.prod(info["dims"]))
        off = data_offset + info["offset"]
        arr = np.frombuffer(mm, dtype=np.float32, count=n_el, offset=off).copy()
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        mean_abs = float(np.mean(np.abs(arr)))
        neg_count = int(np.sum(arr < 0))
        near_zero = int(np.sum(np.abs(arr) < 1e-4))
        max_idx = int(np.argmax(np.abs(arr)))
        min_idx = int(np.argmin(arr))
        ratio = max(abs(min_v), abs(max_v)) / (mean_abs + 1e-12)
        anomalies.append({
            "name": name, "dims": info["dims"], "min": min_v, "max": max_v,
            "mean_abs": mean_abs, "max_to_mean_ratio": ratio,
            "neg_count": neg_count, "near_zero_count": near_zero,
            "max_abs_idx": max_idx, "min_idx": min_idx
        })
    mm.close()

anomalies_sorted = sorted(anomalies, key=lambda x: x["max_to_mean_ratio"], reverse=True)
print("Top 20 Extreme Outlier / Anisotropic F32 Tensors (Max/Mean Ratio):")
for a in anomalies_sorted[:20]:
    print(f"  {a['name']} | dims={a['dims']} | min={a['min']:.6f} (idx {a['min_idx']}) | max={a['max']:.6f} (idx {a['max_abs_idx']}) | mean_abs={a['mean_abs']:.4f} | ratio={a['max_to_mean_ratio']:.2f}x | neg={a['neg_count']} | near_zero={a['near_zero_count']}")

neg_norms = [a for a in anomalies if "norm" in a["name"] and a["neg_count"] > 0]
print(f"\nNormalization tensors with NEGATIVE weights (Phase Inversion): {len(neg_norms)}")
for a in sorted(neg_norms, key=lambda x: x["min"])[:20]:
    print(f"  {a['name']} | min={a['min']:.6f} (idx {a['min_idx']}) | max={a['max']:.6f} | neg_count={a['neg_count']}/{int(np.prod(a['dims']))} | near_zero={a['near_zero_count']}")

near_zero_norms = [a for a in anomalies if "norm" in a["name"] and a["near_zero_count"] > 0]
print(f"\nNormalization tensors with NEAR-ZERO / DEAD channels (|w| < 1e-4): {len(near_zero_norms)}")
for a in sorted(near_zero_norms, key=lambda x: x["near_zero_count"], reverse=True)[:15]:
    print(f"  {a['name']} | min={a['min']:.6f} | max={a['max']:.6f} | near_zero_count={a['near_zero_count']}/{int(np.prod(a['dims']))}")

ssm_tensors = [a for a in anomalies if "ssm" in a["name"]]
print(f"\nSSM (Mamba-2 / Gated-DeltaNet) F32 Tensors ({len(ssm_tensors)}):")
for a in ssm_tensors[:25]:
    print(f"  {a['name']} | dims={a['dims']} | min={a['min']:.6f} | max={a['max']:.6f} | mean_abs={a['mean_abs']:.4f} | ratio={a['max_to_mean_ratio']:.2f}x | neg={a['neg_count']} | near_zero={a['near_zero_count']}")
