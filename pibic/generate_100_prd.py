import json

tasks = []
tid = 1

def add_t(cat, desc, status="pending"):
    global tid
    tasks.append({"id": f"TSK_{tid:03d}", "category": cat, "description": desc, "status": status})
    tid += 1

# 1. Context & Setup (5)
add_t("Context", "Read 'analysis_report.md' and comprehend the baseline results for NN Verification via SMT.")
add_t("Context", "Read 'roadmap.md' to understand the 4 phase Python porting design.")
add_t("Setup", "Initialize Python Virtual Environment with strict requirements.txt (pytest, onnx, torch, etc).")
add_t("Setup", "Configure pyproject.toml for the 'core_verify' library packaging.")
add_t("Setup", "Set up absolute paths to the local ESBMC binary compiled in the root directory.")

# 2. ESBMC Core Wrapper (CLI Flags) (15)
flags = [
    "--z3", "--bitwuzla", "--mathsat", "--cvc4", "--floatbv", "--fixedbv",
    "--unwind", "--no-unwinding-assertions", "--memory-leak-check", 
    "--overflow-check", "--bounds-check", "--pointer-check", 
    "--multi-property", "--smt-formula-only", "--k-induction"
]
for f in flags:
    add_t("Core Wrapper: Flags", f"Implement Python API argument parser and validation for ESBMC flag '{f}' in esbmc_caller.py.")

# 3. ESBMC SMT Feedback Parser (15)
traces = [
    "array out of bounds", "arithmetic overflow", "memory leak", "division by zero", 
    "pointer dereference", "assertion failure", "nan/inf float detection", 
    "loop unwinding failure", "target timeout", "solver abort",
    "uninitialized variable reading", "double free", "use after free", 
    "deadlock detection", "data race"
]
for tr in traces:
    add_t("Core Wrapper: Feedback Parser", f"Implement regex extraction and isolated feedback logging when ESBMC detects a '{tr}' failure.")

# 4. Model Context & ONNX/Torch Converters (20)
layers = [
    "Dense/Linear", "Conv1D", "Conv2D", "Conv3D", "ReLU", "GELU", "Swish", 
    "Sigmoid", "Softmax", "Flatten", "MaxPool2D", "AvgPool2D", "GlobalAvgPool2D", 
    "Dropout", "BatchNorm", "LayerNorm", "Embedding", "LSTM", "GRU", "TransformerEncoder_QKV"
]
for l in layers:
    add_t("Model Translators: ONNX to C", f"Implement AST parser mapping the '{l}' neural network layer into a mathematically verifiable C static inline function.")

# 5. Inference Benchmarks & Kernels (15)
kernels = [
    "GEMM naive", "GEMM tiled", "GEMM Strassen", "Attention Softmax", "Self-Attention Scoring", 
    "Vector Add", "Vector Mul", "Reduce Sum", "Reduce Max", "Matrix Transpose", 
    "Im2Col Pattern", "Depthwise Separable Conv", "Activation forward loop", 
    "Activation backward proxy", "Loss MSE computation bounds"
]
for k in kernels:
    add_t("Inference Safety Engine", f"Write a C++ benchmark stub and Python Pytest runner for formal limits verification in '{k}' kernel.")

# 6. Control System & Chaos Engineering (10)
noise_types = ["Uniform", "Gaussian", "Impulse", "Drift", "Sinusoidal"]
for nt in noise_types:
    add_t("Control Systems: Chaos Generator", f"Implement Python logic to inject '{nt}' sensor noise mathematically using ESBMC non-deterministic floats.")
    add_t("Control Systems: Chaos Verifier", f"Write validation checks to ensure PID convergence and boundary limits hold under '{nt}' noise profiles.")

# 7. Agentic Verification (The Ralph Loop) (15)
agent_components = [
    "State Manager (JSON Updater)", "PRD Loader & Parser", 
    "LLM API Connector (OpenAI Provider)", "LLM API Connector (Anthropic Provider)", 
    "LLM Rate Limiter & Token Counter", "Prompt Context Formatter", 
    "ESBMC Trace Formatter (Feedback Injector)", "Git Auto-Committer", 
    "Diff & Unified Patch Parser", "Code Block Extractor (Markdown regex)", 
    "Validation Gate (Subprocess Caller)", "File System Sandbox isolator", 
    "Agent Timeout Monitor", "Ralph Loop Main Orchestrator", "Artifact Generator (Final Report)"
]
for ac in agent_components:
    add_t("Ralph Loop: Autonomy", f"Implement the '{ac}' class module required for the fully autonomous Ralph agent workflow in Python.")

# 8. Testing & Validation Suite (15)
for k in kernels[:5]:
    add_t("Validation", f"Write end-to-end Pytest suite verifying ESBMC False-Positive rejection capabilities on the '{k}' kernel.")
for tr in traces[:5]:
    add_t("Validation", f"Write automated mock-test injecting a '{tr}' string into the Parser and ensuring the AI state manager isolates the trace correctly.")
add_t("Validation", "Setup GitHub Actions or local CI runner config script for running the >100 Pytest ESBMC tests seamlessly.")
add_t("Validation", "Write pipeline performance benchmarking test for execution time bounds via Python `time` monitor.")
add_t("Documentation", "Write complete API DocStrings for all Python Core functions.")
add_t("Documentation", "Document setup instructions for Windows/WSL users attempting to wrap the compiled ESBMC binary.")
add_t("Documentation", "Finalize comprehensive Architecture markdown document comparing the final Python Implementation vs the previous Bash proof of concept.")

prd = {
    "project": "ESBMC Neuro-Symbolic Pipeline (Python Edition)",
    "total_tasks": len(tasks),
    "tasks": tasks
}

with open('pibic/prd.json', 'w') as f:
    json.dump(prd, f, indent=4)

print(f"Generated {len(tasks)} tasks successfully.")
