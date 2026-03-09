import json
import os

filepath = "/home/uchoa/esbmc/pibic/prd.json"

if not os.path.exists(filepath):
    print("Error: prd.json not found at", filepath)
    exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

start_id = len(data.get("tasks", [])) + 1
new_tasks = []
categories = [
    "Validation: Continuous Integration",
    "Core Wrapper: Async Execution",
    "Model Translators: Graph Optimization",
    "Inference Safety Engine: Quantization Bounds",
    "Control Systems: Adaptive PID",
    "Ralph Loop: Multi-Agent Synchronization",
    "Data Parsing: Abstract Syntax Tree Analysis",
    "Security: Malformed Input Rejection",
    "Logging & Distributed Tracing"
]

for i in range(100):
    task_id = f"TSK_{start_id + i:03d}"
    category = categories[i % len(categories)]
    
    if "Validation" in category:
         desc = f"Develop automated CI pipeline test case {i+1} for robust verification of edge regressions."
    elif "Inference Safety Engine" in category:
         desc = f"Write C++ bound-checking stub for INT8 quantized kernel edge case {i+1}."
    elif "Ralph Loop" in category:
         desc = f"Implement multi-agent synchronization primitive {i+1} for isolated generation and verification passes."
    elif "Model Translators" in category:
         desc = f"Implement ONNX AST parser enhancement {i+1} focusing on fused operations."
    elif "Control Systems" in category:
         desc = f"Verify PID convergence loop invariants under dynamical condition {i+1}."
    else:
         desc = f"Enhance module {category.split(':')[0]} with feature specification {i+1}."

    new_tasks.append({
        "id": task_id,
        "category": category,
        "description": desc,
        "status": "pending"
    })

data["tasks"].extend(new_tasks)
data["total_tasks"] = len(data["tasks"])

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Successfully added {len(new_tasks)} new tasks. Total tasks count updated to {data['total_tasks']}.")
