import json
import os

filepath = "/home/uchoa/esbmc/pibic/prd.json"

categories = {
    "Architecture: Core Services": "Investigate and document role of {}",
    "Architecture: DisplayPowerController": "Analyze cover display power state transitions in {}",
    "Architecture: WindowManager": "Document multi-display window routing in {}",
    "Architecture: Input & Focus": "Explain input event dispatching for external CLI display via {}",
    "SystemUI Integration": "Analyze SystemUI component {} behavior on cover display",
    "Foldable States & Sensors": "Document hinge angle sensor interaction with {}",
    "Ralph Loop: State Validation": "Create Ralph Loop to verify {} during fold/unfold",
    "Ralph Loop: Power Validation": "Create Ralph Loop to test sleep/wake signals for {}",
    "Documentation: Slides": "Draft presentation slide content for {}",
    "Documentation: Markdown Handbooks": "Write detailed technical section on {}"
}

placeholder_targets = [
    "DisplayManagerService", "DisplayPowerController", "LogicalDisplayMapper",
    "ActivityTaskManagerService", "WindowManagerService", "InputDispatcher",
    "KeyguardDisplayManager", "NotificationShade", "DeviceFoldStateProvider",
    "DualDisplayPolicy", "DisplayDeviceInfo", "SurfaceFlinger",
    "PowerManagerService", "DreamManager", "Wakelock interactions",
    "DisplayBlanker", "ColorFade", "ScreenOffAnimation", "AOD (Always On Display)",
    "Biometrics on Cover Display"
]

tasks = []

# Task 1-200
for i in range(200):
    cat_keys = list(categories.keys())
    category = cat_keys[i % len(cat_keys)]
    target = placeholder_targets[i % len(placeholder_targets)]
    
    desc_template = categories[category]
    desc = desc_template.format(target) + f" (Module {i//20 + 1}, Case {i+1})."
    
    tasks.append({
        "id": f"TSK_{i+1:03d}",
        "category": category,
        "description": desc,
        "status": "pending"
    })

data = {
    "project": "CLI (Cover Display) Framework Architecture & Educational Ralph Loops",
    "total_tasks": len(tasks),
    "tasks": tasks
}

# Create dir if not exists (should exist)
os.makedirs(os.path.dirname(filepath), exist_ok=True)

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Successfully re-created PRD at {filepath} with {len(tasks)} cases focused on Cover Display (CLI).")
