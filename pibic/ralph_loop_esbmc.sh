#!/bin/bash
# ralph_loop_esbmc.sh
set -e

PRD_FILE="pibic/ralph_prd.json"
MAX_ITERATIONS=5

echo "=================================================="
echo " Starting Ralph Loop (Demo Case 2 & 3)"
echo "=================================================="

# Exporting esbmc binary
export PATH=$PATH:$(pwd)/build/src/esbmc

while true; do
    # Get next task
    TASK_INFO=$(python3 -c "
import json, sys
try:
    with open('$PRD_FILE') as f: prd = json.load(f)
    for i, t in enumerate(prd['tasks']):
        if t['status'] == 'pending':
            print(f\"{i}|{t['id']}|{t['file_to_edit']}|{t['verify_cmd']}\")
            sys.exit(0)
    print('DONE')
except Exception as e: print('ERROR:', e); sys.exit(1)
")
    
    if [ "$TASK_INFO" == "DONE" ]; then
        echo "All tasks completed! Ralph Demo is finished."
        break
    fi
    
    if [[ "$TASK_INFO" == ERROR* ]]; then
        echo "Failed to read PRD: $TASK_INFO"
        exit 1
    fi

    IFS='|' read -r TASK_INDEX TASK_ID TASK_FILE VERIFY_CMD <<< "$TASK_INFO"
    
    echo ""
    echo "--------------------------------------------------"
    echo "Targeting Task: [$TASK_ID] -> $TASK_FILE"
    echo "--------------------------------------------------"
    
    ITERATION=1
    SUCCESS=false
    
    while [ $ITERATION -le $MAX_ITERATIONS ]; do
        echo ">> Iteration $ITERATION"
        
        # 1. AI Generation
        echo "   [AI] Interacting with Agent LLM to apply fixes..."
        python3 pibic/mock_llm.py "$TASK_ID" "$ITERATION" "$TASK_FILE"
        sleep 1
        
        # 2. Verification Step
        echo "   [Verifier] Checking Math/Memory Properties via ESBMC..."
        VERIFY_OUT="/tmp/ralph_verify_out_${TASK_ID}.txt"
        
        set +e
        eval "$VERIFY_CMD" > "$VERIFY_OUT" 2>&1
        VERIFY_CODE=$?
        set -e
        
        if grep -q "VERIFICATION SUCCESSFUL" "$VERIFY_OUT"; then
            echo "   [Verifier - OK] Success! Property holds, no flaws found."
            SUCCESS=true
            break
        else
            echo "   [Verifier - FAIL] Mathematical invariant broken or Counter-example detected!"
            echo "   [Agent Feedback] Feeding the traceback logs back into the LLM context..."
        fi
        
        ITERATION=$((ITERATION + 1))
    done
    
    if [ "$SUCCESS" = true ]; then
        echo "Task $TASK_ID completed successfully."
        # Mark as completed
        python3 -c "
import json
with open('$PRD_FILE') as f: prd = json.load(f)
prd['tasks'][$TASK_INDEX]['status'] = 'completed'
with open('$PRD_FILE', 'w') as f: json.dump(prd, f, indent=4)
"
    else
        echo "Task $TASK_ID failed after $MAX_ITERATIONS iterations."
        exit 1
    fi
done
