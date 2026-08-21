#!/bin/bash
# ralph_loop_esbmc.sh
set -euo pipefail

PRD_FILE="pibic/ralph_prd.json"
MAX_ITERATIONS=5

echo "=================================================="
echo " Starting Ralph Loop (Demo Case 2 & 3)"
echo "=================================================="

export PATH="$PATH:$(pwd)/build/src/esbmc"

while true; do
    # NUL separa campos sem reinterpretar espaços, pipes, glob ou metacaracteres.
    # verify_argv e um array JSON: string de shell editavel pelo agente nao e
    # aceita, pois exigiria eval e permitiria executar comandos arbitrarios.
    TASK_FIELDS_FILE=$(mktemp /tmp/ralph_task_fields.XXXXXX)
    python3 - "$PRD_FILE" > "$TASK_FIELDS_FILE" <<'PY'
import json
import os
import re
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    prd = json.load(f)

for i, task in enumerate(prd["tasks"]):
    if task.get("status") != "pending":
        continue
    task_id = task["id"]
    target = os.path.realpath(os.path.abspath(task["file_to_edit"]))
    argv = task.get("verify_argv")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
        raise SystemExit(f"id de tarefa invalido: {task_id!r}")
    if os.path.commonpath((target, "/tmp")) != "/tmp":
        raise SystemExit(f"file_to_edit fora de /tmp: {target!r}")
    if not isinstance(argv, list) or not argv or not all(
            isinstance(arg, str) and "\0" not in arg for arg in argv):
        raise SystemExit("verify_argv deve ser um array JSON nao vazio de strings")
    if argv[0] != "esbmc":
        raise SystemExit("verify_argv deve invocar diretamente o executavel esbmc")
    if len(argv) < 2:
        raise SystemExit("verify_argv deve incluir file_to_edit como primeiro argumento")
    verify_target = os.path.realpath(os.path.abspath(argv[1]))
    if verify_target != target:
        raise SystemExit(
            f"verify_argv verifica {verify_target!r}, mas file_to_edit e {target!r}"
        )
    argv[1] = target
    for value in (str(i), task_id, target, *argv):
        sys.stdout.write(value + "\0")
    break
else:
    sys.stdout.write("DONE\0")
PY

    # Bash 3 nao possui os leitores de array das versoes novas. Esta leitura
    # via arquivo preserva NUL,
    # mantem cada argumento literal e permite propagar falhas do parser acima.
    TASK_FIELDS=()
    while IFS= read -r -d '' TASK_FIELD; do
        TASK_FIELDS[${#TASK_FIELDS[@]}]="$TASK_FIELD"
    done < "$TASK_FIELDS_FILE"
    rm -f -- "$TASK_FIELDS_FILE"

    if [ "${TASK_FIELDS[0]:-}" = "DONE" ]; then
        echo "All tasks completed! Ralph Demo is finished."
        break
    fi
    if [ "${#TASK_FIELDS[@]}" -lt 4 ]; then
        echo "Failed to read a valid task from $PRD_FILE" >&2
        exit 1
    fi

    TASK_INDEX=${TASK_FIELDS[0]}
    TASK_ID=${TASK_FIELDS[1]}
    TASK_FILE=${TASK_FIELDS[2]}
    VERIFY_ARGV=("${TASK_FIELDS[@]:3}")

    echo ""
    echo "--------------------------------------------------"
    echo "Targeting Task: [$TASK_ID] -> $TASK_FILE"
    echo "--------------------------------------------------"

    ITERATION=1
    SUCCESS=false

    while [ "$ITERATION" -le "$MAX_ITERATIONS" ]; do
        echo ">> Iteration $ITERATION"
        echo "   [AI] Interacting with Agent LLM to apply fixes..."
        python3 pibic/mock_llm.py "$TASK_ID" "$ITERATION" "$TASK_FILE"
        sleep 1

        echo "   [Verifier] Checking Math/Memory Properties via ESBMC..."
        VERIFY_OUT="/tmp/ralph_verify_out_${TASK_ID}.txt"

        set +e
        "${VERIFY_ARGV[@]}" > "$VERIFY_OUT" 2>&1
        VERIFY_CODE=$?
        set -e

        if [ "$VERIFY_CODE" -eq 0 ] \
                && grep -q "VERIFICATION SUCCESSFUL" "$VERIFY_OUT" \
                && ! grep -q "VERIFICATION FAILED" "$VERIFY_OUT"; then
            echo "   [Verifier - OK] Success! Property holds, no flaws found."
            SUCCESS=true
            break
        elif [ "$VERIFY_CODE" -eq 1 ] \
                && grep -q "VERIFICATION FAILED" "$VERIFY_OUT" \
                && ! grep -q "VERIFICATION SUCCESSFUL" "$VERIFY_OUT"; then
            echo "   [Verifier - FAIL] Mathematical invariant broken or Counter-example detected!"
            echo "   [Agent Feedback] Feeding the traceback logs back into the LLM context..."
        else
            echo "   [Verifier - ERROR] ESBMC did not emit a verdict (exit=$VERIFY_CODE)." >&2
            tail -n 20 "$VERIFY_OUT" >&2
            exit 1
        fi

        ITERATION=$((ITERATION + 1))
    done

    if [ "$SUCCESS" = true ]; then
        echo "Task $TASK_ID completed successfully."
        python3 - "$PRD_FILE" "$TASK_INDEX" <<'PY'
import json
import sys

path, index = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    prd = json.load(f)
prd["tasks"][index]["status"] = "completed"
with open(path, "w", encoding="utf-8") as f:
    json.dump(prd, f, indent=4)
PY
    else
        echo "Task $TASK_ID failed after $MAX_ITERATIONS iterations."
        exit 1
    fi
done
