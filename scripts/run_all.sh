#!/usr/bin/env bash
# Run every example sequentially; tally PASS/FAIL.  Usage: ./scripts/run_all.sh [mXX]
#   Extra PYTHONPATH prefix (used to test the torch-free fallback path):
#     EXTRA_PYTHONPATH=/tmp/no_torch: ./scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="${EXTRA_PYTHONPATH:-}$PWD/packages:$PWD"
pass=0; fail=0; failed=()

for dir in examples/m*/; do
  name=$(basename "$dir")
  if [ -n "${1:-}" ] && [[ "$name" != "$1"* ]]; then continue; fi
  out=$(python3 "$dir/main.py" 2>&1)
  if [ $? -eq 0 ] && echo "$out" | grep -q "^PASS"; then
    echo "✅ $name — $(echo "$out" | grep '^PASS' | head -1)"
    pass=$((pass+1))
  else
    echo "❌ $name"
    echo "$out" | tail -15 | sed 's/^/     /'
    fail=$((fail+1)); failed+=("$name")
  fi
done

echo "----------------------------------------"
echo "RESULT: $pass passed, $fail failed"
[ ${#failed[@]} -gt 0 ] && printf 'FAILED: %s\n' "${failed[@]}"
[ $fail -eq 0 ]
