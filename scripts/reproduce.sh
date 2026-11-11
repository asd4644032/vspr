#!/usr/bin/env bash
# Full experiment matrix of the paper. Requires Ollama and ComfyUI running.
# Usage: T2I_COMPBENCH=../T2I-CompBench UNIDET_PY=/path/to/unidet/python bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
T2I_COMPBENCH=${T2I_COMPBENCH:-../T2I-CompBench}
UNIDET_PY=${UNIDET_PY:-python}
EXTERNAL=${EXTERNAL:-"sspo=data/baseline_prompts/sspo.json pag=data/baseline_prompts/pag.json"}
for T2I in sdxl cascade; do
  for LLM in llama3.1:8b deepseek-r1:14b; do
    python scripts/run_vspr.py --set t2i.backend=$T2I llm.reasoner=$LLM
    python scripts/run_vspr.py --set t2i.backend=$T2I llm.reasoner=$LLM vspr.use_vqa=false
  done
  python scripts/run_baselines.py --set t2i.backend=$T2I --methods none gpt4 beautiful magic promptist $EXTERNAL
  python scripts/evaluate.py --eval-root outputs/eval/$T2I --unidet-dir "$T2I_COMPBENCH" --unidet-python "$UNIDET_PY"
done
python scripts/make_tables.py --eval-root outputs/eval --out outputs/tables.md
