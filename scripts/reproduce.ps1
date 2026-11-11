# Full experiment matrix of the paper (Windows PowerShell). Requires Ollama and ComfyUI running.
# Usage: .\scripts\reproduce.ps1 -T2ICompBench ..\T2I-CompBench -UnidetPython C:\envs\unidet\python.exe
param(
  [string]$T2ICompBench = "..\T2I-CompBench",
  [string]$UnidetPython = "python",
  # baseline prompts of the paper (shipped); pass e.g. -External gpt4,beautiful,magic,promptist to regenerate them
  [string[]]$External = @("gpt4=data/baseline_prompts/gpt4o.json", "beautiful=data/baseline_prompts/beautiful.json", "magic=data/baseline_prompts/magic.json", "promptist=data/baseline_prompts/promptist.json", "sspo=data/baseline_prompts/sspo.json", "pag=data/baseline_prompts/pag.json")
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
foreach ($T2I in @("sdxl", "cascade")) {
  foreach ($LLM in @("llama3.1:8b", "deepseek-r1:14b")) {
    python scripts/run_vspr.py --set "t2i.backend=$T2I" "llm.reasoner=$LLM"
    python scripts/run_vspr.py --set "t2i.backend=$T2I" "llm.reasoner=$LLM" "vspr.use_vqa=false"
  }
  python scripts/run_baselines.py --set "t2i.backend=$T2I" --methods none @External
  python scripts/evaluate.py --eval-root "outputs/eval/$T2I" --unidet-dir $T2ICompBench --unidet-python $UnidetPython
}
python scripts/make_tables.py --eval-root outputs/eval --out outputs/tables.md
