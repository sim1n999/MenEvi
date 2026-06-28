$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Repo = Split-Path -Parent $Root
$Dataset = Join-Path $Repo "memlens_repro/data/memlens_agent_subset/dataset_32k.json"
$Out = Join-Path $Root "results/rescore_existing_v2"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$Methods = @(
  @{ Name = "kg_memory_32k_agent"; Dir = Join-Path $Repo "memlens_repro/outputs/kg_memory_32k_agent" },
  @{ Name = "qwen_vl_caption_rag_32k_agent"; Dir = Join-Path $Repo "memlens_repro/outputs/qwen_vl_caption_rag_32k_agent" },
  @{ Name = "oracle_evidence_qwen25vl_32k_agent"; Dir = Join-Path $Repo "memlens_repro/outputs/oracle_evidence_qwen25vl_32k_agent" },
  @{ Name = "kg_soft_refusal_budget120"; Dir = Join-Path $Repo "kg_retrieval/results/kg_soft_refusal_budget120" }
)

$RescoredDirs = @()
foreach ($Method in $Methods) {
  $MethodOut = Join-Path $Out ("rescored_predictions/" + $Method.Name)
  New-Item -ItemType Directory -Force -Path $MethodOut | Out-Null
  python (Join-Path $Root "scripts/eval_v2.py") `
    --dataset $Dataset `
    --result-dir $Method.Dir `
    --output (Join-Path $MethodOut "predictions.json")
  $RescoredDirs += $MethodOut
}

python (Join-Path $Root "scripts/aggregate_eval_v2.py") `
  --output-dir $Out `
  --result-dirs $RescoredDirs

