$ErrorActionPreference = "Stop"

$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ExperimentRoot

$Dataset = Join-Path $ProjectRoot "memlens_repro/data/memlens_agent_subset/dataset_32k.json"
$SubgraphDir = Join-Path $ProjectRoot "kg_retrieval/results/retrieval_budget120/retrieved_subgraphs"
$PacketOut = Join-Path $ExperimentRoot "results/visual_ocr_packets_packet80"
$PromptOut = Join-Path $ExperimentRoot "results/kg_visual_ocr_packet80_prompt_only"
$Policy = Join-Path $ExperimentRoot "prompts/type_aware_visual_ocr_policy.md"

python (Join-Path $ExperimentRoot "scripts/build_visual_ocr_packets.py") `
  --dataset $Dataset `
  --subgraph-dir $SubgraphDir `
  --output-dir $PacketOut `
  --packet-budget 80

python (Join-Path $ExperimentRoot "scripts/type_aware_packet_answering.py") `
  --dataset $Dataset `
  --packet-dir (Join-Path $PacketOut "packets") `
  --policy $Policy `
  --output-dir $PromptOut `
  --prompt-only

Write-Host "Prompt-only outputs written to $PromptOut"
