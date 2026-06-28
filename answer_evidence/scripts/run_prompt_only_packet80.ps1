$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Repo = Split-Path -Parent $Root
$Dataset = Join-Path $Repo "memlens_repro/data/memlens_agent_subset/dataset_32k.json"
$SubgraphDir = Join-Path $Repo "kg_retrieval/results/retrieval_budget120/retrieved_subgraphs"
$PacketRoot = Join-Path $Root "results/packets_packet80"
$OutputDir = Join-Path $Root "results/kg_answer_focused_packet80_prompt_only"

python (Join-Path $Root "scripts/build_answer_focused_packets.py") `
  --dataset $Dataset `
  --subgraph-dir $SubgraphDir `
  --output-dir $PacketRoot `
  --packet-budget 80

python (Join-Path $Root "scripts/answer_focused_kg_answering.py") `
  --dataset $Dataset `
  --packet-dir (Join-Path $PacketRoot "packets") `
  --policy (Join-Path $Root "prompts/answer_focused_policy.md") `
  --output-dir $OutputDir `
  --prompt-only

