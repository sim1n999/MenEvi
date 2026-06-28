$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Repo = Split-Path -Parent $Root
python "$Root\scripts\soft_refusal_kg_answering.py" `
  --input "$Repo\memlens_repro\data\memlens_agent_subset\dataset_32k.json" `
  --subgraph-dir "$Repo\memlens_repro\outputs_node_budget_sweep_20260618\budget_120\retrieved_subgraphs" `
  --output-dir "$Root\results\kg_soft_refusal_budget120_prompt_only" `
  --policy "$Root\prompts\soft_refusal_policy.md" `
  --prompt-only
