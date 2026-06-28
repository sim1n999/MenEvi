$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Project = Split-Path -Parent $Root
$Dataset = Join-Path $Project "memlens_repro/data/memlens_agent_subset/dataset_32k.json"
$Graphs = Join-Path $Project "memlens_repro/outputs/kg_memory_32k_agent/graphs"
$Images = Join-Path $Project "memlens_repro/data/memlens/release_images"
$Packets = Join-Path $Root "results/typed_packets"
$Visual = Join-Path $Root "results/visual_prompt_audit"
$Specialists = Join-Path $Root "results/specialist_prompt_audit"

python (Join-Path $Root "scripts/eval_v21.py") --self-test
python (Join-Path $Root "scripts/build_typed_evidence.py") --dataset $Dataset --graph-dir $Graphs --image-dir $Images --output-dir $Packets
python (Join-Path $Root "scripts/run_visual_inspection.py") --packet-dir (Join-Path $Packets "packets") --output-dir $Visual --prompt-only
python (Join-Path $Root "scripts/run_typed_specialists.py") --dataset $Dataset --packet-dir (Join-Path $Packets "packets") --policy (Join-Path $Root "prompts/typed_specialist_policy.md") --output-dir $Specialists --prompt-only
python -m unittest discover -s (Join-Path $Root "tests") -p "test_*.py" -v
