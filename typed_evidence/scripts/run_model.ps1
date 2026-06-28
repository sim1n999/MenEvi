param(
    [Parameter(Mandatory = $true)][string]$TextModel,
    [Parameter(Mandatory = $true)][string]$VisionModel
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Project = Split-Path -Parent $Root
$Dataset = Join-Path $Project "memlens_repro/data/memlens_agent_subset/dataset_32k.json"
$Graphs = Join-Path $Project "memlens_repro/outputs/kg_memory_32k_agent/graphs"
$Images = Join-Path $Project "memlens_repro/data/memlens/release_images"
$Packets = Join-Path $Root "results/typed_packets/packets"
$Visual = Join-Path $Root "results/visual_inspection"
$Specialists = Join-Path $Root "results/typed_specialists"
$BaselinePredictions = Join-Path $Project "runtime_routing/results/runtime_specialists/predictions.json"

python (Join-Path $Root "scripts/eval_v21.py") --self-test
python -m unittest discover -s (Join-Path $Root "tests") -p "test_*.py" -v
python (Join-Path $Root "scripts/build_typed_evidence.py") --dataset $Dataset --graph-dir $Graphs --image-dir $Images --output-dir (Join-Path $Root "results/typed_packets")
python (Join-Path $Root "scripts/run_visual_inspection.py") --packet-dir $Packets --output-dir $Visual --model $VisionModel
python (Join-Path $Root "scripts/run_typed_specialists.py") --dataset $Dataset --packet-dir $Packets --visual-observation-dir (Join-Path $Visual "observations") --policy (Join-Path $Root "prompts/typed_specialist_policy.md") --output-dir $Specialists --model $TextModel
function Run-Condition([string]$Name, [string[]]$Subtypes) {
    $OutDir = Join-Path $Root "results/$Name"
    $Prediction = Join-Path $OutDir "predictions.json"
    python (Join-Path $Root "scripts/merge_with_baseline.py") --dataset $Dataset --baseline $BaselinePredictions --candidate (Join-Path $Specialists "predictions.json") --output $Prediction --require-all-targets --include-subtypes @Subtypes
    python (Join-Path $Root "scripts/compare_predictions.py") --dataset $Dataset --baseline $BaselinePredictions --candidate $Prediction --output-json (Join-Path $OutDir "comparison.json") --output-md (Join-Path $OutDir "COMPARISON.md")
}

Run-Condition "typed_arithmetic" @("arithmetic")
Run-Condition "typed_duration" @("duration_comparison")
Run-Condition "typed_visual" @("entity", "previnfo")
Run-Condition "typed_merged" @("arithmetic", "duration_comparison", "entity", "previnfo")

