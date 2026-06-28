$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Dataset = Join-Path $Root "memlens_repro\data\memlens_agent_subset\dataset_32k.json"
$Subgraphs = Join-Path $Root "kg_retrieval\results\retrieval_budget120\retrieved_subgraphs"
$CPackets = Join-Path $Root "answer_evidence\results\packets_packet80\packets"
$DPackets = Join-Path $Root "visual_evidence\results\visual_ocr_packets_packet80\packets"
$SpecialistRoot = Join-Path $Root "runtime_routing\results\specialist_packets"
$SpecialistPackets = Join-Path $SpecialistRoot "packets"
$Scripts = Join-Path $Root "runtime_routing\scripts"
$Prompts = Join-Path $Root "runtime_routing\prompts"

python (Join-Path $Scripts "build_specialist_packets.py") `
  --dataset $Dataset `
  --subgraph-dir $Subgraphs `
  --output-dir $SpecialistRoot

$Common = @(
  "--dataset", $Dataset,
  "--c-packet-dir", $CPackets,
  "--d-packet-dir", $DPackets,
  "--specialist-packet-dir", $SpecialistPackets,
  "--policy-c", (Join-Path $Root "answer_evidence\prompts\answer_focused_policy.md"),
  "--policy-d", (Join-Path $Root "visual_evidence\prompts\type_aware_visual_ocr_policy.md"),
  "--policy-duration", (Join-Path $Prompts "duration_specialist_policy.md"),
  "--policy-arithmetic", (Join-Path $Prompts "arithmetic_specialist_policy.md"),
  "--prompt-only"
)

python (Join-Path $Scripts "runtime_hybrid_answering.py") @Common `
  --variant runtime_cd `
  --output-dir (Join-Path $Root "runtime_routing\results\runtime_cd_prompt_only")

python (Join-Path $Scripts "runtime_hybrid_answering.py") @Common `
  --variant runtime_specialists `
  --output-dir (Join-Path $Root "runtime_routing\results\runtime_specialists_prompt_only")

python (Join-Path $Scripts "runtime_hybrid_answering.py") @Common --variant runtime_specialists_override --output-dir (Join-Path $Root "runtime_routing\results\runtime_specialists_override_prompt_only")

Write-Host "Runtime-routing evaluation prompt-only routing checks completed."

