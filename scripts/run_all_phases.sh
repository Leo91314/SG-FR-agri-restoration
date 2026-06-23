#!/usr/bin/env bash
# Master runner for CEA experiment phases 2-8 + report (phase 9 already done).
# Each phase logs to results/cea_<phase>.log and records its exit code; a failure
# in one phase does not abort the rest.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=python3.12
mkdir -p results/cea

run() {  # name  command...
  local name="$1"; shift
  echo "=== START ${name} $(date '+%H:%M:%S') ==="
  "$@" > "results/cea_${name}.log" 2>&1
  echo "=== END   ${name} exit=$? $(date '+%H:%M:%S') ==="
}

# Phase 2: main result (two pretrained-backbone segmenters, 3 seeds, all regimes)
run phase2 $PY scripts/cea_exp.py --tag phase2_main \
  --dataset weedsgalore --plan structure,composite,veil \
  --methods bicubic,uniform_sharp,semantic_inr \
  --archs segformer_b0_imagenet,deeplabv3plus_imagenet --seeds 71,72,73

# Phase 3: fair same-protocol baselines (learned generic restorer + no-semantic INR)
run phase3 $PY scripts/cea_exp.py --tag phase3_baselines \
  --dataset weedsgalore --plan structure,composite \
  --methods bicubic,uniform_sharp,tiny_rescnn,inr_no_semantic,semantic_inr \
  --archs segformer_b0_imagenet --seeds 71,72

# Phase 4: ablations (structure/texture/semantic/frequency)
run phase4 $PY scripts/cea_exp.py --tag phase4_ablation \
  --dataset weedsgalore --plan structure,composite \
  --methods bicubic,inr_no_semantic,inr_no_texture,inr_no_freq_loss,semantic_inr \
  --archs segformer_b0_imagenet --seeds 71,72

# Phase 5: LoveDA cross-dataset (rural + urban)
run phase5_rural $PY scripts/cea_exp.py --tag phase5_loveda_rural \
  --dataset loveda_rural --plan structure,composite \
  --methods bicubic,uniform_sharp,semantic_inr \
  --archs segformer_b0_imagenet --seeds 71,72
run phase5_urban $PY scripts/cea_exp.py --tag phase5_loveda_urban \
  --dataset loveda_urban --plan structure,composite \
  --methods bicubic,uniform_sharp,semantic_inr \
  --archs segformer_b0_imagenet --seeds 71,72

# Phase 6: semantic-source robustness
run phase6 $PY scripts/cea_semantic_source.py

# Phase 7: blind generalization
run phase7 $PY scripts/cea_blind_generalization.py

# Phase 8: interpretability + frequency decoupling
run phase8 $PY scripts/cea_interpret.py

# Phase 10: assemble consolidated report
run phase10 $PY scripts/cea_report.py

echo "ALL_PHASES_DONE $(date '+%H:%M:%S')"
