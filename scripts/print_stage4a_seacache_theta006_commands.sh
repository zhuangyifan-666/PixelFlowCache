#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
# SeaCache-style adapted baseline, theta/delta = 0.06.
# These commands are printed for manual execution only.
# Run one generation command at a time unless you intentionally assign different
# physical GPUs. Do not paste the whole block as a parallel launch script.

cd /mnt/iset/nfs-main/private/zhuangyifan/PixelFlowCache
mkdir -p logs/stage4a/manual_launch

# Optional 1000-image smoke commands.
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 1000 --batch-size 8 --seed 0 --run-id stage4a_jit_seacache_theta0p06_n1000_seed0 --output-root outputs/stage4a/full_generation --save-png --no-save-npz --resume
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 1000 --batch-size 4 --seed 0 --run-id stage4a_deco_seacache_theta0p06_n1000_seed0 --output-root outputs/stage4a/full_generation --save-png --no-save-npz --resume

# JiT SeaCache-style theta=0.06, 50k generation.
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/run_jit_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 50000 --batch-size 8 --seed 0 --run-id stage4a_jit_seacache_theta0p06_n50000_seed0 --output-root outputs/stage4a/full_generation --save-png --no-save-npz --resume > logs/stage4a/manual_launch/jit_seacache_theta0p06_50k.log 2>&1

# DeCo SeaCache-style theta=0.06, 50k generation.
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n deco python scripts/run_deco_stage4a_generate.py --method seacache_style --dynamic-cache-threshold 0.06 --num-images 50000 --batch-size 4 --seed 0 --run-id stage4a_deco_seacache_theta0p06_n50000_seed0 --output-root outputs/stage4a/full_generation --save-png --no-save-npz --resume > logs/stage4a/manual_launch/deco_seacache_theta0p06_50k.log 2>&1

# JiT FID/IS after generation completes.
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/evaluate_stage4a_fid.py --fake-dir outputs/stage4a/full_generation/jit/stage4a_jit_seacache_theta0p06_n50000_seed0/seacache_style/images --fid-stats third_party/JiT/fid_stats/jit_in256_stats.npz --backend torch_fidelity --metrics fid,is --batch-size 64 --out logs/stage4a/fid/stage4a_jit_seacache_theta0p06_n50000_seed0/seacache_style/fid_results.json

# DeCo FID/IS after generation completes.
CUDA_VISIBLE_DEVICES=0 PFC_CUDA_DEVICES=0 conda run -n jit python scripts/evaluate_stage4a_fid.py --fake-dir outputs/stage4a/full_generation/deco/stage4a_deco_seacache_theta0p06_n50000_seed0/seacache_style/images --fid-stats third_party/JiT/fid_stats/jit_in256_stats.npz --backend torch_fidelity --metrics fid,is --batch-size 64 --out logs/stage4a/fid/stage4a_deco_seacache_theta0p06_n50000_seed0/seacache_style/fid_results.json

# Collect both theta=0.06 result rows after both FID JSON files are written.
conda run -n jit python scripts/collect_stage4a_fid_results.py --root outputs/stage4a/full_generation --fid-root logs/stage4a/fid --run-id stage4a_jit_seacache_theta0p06_n50000_seed0,stage4a_deco_seacache_theta0p06_n50000_seed0 --num-images 50000 --out-dir logs/stage4a/summary/seacache_theta0p06_50k

# Optional plot from the collected theta=0.06 rows.
conda run -n jit python scripts/plot_stage4a_full_eval.py --summary-dir logs/stage4a/summary/seacache_theta0p06_50k --num-images 50000
EOF
