# Cleanup Inventory Before Final-Method Refactor

- branch before cleanup: `main`
- cleanup branch: `cleanup/final-method-codebase`
- HEAD before cleanup: `8d36e3c3c62377824a8b44fc60a7bc797716d240`
- tracked file count before cleanup: 216
- full tracked file snapshot: `/tmp/pfc_tracked_files_before_cleanup.txt`

## Tracked Artifact Bundles

```text
deco_reduced_steps_35_timing_clean50k_gpu1.tgz
stage2_review_artifacts.tgz
stage2b_review_artifacts.tgz
stage2c_review_artifacts.tgz
stage2d_review_artifacts.tgz
stage3a_review_artifacts.tgz
stage3b2_review_artifacts.tgz
stage3b_review_artifacts.tgz
stage3b_review_final.tgz
stage3c_review_artifacts.tgz
stage4a_clean50k_review_artifacts.tgz
stage4a_review_artifacts.tgz
```

## Scripts Before Cleanup

```text
scripts/collect_stage3c_unified_results.py
scripts/collect_stage4a_fid_results.py
scripts/deco_stage3b2_common.py
scripts/deco_stage3b_common.py
scripts/evaluate_stage4a_fid.py
scripts/export_stage2_cache_candidates.py
scripts/inspect_deco_cache_units.py
scripts/inspect_repos.py
scripts/jit_official_debug_no_fid.py
scripts/jit_stubs/torch_fidelity.py
scripts/launch_stage4a_full_50k_deco_stats.sh
scripts/launch_stage4a_full_50k_jit_stats.sh
scripts/launch_stage4a_full_50k_stats.sh
scripts/launch_stage4a_smoke_100.sh
scripts/make_stage3a_report_tables.py
scripts/make_stage3b2_report_tables.py
scripts/make_stage3c_paper_tables.py
scripts/plot_stage1_profiles.py
scripts/plot_stage2_jit_cache.py
scripts/plot_stage2b_jit.py
scripts/plot_stage2c_jit.py
scripts/plot_stage2d_jit.py
scripts/plot_stage3a_jit.py
scripts/plot_stage3b2_deco.py
scripts/plot_stage3b_deco.py
scripts/plot_stage3c_unified.py
scripts/plot_stage4a_full_eval.py
scripts/prepare_stage4a_imagenet_reference.py
scripts/print_stage4a_full_50k_commands.sh
scripts/print_stage4a_proxy_fid_commands.sh
scripts/print_stage4a_smoke_commands.sh
scripts/profile_deco_stage1.py
scripts/profile_jit_stage1.py
scripts/run_deco_stage3b2_decomposition.py
scripts/run_deco_stage3b2_decomposition.sh
scripts/run_deco_stage3b2_seed_sweep.py
scripts/run_deco_stage3b2_seed_sweep.sh
scripts/run_deco_stage3b2_validate.py
scripts/run_deco_stage3b2_validate.sh
scripts/run_deco_stage3b_benchmark.py
scripts/run_deco_stage3b_benchmark.sh
scripts/run_deco_stage3b_cache.py
scripts/run_deco_stage3b_cache.sh
scripts/run_deco_stage3b_inspect.sh
scripts/run_deco_stage3b_reduced_steps.py
scripts/run_deco_stage3b_reduced_steps.sh
scripts/run_deco_stage3c_50step_seed_validation.py
scripts/run_deco_stage4a_generate.py
scripts/run_jit_stage2_cache.py
scripts/run_jit_stage2_cache.sh
scripts/run_jit_stage2_grid.py
scripts/run_jit_stage2_grid.sh
scripts/run_jit_stage2b_cache.py
scripts/run_jit_stage2b_cache.sh
scripts/run_jit_stage2b_sweep.py
scripts/run_jit_stage2b_sweep.sh
scripts/run_jit_stage2b_validate.py
scripts/run_jit_stage2b_validate.sh
scripts/run_jit_stage2c_probe.py
scripts/run_jit_stage2c_probe.sh
scripts/run_jit_stage2c_validate.py
scripts/run_jit_stage2c_validate.sh
scripts/run_jit_stage2c_window_ablation.py
scripts/run_jit_stage2c_window_ablation.sh
scripts/run_jit_stage2d_first_hit_delay.py
scripts/run_jit_stage2d_first_hit_delay.sh
scripts/run_jit_stage2d_seed_sweep.py
scripts/run_jit_stage2d_seed_sweep.sh
scripts/run_jit_stage2d_validate_best_windows.py
scripts/run_jit_stage2d_validate_best_windows.sh
scripts/run_jit_stage3a_backbone_benchmark.py
scripts/run_jit_stage3a_backbone_benchmark.sh
scripts/run_jit_stage3a_backbone_benchmark_32samples.py
scripts/run_jit_stage3a_backbone_benchmark_32samples.sh
scripts/run_jit_stage3a_reduced_steps.py
scripts/run_jit_stage3a_reduced_steps.sh
scripts/run_jit_stage4a_generate.py
scripts/run_official_deco_baseline.sh
scripts/run_official_jit_baseline.sh
scripts/run_profile_deco_stage1.sh
scripts/run_profile_jit_stage1.sh
scripts/run_stage0_smoke.py
scripts/run_stage4a_full_eval_plan.py
scripts/setup_third_party.sh
scripts/stage0_common.sh
scripts/summarize_stage1_profiles.py
```

## Docs Before Cleanup

```text
docs/STAGE0_REPRO.md
docs/STAGE1_PROFILING.md
docs/STAGE2B_TIMESTEP_WINDOW_AND_DIAGNOSTICS.md
docs/STAGE2C_BOUNDARY_CACHE_OBSERVATION.md
docs/STAGE2C_WINDOW_ABLATION_AND_PROBE.md
docs/STAGE2D_VALIDATION_AND_SEED_STABILITY.md
docs/STAGE2_FIXED_BLOCK_CACHE.md
docs/STAGE3A_JIT_BACKBONE_CACHE_BENCHMARK.md
docs/STAGE3B2_DECO_CACHE_DECOMPOSITION.md
docs/STAGE3B_DECO_DIRECT_VELOCITY_CACHE.md
docs/STAGE3C_BOUNDARY_FLOW_CACHE_SYNTHESIS.md
docs/STAGE4A_FULL_INFERENCE_AND_FID.md
docs/repro_log.md
```

## PFC Modules Before Cleanup

```text
pfc/__init__.py
pfc/adapters/__init__.py
pfc/adapters/base.py
pfc/adapters/deco_adapter.py
pfc/adapters/jit_adapter.py
pfc/cache/__init__.py
pfc/cache/backbone_cache_presets.py
pfc/cache/base_policy.py
pfc/cache/cache_state.py
pfc/cache/cached_module.py
pfc/cache/deco_cached_sampler.py
pfc/cache/deco_wrap.py
pfc/cache/fixed_interval_policy.py
pfc/cache/wrap.py
pfc/diagnostics/__init__.py
pfc/diagnostics/velocity_error.py
pfc/eval/__init__.py
pfc/eval/generation_io.py
pfc/eval/label_schedule.py
pfc/eval/method_presets.py
pfc/profiling/__init__.py
pfc/profiling/deco_profiled_sampler.py
pfc/profiling/feature_recorder.py
pfc/profiling/frequency.py
pfc/profiling/jsonl.py
pfc/profiling/latency.py
pfc/profiling/module_selectors.py
pfc/profiling/run_meta.py
pfc/profiling/tensor_stats.py
pfc/profiling/velocity_recorder.py
pfc/samplers/__init__.py
pfc/samplers/solver_state.py
pfc/samplers/unified_sampler.py
pfc/utils/__init__.py
pfc/utils/repo.py
pfc/utils/seeding.py
```
