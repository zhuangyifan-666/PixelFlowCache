# Cleanup Manifest

Cleanup branch: `cleanup/final-method-codebase`

Commit before cleanup: `8d36e3c3c62377824a8b44fc60a7bc797716d240`

Tracked file count before cleanup: 216

Tracked file count after cleanup: 62

## Removed

- Root review/result bundles: all tracked `*.tgz`, `*.tar.gz`, and `*.zip` artifacts.
- Historical research pack: `pixelflowcache_research_pack/`.
- Historical reproduction/profiling/ablation docs and the long historical reproduction log.
- Historical smoke, profiling, candidate-export, exploratory benchmark, and report-generation scripts.
- Old local evaluation stub under `scripts/jit_stubs/`.
- Old packages that are not part of the final method tree: `pfc/adapters`, `pfc/profiling`, `pfc/samplers`, and `pfc/utils`.
- Old configuration files that described historical experiments.
- Tests that only covered removed exploratory code.

## Retained Scripts

- `scripts/setup_third_party.sh`
- `scripts/run_jit_stage4a_generate.py`
- `scripts/run_deco_stage4a_generate.py`
- `scripts/evaluate_stage4a_fid.py`
- `scripts/prepare_stage4a_imagenet_reference.py`
- `scripts/run_stage4a_full_eval_plan.py`
- `scripts/collect_stage4a_fid_results.py`
- `scripts/plot_stage4a_full_eval.py`
- `scripts/print_stage4a_smoke_commands.sh`
- `scripts/print_stage4a_proxy_fid_commands.sh`
- `scripts/print_stage4a_full_50k_commands.sh`

## Retained Package Modules

- `pfc/cache`: cache state, fixed-interval policy, cached module wrapper, JiT wrapper, DeCo wrapper, DeCo cached sampler, and backbone presets.
- `pfc/eval`: method presets, class label scheduling, generation IO, JiT runtime helper, and DeCo runtime helper.
- `pfc/diagnostics`: tensor and frequency diagnostics used by retained runtime code.

## Retained Tests

- Cache state, fixed-interval policy, cached module, JiT layer parsing, DeCo cache selection/wrapping, method presets, label scheduling, generation IO, FID dry-run, command-plan generation, result collection filters, plotting filters, and diagnostics tests.

## Commands Run

- `git status`
- `git branch --show-current`
- `git rev-parse HEAD`
- `git ls-files > /tmp/pfc_tracked_files_before_cleanup.txt`
- `git checkout -b cleanup/final-method-codebase`
- `git rm` for tracked review bundles, historical docs, historical scripts, old packages, old configs, and old tests.
- `python -m compileall pfc scripts tests`
- `conda run -n jit pytest -q`
- `conda run -n jit python scripts/run_jit_stage4a_generate.py --method no_cache_50 --dry-run`
- `conda run -n jit python scripts/run_jit_stage4a_generate.py --method bfc_speed_t02_10 --dry-run`
- `conda run -n jit python scripts/run_deco_stage4a_generate.py --method no_cache_50 --dry-run`
- `conda run -n jit python scripts/run_deco_stage4a_generate.py --method bfc_all_candidates_t02_10 --dry-run`
- `conda run -n jit python scripts/evaluate_stage4a_fid.py --help`
- `conda run -n jit python scripts/prepare_stage4a_imagenet_reference.py --dry-run`
- `conda run -n jit python scripts/run_stage4a_full_eval_plan.py --models jit,deco --num-images 100 --out-script /tmp/pfc_launch_test.sh`

## Validation Result

- `python -m compileall pfc scripts tests`: passed.
- `conda run -n jit pytest -q`: `58 passed in 6.42s`.
- Final dry-run/help commands listed above: passed.
- No generation, FID computation, image creation, or checkpoint download was launched by this cleanup.

## Artifact Policy

Generated results were intentionally removed from the tracked tree. Logs, outputs, samples, checkpoints, datasets, local reference folders, and compressed result bundles are ignored by `.gitignore`.

`git ls-tree HEAD third_party/` contains only `third_party/JiT` and `third_party/DeCo` gitlinks. Existing dirty/untracked state inside those submodules was not staged.

The final grep check intentionally still finds historical names in `docs/CLEANUP_INVENTORY_BEFORE.md`, because that file records the pre-cleanup inventory requested before deletion. It also finds the `scripts/jit_stubs` warning in `docs/INFERENCE_AND_FID.md`; the stub itself was deleted and no code imports it.

## History Note

This cleanup does not rewrite git history. Large artifacts removed from the current tree may still exist in previous commits. To fully remove them from repository history later, use `git filter-repo` or BFG Repo-Cleaner after coordinating with collaborators.

Example history cleanup command for a separate maintenance operation:

```bash
git filter-repo --path '*.tgz' --invert-paths
```

Do not run history rewriting as part of this cleanup commit.
