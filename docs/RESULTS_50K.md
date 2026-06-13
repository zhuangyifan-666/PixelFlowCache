# 50k Results

The table below summarizes the clean 50k ImageNet-256 generation runs collected from existing artifacts. Logs, images, figures, and FID JSON files are not committed.

| model | method | steps | speedup vs no-cache | FID | IS | note |
|---|---|---:|---:|---:|---:|---|
| JiT | no_cache_50 | 50 | 1.000 | 4.173385 | 279.085702 | reference |
| JiT | bfc_quality_t02_08 | 50 | 1.461638 | 4.246515 | 276.804921 | quality preset |
| JiT | bfc_speed_t02_10 | 50 | 1.756686 | 4.286748 | 277.535798 | speed preset |
| JiT | reduced_steps_35 | 35 | 1.557314 | 4.677389 | 283.538294 | reduced-step baseline |
| JiT | reduced_steps_30 | 30 | 1.756623 | 5.179340 | 291.259182 | reduced-step baseline |
| DeCo | no_cache_50 | 50 | 1.000 | 2.057223 | 316.068417 | reference |
| DeCo | bfc_all_candidates_t02_10 | 50 | 1.651997 | 2.359229 | 307.574356 | main BFC speed preset |
| DeCo | bfc_backbone_plus_final_t02_10 | 50 | 1.558701 | 2.359229 | 307.574356 | conservative BFC preset |
| DeCo | reduced_steps_30 | 30 | 1.602408 | 2.671407 | 304.857249 | reduced-step baseline |
| DeCo | reduced_steps_35 | 35 | 0.938407 | 2.497287 | 321.075642 | timing anomaly; not used as main speed baseline |

The DeCo `reduced_steps_35` timing is anomalous relative to the no-cache reference and the 30-step baseline. Keep it as a reported observation, not as the primary speed comparison.

For regenerated summaries, use `scripts/collect_stage4a_fid_results.py` with both `--run-id` and `--num-images` so 100-image command checks cannot mix with 50k rows.

## SeaCache-Style Baseline

The final adapted SeaCache-style baseline threshold is `theta/delta = 0.06`. The 50k JiT and DeCo rows are pending until those runs and FID/IS evaluation are completed and collected.
