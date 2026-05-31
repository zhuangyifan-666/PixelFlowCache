# Stage 2C Boundary Cache Observation

This note explains a Stage 2C observation for JiT whole-block output reuse. It only applies to sequential Transformer block output caching without correction. It is not a claim about token cache, branch cache, solver-aware cache, or any future calibrated method.

## Observation

For a sequential Transformer, if block `j` returns a cached output, the fresh computation from earlier blocks in the current step is overwritten at that boundary. The downstream blocks then consume the cached block-`j` state rather than the freshly computed state.

This makes arbitrary whole-block subsets behave like boundary choices:

- Caching a suffix `[j..L]` discards fresh compute from blocks before `j` at the cached block boundary.
- Caching all blocks `[0..L]` can produce the same final stale trajectory as a cached suffix ending at `L`, while doing less fresh computation.
- Caching a middle range `[j..k]` is equivalent to choosing a stale boundary at block `j`, using stale outputs through block `k`, then recomputing the fresh suffix after `k`.

## Example

For blocks `0..11`:

- Cache suffix `6..11`: blocks `0..5` are computed fresh, but the cached output at block `6` replaces that fresh path. The final output follows the stale block-`11` trajectory.
- Cache all `0..11`: the final output follows the same stale block-`11` trajectory, with less compute than the suffix case.
- Cache middle `3..8`: blocks `0..2` are computed fresh, block `3` replaces that path with a cached state, cached outputs carry through block `8`, then blocks `9..11` are recomputed fresh from the stale block-`8` state.

## Implication

Stage 2C keeps the broad whole-block ablations for diagnosis, but the observation motivates moving away from arbitrary block-subset output caches. A later stage should prefer boundary-aware designs such as prefix/backbone-level cache policies or corrected boundary reuse, with explicit validation against same-seed drift and local error diagnostics.
