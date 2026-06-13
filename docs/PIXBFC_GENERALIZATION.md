# PixBFC Generalization

PixBFC is the general BoundaryFlowCache framework for pixel-space diffusion and flow models. This refactor keeps the existing JiT and DeCo results reproducible while making the cache boundary, prediction parameterization, and schedule explicit interfaces.

## Core Abstraction

For a pixel-space model, decompose one denoiser/velocity evaluation as:

```text
f = G o B o A
```

where `A` is the input and conditioning pathway, `B` is the cacheable boundary, and `G` maps the boundary state to the solver-facing model output.

For step `i`, a boundary value is:

```text
b_i = B(A(x_i, t_i, c), t_i, c)
```

The sampler consumes velocity:

```text
v_i = Psi(o_i, x_i, t_i)
```

Refresh computes and stores `b_i`; reuse substitutes the cached boundary value for the selected modules.

## Prediction Parameterization

PixBFC keeps model output conversion separate from cache mechanics:

- x-pred: `Psi(o, x, t) = (o - x) / clamp(1 - t, eps)`
- v-pred: `Psi(o, x, t) = o`

Other parameterizations such as eps-pred or score can be added by overriding `PixelDiffusionModelAdapter.output_to_velocity`.

## Cache Schedule

The current final method uses a fixed window scheduler:

- cache interval `K`;
- active timestep window `[t_min, t_max)`;
- optional first-hit refreshes inside the active window;
- branch and solver-stage filters.

Dynamic schedulers such as the adapted TeaCache/SeaCache-style policies can plug into the same boundary units as when-to-cache policies. They are not changed by this Step 1 refactor.

## Current Instantiations

JiT:

- adapter: `JiTBoundaryAdapter`
- prediction type: x-pred
- branch mode: conditional and unconditional forwards
- default boundary: whole transformer backbone, represented as `blocks.*`

DeCo:

- adapter: `DeCoBoundaryAdapter`
- prediction type: v-pred
- branch mode: concatenated CFG batch
- default boundaries: `all_candidates` or `backbone_plus_final`
- safe modules exclude normalization, modulation, embedding, dropout, and tiny modules.

## Adding A New Pixel Model

To add PixelGen, PixelDiT, or another pixel-space model:

1. Implement `PixelDiffusionModelAdapter`.
2. Set `prediction_type` and override `output_to_velocity` if needed.
3. Implement `list_boundary_candidates(model)` with `BoundarySpec` records.
4. Implement `default_boundary_set(model, preset_name)`.
5. Implement `wrap_boundary_set(model, boundary_set, cache_state, policy)` using `CachedModule` and `RuntimeCacheState`.
6. Add method presets and a generation wrapper or integrate the adapter into an existing pipeline.
7. Validate with `--dry-run` and CPU tests before launching any generation.

## Non-Goals

This step does not discover optimal boundaries automatically, implement token cache, train calibration modules, add solver-aware caching, or add a new model. It only creates the interfaces needed to support those steps later.
