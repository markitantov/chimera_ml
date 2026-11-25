# Research-aligned XAI

CHIMERA-ML exposes `explain_batch` and the registered `xai_step` for the
Stage 2 leave-one-modality-out method. For a selected scalar score `s`,
`Delta_m(x) = s(f(x)) - s(f(x^(-m)))`. Positive values mean that excluding
the modality lowered the score; negative values mean that exclusion raised it.
These are local, model-specific sensitivity measurements, not SHAP values,
causal effects, calibrated confidence, or reliability scores.

The default `mask` policy preserves modality keys and disables the effective
presence/temporal mask. `[B,D]` and `[B,T,D]` inputs are supported, and padded
timesteps remain invalid. Classification requires an explicit class index (or
fixed originally predicted class); regression selects `output_dim`. Temporal
outputs use an explicit `mean`, `sum`, or `first` reduction.

Models return diagnostics independently of deltas: TFS exposes deterministic
directed `pair_*`, `pair_weights`, and `group_weights`; TFS-DCMMOE exposes its
adaptive pair weights; RAAV exposes frame/context masks and attention tensors;
TOMAR exposes `anchor`, `rho_*`, and `delta_*`. TOMAR anchor exclusion is
unsupported. Any exclusion that leaves a model without valid information is
reported as unsupported with a reason.

```yaml
steps:
  - id: predictions
    name: my_model_inference_step
  - id: xai
    name: xai_step
    after: predictions
```

The model-owning step should call `explain_batch` and store its detached result
under `explanation`; `xai_step` validates it and publishes `explanations`, without
overwriting `predictions`. A result contains `schema_version`, score and
ablation policies, per-sample baseline scores, signed/absolute deltas,
diagnostics, and unsupported reasons.

Held-out quality changes are computed separately with
`evaluate_modality_quality`. They compare actual baseline and ablated
predictions using a supplied metric and explicit metric direction. Researchers
should compare internal coefficients, local deltas, and aggregate quality
changes as complementary measurements; the library does not combine them into an
arbitrary trust score or call an external language model.

Plugin authors must own the model and batch, run the engine while the model is
available, detach and serialize the explanation, and make `xai_step` depend on
that producing artifact in sequential or parallel DAG configurations.
