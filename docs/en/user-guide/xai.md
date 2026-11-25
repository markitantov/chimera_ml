# Research-aligned XAI

'xai_step' wraps an inference step that exposes an optional callback:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None

The wrapped step remains an ordinary prediction step. The wrapper installs the callback for the duration of 'run', invokes 'explain_batch' for every prediction batch, restores the previous callback in 'finally', and publishes the collected result as the 'xai' artifact. The existing 'predictions' artifact is unchanged.

    steps:
      - name: xai_step
        params:
          step:
            name: fusion_step
            params:
              batch_size: 8
          ablation_policy: zero
          modalities: [audio, image]

The nested step is built through 'INFERENCE_STEPS'. It must expose 'on_prediction'. Existing steps without XAI configuration continue to work unchanged.

The generic engine supports three explicit interventions:

- 'zero': replace one input tensor with zeros;
- 'mask': disable its canonical presence and temporal masks;
- 'remove': remove its input key.

The caller chooses the policy compatible with the model. Masks are optional. Unexpected model errors propagate.

By default, predictions are compared directly. No softmax or task semantics are inferred. Pass 'score_fn(preds)' for a task-specific score. The result contains baseline values, ablated values, delta = baseline - ablated, and absolute deltas. 'output_mask_key' can exclude invalid output positions. Diagnostics are copied only for explicitly requested 'diagnostic_keys'.

ORAGEN uses zero intervention because its model requires both input keys and does not consume modality masks:

    - name: xai_step
      params:
        step:
          name: fusion_step
          params:
            batch_size: 8
        ablation_policy: zero
        modalities: [audio, image]
    - name: aggregate_windows_step

These deltas measure local prediction sensitivity. They are not causal effects, SHAP values, calibrated confidence, or a composite trust score. Held-out quality comparison remains separate in 'evaluate_modality_quality'.
