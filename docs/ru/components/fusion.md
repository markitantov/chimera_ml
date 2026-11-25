# Сравнение fusion architectures

Это четыре built-in registered models. Их `aux` — architecture diagnostics, а не XAI influence measurements.

| Architecture | Required inputs | Temporal behavior | Missing data | Основные `aux` |
| --- | --- | --- | --- | --- |
| TFS | Минимум два configured tensors | `[B,D]`/`[B,T,D]`, pooled в sample output | Missing tensors пропускаются; fully masked pairs zero | Pair values/validity, `pair_weights`, `group_weights`, `fused` |
| TFS-DCMMOE | Как TFS | Та же hierarchy и shapes | Masked pair experts получают zero adaptive weight | TFS diagnostics и learned weights |
| RAAV | Non-empty `frame_modalities`, отдельный context | Frame streams имеют общий `T`; context `T` может отличаться; output per frame | Missing frame skip, missing context даёт zero context contribution | frame/context weights, masks, attention, refined |
| TOMAR | Anchor valid для каждого sample | Temporal inputs pooled; sample output | Missing non-anchor не даёт residual; missing anchor error | `anchor`, `rho_<modality>`, `delta_<modality>`, `fused` |

TFS использует directed pair hierarchy: для трёх names/default groups — шесть Transformer cross-attention blocks, reciprocal second level, masked mean/std и dense/batchnorm/tanh. TFS-DCMMOE добавляет input-dependent gate; `groups` — optional nested list directed pairs, invalid/repeated/empty groups дают `ValueError`.

TFS и DCMMOE имеют `input_dims`, `shared_dim`, `output_dim`, optional `groups`; DCMMOE также `gate_hidden=64`. Они возвращают `[B, output_dim]`; если доступно меньше двух configured tensors — error, при fully masked positions `fused` и weights zero, но head всё ещё возвращает tensor.

RAAV применяет frame gates, `num_latents=4` context bottleneck, frame-to-context attention и optional `temporal`. Параметры: `input_dims`, `shared_dim`, `output_dim`, `frame_modalities`, `context_modality`, `num_latents=4`, `gate_hidden=64`, `temporal=None`; temporal должен сохранять `[B,T,shared_dim]`.

TOMAR усредняет required anchor, строит gated residual для каждой другой modality, нормализует сумму и возвращает `[B, output_dim]`. Параметры: `input_dims`, `shared_dim`, `anchor_modality`, `output_dim`, `gate_hidden=64`; anchor должен быть present и valid для каждого sample.

Точные Python/YAML примеры находятся на индивидуальных architecture pages; не создавайте общий fictitious fusion config.
