# Adaptive TFS-DCMMOE fusion

`tfs_dcmmmoe_fusion_model` computes the directed TFS pair representations and
uses an input-dependent gate to weight the available pairs. Pair masks are
respected, and the normalized weights are returned in `ModelOutput.aux` as
`pair_weights` for inspection.
