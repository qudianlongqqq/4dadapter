# MCVR Stage 2b Run A data audit

Status: `PASS`.

- train: 500 molecules / 750 records
- val: 100 molecules / 130 records
- mixture: {'real_error': 0.45, 'synthetic_error': 0.3, 'clean_identity': 0.25}
- real sources as total batch fraction: {'Cartesian_teacher_100k': 0.225, 'ETFlow_formal_upstream': 0.225}
- severity counts: {'normal': 281, 'medium': 57, 'mild': 56, 'severe': 56}
- target status train: {'minimal_validity_success': 678, 'identity_fallback': 66, 'identity_clean': 6}
- target status val: {'minimal_validity_success': 125, 'identity_fallback': 5}
- train/val molecule intersection: []
- unseen update scale: train 0.50; validation-only 0.35
- test records read: 0
- audit identity: `b698ab70d2873e4d72140009402ac850f5ba76a1f43619cd3c9183462a692e21`
