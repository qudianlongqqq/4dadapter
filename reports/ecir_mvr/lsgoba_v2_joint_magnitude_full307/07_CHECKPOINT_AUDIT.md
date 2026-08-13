# V2 checkpoint audit

- selection rule frozen before training: eligible if VAL prior <= initial V1 VAL prior + 0.10; then minimum VAL(post + lambda*move), tie earlier step
- initial v1-warm-start VAL prior: -2.834036547
- selected step: 12500
- selected VAL prior/post/move/selection objective: -2.858191745 / 0.629339155 / 0.198674199 / 0.710385159
- selected checkpoint SHA256: `9c7bfae6ea2bd0035d83c79e104cebb158904a1b8c4721d9f32a7aba890c8673`
- V3D/PoseBusters/xTB/ORCA used for selection: no
