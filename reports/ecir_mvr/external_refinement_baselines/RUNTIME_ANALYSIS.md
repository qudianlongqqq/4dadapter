# External refinement runtime analysis

| Method | Wall seconds | Wall seconds/record | Mean CPU % | Peak CPU % | Process peak RAM MB | GPU % | Peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW | 10.926773 | 0.0010926773 | 35.314286 | 50.6 | 1807.1602 | 0 | 0 |
| MMFF94S | 246.02248 | 0.024602248 | 37.416211 | 54.2 | 1812.5117 | 0 | 0 |
| GFN2_XTB | 7156.249 | 0.7156249 | 32.242286 | 56.7 | 1930.9492 | 0 | 0 |
| V8_FULL_12P5K | 563.67607 | 0.056367607 | n/a | n/a | n/a | n/a | n/a |
| MATCHED_D1_12P5K | 197.37314 | 0.019737314 | n/a | n/a | n/a | n/a | n/a |

Observed xTB/V8 wall-time ratio per record: 12.7x.

The xTB run used two CPU workers and one OMP/MKL/OpenBLAS thread per worker. GPU utilization was fixed at zero for external baselines.
