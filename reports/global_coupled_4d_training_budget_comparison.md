# Global Coupled 4D training budget comparison

Status: **NOT_DIRECTLY_COMPARABLE**

Reference confidence: `low`

| field | reference 4D | global coupled 4D | match |
|---|---|---|---|
| max_steps | 100000 | 100000 | True |
| batch_size | 4 | 4 | True |
| accumulate_grad_batches | 2 | 2 | True |
| effective_batch_size | 8 | 8 | True |
| learning_rate | 0.0008 | 0.0002 | False |
| scheduler | CosineAnnealingWarmupRestarts | none | False |
| t_min | 0.0001 | 0.0 | False |
| t_max | 0.9999 | 0.25 | False |
| seed | 42 | 42 | True |
| precision | unknown | 32-true | False |
