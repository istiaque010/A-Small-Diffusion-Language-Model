| Model | Parameters | Setting | Latency, 256-token window (s) | Tokens/s | Speedup vs AR teacher | RAM added (MB) | Reliable |
|---|---|---|---|---|---|---|---|
| mdlm_small | 5,322,496 | S=4 | 0.096 | 2619 | 16.24x | 58 | yes |
| mdlm_small | 5,322,496 | S=8 | 0.191 | 1317 | 8.17x | 58 | yes |
| mdlm_small | 5,322,496 | S=16 | 0.364 | 689 | 4.27x | 58 | yes |
| mdlm_small | 5,322,496 | S=32 | 0.723 | 347 | 2.15x | 58 | yes |
| mdlm_base | 13,891,968 | S=4 | 0.121 | 2072 | 12.85x | 127 | yes |
| mdlm_base | 13,891,968 | S=8 | 0.228 | 1100 | 6.82x | 127 | yes |
| mdlm_base | 13,891,968 | S=16 | 0.432 | 581 | 3.60x | 127 | yes |
| mdlm_base | 13,891,968 | S=32 | 0.843 | 298 | 1.85x | 127 | yes |
| ar_small | 5,322,240 | full window | 0.850 | 295 | 1.83x | 100 | yes |
| ar_teacher | 13,891,584 | full window | 1.557 | 161 | 1.00x | 132 | yes |
