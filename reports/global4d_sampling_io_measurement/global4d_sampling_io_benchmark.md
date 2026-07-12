# Global 4D sampling I/O benchmark

Synthetic records: 500; atoms/record: 40.

| Mode | Save every | Saves | Total s | Save s | State s | Serialized MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_rewrite | 1 | 500 | 10.198759 | 6.970746 | 3.208166 | 257.204 |
| full_rewrite | 10 | 50 | 0.955321 | 0.684863 | 0.268393 | 26.182 |
| full_rewrite | 50 | 10 | 0.224496 | 0.168605 | 0.055466 | 5.647 |
| full_rewrite | 100 | 5 | 0.110462 | 0.079148 | 0.031116 | 3.080 |
| chunk | 1 | 500 | 3.978167 | 1.512805 | 2.446277 | 1.725 |
| chunk | 10 | 50 | 0.403208 | 0.157387 | 0.243781 | 1.056 |
| chunk | 50 | 10 | 0.100450 | 0.051809 | 0.048244 | 1.028 |
| chunk | 100 | 5 | 0.060380 | 0.036097 | 0.024051 | 1.025 |

The current sampler corresponds to `full_rewrite`, save every 1 record, plus two atomic JSON state writes per record.
