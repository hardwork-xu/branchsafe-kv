# Measured cache operations

| Case | Dense ms | Eager paged ms | Shared paged ms | Speedup | Payload ratio |
|---|---:|---:|---:|---:|---:|
| short | 0.025 | 0.032 | 0.019 | 1.30x | 50.0% |
| medium | 1.120 | 1.492 | 0.194 | 5.77x | 23.5% |
| primary | 22.587 | 42.500 | 4.502 | 5.02x | 11.7% |
| long | 26.974 | 53.883 | 6.866 | 3.93x | 20.2% |
| partial_page | 20.668 | 48.522 | 5.656 | 3.65x | 12.4% |
| no_prefix | 0.350 | 0.291 | 0.288 | 1.21x | 88.9% |
| read_heavy | 20.551 | 50.011 | 6.666 | 3.08x | 11.7% |

## Management + materialization

| Case | Dense ms | Eager paged ms | Shared paged ms | Speedup | Payload ratio |
|---|---:|---:|---:|---:|---:|
| short | 0.036 | 0.049 | 0.037 | 0.97x | 50.0% |
| medium | 2.060 | 2.473 | 1.346 | 1.53x | 23.5% |
| primary | 40.748 | 71.401 | 25.481 | 1.60x | 11.7% |
| long | 41.452 | 77.955 | 30.450 | 1.36x | 20.2% |
| partial_page | 35.967 | 68.727 | 23.014 | 1.56x | 12.4% |
| no_prefix | 0.346 | 0.266 | 0.327 | 1.06x | 88.9% |
| read_heavy | 72.620 | 136.611 | 87.468 | 0.83x | 11.7% |
