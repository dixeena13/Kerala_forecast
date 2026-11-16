# Rainfall proposal verification

SIMULATED paired rainfall | seed 23

| Samples | Threshold (mm/24h) | Hits | Misses | False alarms | Valid pairs | POD | FAR | CSI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| station | 10.0 | 97 | 21 | 38 | 348 | 0.822 | 0.281 | 0.622 |
| station | 20.0 | 45 | 14 | 12 | 348 | 0.763 | 0.211 | 0.634 |
| station | 40.0 | 9 | 7 | 2 | 348 | 0.562 | 0.182 | 0.500 |
| grid | 10.0 | 1378 | 388 | 674 | 5547 | 0.780 | 0.328 | 0.565 |
| grid | 20.0 | 671 | 237 | 270 | 5547 | 0.739 | 0.287 | 0.570 |
| grid | 40.0 | 158 | 83 | 34 | 5547 | 0.656 | 0.177 | 0.575 |

Counts are pooled before division. NaN means the denominator is zero.
These are exact daily sample comparisons, not warning-episode or polygon verification.
Illustrative thresholds are not official warning criteria. Simulated scores are not real-world skill.
See docs/warning_verification.md for formulas, input requirements and limitations.
