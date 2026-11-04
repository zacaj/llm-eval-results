# Sampler sweep (IQ1_M coder, Strata)

updated 2026-10-15 11:35:47 local. Compared on: bowling, full-paasio, pov, scale-generator; top_k=20 top_p=0.95. Includes older runs with a matching effective sampler.

| temp | min_p | trials | pass% (±SE) | avg try (passes) | ctx-exhaust | bailed |
|---|---|---|---|---|---|---|
| 0.8 | 0.1 | 20 | 95% ±5 | 2.05 | 0 | 0 |
| 0.8 | 0.0 | 20 | 95% ±5 | 2.16 | 1 | 0 |
| 0.4 | 0.05 | 12 | 92% ±8 | 1.82 | 0 | 1 |
| 0.4 | 0.1 | 12 | 92% ±8 | 2.36 | 2 | 0 |
| 0.4 | 0.0 | 12 | 83% ±11 | 1.90 | 1 | 0 |
| 0.8 | 0.01 | 12 | 83% ±11 | 2.40 | 0 | 0 |
| 1.0 | 0.0 | 16 | 81% ±10 | 2.23 | 8 | 0 |
| 0.6 | 0.0 | 12 | 75% ±12 | 2.00 | 4 | 0 |
| 0.8 | 0.05 | 12 | 75% ±12 | 2.56 | 4 | 0 |

SE is a rough binomial estimate treating each (exercise, seed) as independent; differences under ~2 SE are within noise. Pass rates near 100% are a ceiling: avg try and ctx-exhaust discriminate better there.
