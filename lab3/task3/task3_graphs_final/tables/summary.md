# Task 3 summary

Source: runs (every run split over 2 CAAS compute nodes); 94 runs

## 2D grid: 4-30 nodes (on-demand)

| measurement | mean over all sizes (ms) | lowest size mean | highest size mean |
|---|---|---|---|
| Neighbour request -> response | 1.6799 | 1.2590 | 1.7840 |
| Alert -> base reply | 3.3368 | 3.3102 | 3.3991 |
| Report one-way | 1.6887 | 1.6497 | 1.7700 |
| Reply one-way | 1.6478 | 1.6017 | 1.6837 |
| Terminate one-way | 1.7521 | 1.4255 | 1.8489 |
| Base processing per alert | 0.0003 | 0.0003 | 0.0005 |

## Line: 2-31 nodes (on-demand)

| measurement | mean over all sizes (ms) | lowest size mean | highest size mean |
|---|---|---|---|
| Neighbour request -> response | 1.5733 | 1.2057 | 1.6227 |
| Alert -> base reply | 3.3415 | 3.3051 | 3.4044 |
| Report one-way | 1.6751 | 1.6377 | 1.8443 |
| Reply one-way | 1.6662 | 1.5094 | 1.7470 |
| Terminate one-way | 1.7532 | 1.4489 | 2.0239 |
| Base processing per alert | 0.0002 | 0.0002 | 0.0004 |

## On-demand vs periodic (grid)

| nodes | on-demand wait (ms) | periodic Allreduce wait avg / max (ms) | msgs/node/min on-demand | periodic |
|---|---|---|---|---|
| 4 | 1.38 | 2.58 / 6.6 | 179 | 118 |
| 6 | 1.74 | 2.59 / 6.0 | 222 | 138 |
| 8 | 2.03 | 3.33 / 7.5 | 242 | 148 |
| 9 | 2.05 | 4.78 / 9.7 | 242 | 157 |
| 10 | 2.07 | 4.84 / 10.8 | 243 | 153 |
| 12 | 2.10 | 8.48 / 27.3 | 260 | 162 |
| 14 | 2.02 | 7.98 / 19.5 | 250 | 160 |
| 15 | 2.33 | 10.99 / 37.5 | 277 | 173 |
| 16 | 2.17 | 13.47 / 43.3 | 264 | 167 |
| 18 | 2.29 | 14.62 / 35.5 | 280 | 177 |
| 20 | 2.39 | 9.46 / 23.5 | 292 | 183 |
| 21 | 2.32 | 8.28 / 19.6 | 285 | 180 |
| 24 | 2.40 | 14.10 / 37.5 | 295 | 187 |
| 25 | 2.39 | 8.75 / 19.4 | 308 | 189 |
| 30 | 2.33 | 11.84 / 61.5 | 290 | 185 |

## One vs two compute nodes

Neighbour round trip: 1.158 -> 1.640 ms
Alert round trip: 3.246 -> 3.339 ms
