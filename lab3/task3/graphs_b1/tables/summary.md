# Task 3 summary

Source: runs (every run split over 2 CAAS compute nodes); 94 runs

## 2D grid: 4-30 nodes (on-demand)

| measurement | mean over all sizes (ms) | lowest size mean | highest size mean |
|---|---|---|---|
| Neighbour request -> response | 1.6731 | 1.3383 | 1.7620 |
| Alert -> base reply | 2.2378 | 2.1906 | 2.2652 |
| Report one-way | 0.5897 | 0.5663 | 0.6389 |
| Reply one-way | 1.6476 | 1.6024 | 1.6822 |
| Terminate one-way | 1.7383 | 1.5572 | 1.9640 |
| Base processing per alert | 0.0005 | 0.0004 | 0.0008 |

## Line: 2-31 nodes (on-demand)

| measurement | mean over all sizes (ms) | lowest size mean | highest size mean |
|---|---|---|---|
| Neighbour request -> response | 1.5822 | 1.3013 | 1.6144 |
| Alert -> base reply | 2.2424 | 2.2097 | 2.2996 |
| Report one-way | 0.5778 | 0.5401 | 0.6971 |
| Reply one-way | 1.6642 | 1.5974 | 1.6907 |
| Terminate one-way | 1.7228 | 1.5367 | 2.0074 |
| Base processing per alert | 0.0004 | 0.0003 | 0.0008 |

## On-demand vs periodic (grid)

| nodes | on-demand wait (ms) | periodic Allreduce wait avg / max (ms) | msgs/node/min on-demand | periodic |
|---|---|---|---|---|
| 4 | 1.53 | 2.58 / 6.6 | 184 | 118 |
| 6 | 1.77 | 2.59 / 6.0 | 225 | 138 |
| 8 | 2.03 | 3.33 / 7.5 | 234 | 148 |
| 9 | 2.01 | 4.78 / 9.7 | 243 | 157 |
| 10 | 2.06 | 4.84 / 10.8 | 249 | 153 |
| 12 | 2.21 | 8.48 / 27.3 | 258 | 162 |
| 14 | 2.13 | 7.98 / 19.5 | 255 | 160 |
| 15 | 2.22 | 10.99 / 37.5 | 273 | 173 |
| 16 | 2.14 | 13.47 / 43.3 | 264 | 167 |
| 18 | 2.19 | 14.62 / 35.5 | 286 | 177 |
| 20 | 2.32 | 9.46 / 23.5 | 287 | 183 |
| 21 | 2.24 | 8.28 / 19.6 | 281 | 180 |
| 24 | 2.33 | 14.10 / 37.5 | 298 | 187 |
| 25 | 2.32 | 8.75 / 19.4 | 301 | 189 |
| 30 | 2.35 | 11.84 / 61.5 | 295 | 185 |
