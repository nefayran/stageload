| mode | runs | ran to | peak footprint (GB) | swap rise (GB) | time (s) | of which loading (s) |
|---|---|---|---|---|---|---|
| eager | 1 | decode (stopped) | 43.57 | 7.97 | 118.4 | 0 |
| staged | 2 | end / end | 19.0 / 19.0 | 0.0 / 0.0 | 124.3 / 94.3 | 12.6 / 13.0 |

| stage | eager (GB) | staged (GB) |
|---|---|---|
| setup | 31.4 | 2.08 / 2.08 |
| encode | 31.44 | 19.0 / 19.0 |
| denoise | 32.52 | 16.2 / 18.62 |
| decode | 43.57 | 14.41 / 14.43 |
