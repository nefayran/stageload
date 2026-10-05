| mode | runs | ran to | peak footprint (GB) | swap rise (GB) | time (s) | of which loading (s) |
|---|---|---|---|---|---|---|
| eager | 2 | texture (stopped) / texture (stopped) | 30.55 / 30.53 | 2.27 / 3.87 | 351.0 / 410.5 | 0 / 0 |
| staged | 2 | end / end | 69.57 / 70.19 | 0.0 / 2.03 | 595.9 / 631.9 | 24.4 / 23.8 |

| stage | eager (GB) | staged (GB) |
|---|---|---|
| setup | 20.41 / 20.4 | 2.93 / 2.93 |
| preprocess | – / – | – / 3.0 |
| camera | 23.18 / 23.18 | 6.72 / 6.72 |
| structure | 22.21 / 22.21 | 10.9 / 10.91 |
| shape_512 | 26.6 / 26.6 | 11.46 / 11.44 |
| shape_1024 | 30.55 / 30.53 | 13.06 / 13.06 |
| texture | – / – | 30.08 / 30.1 |
| decode | – / – | 17.87 / 18.35 |
| export | – / – | 69.57 / 70.19 |
