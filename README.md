# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emmanuel-agbede/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emmanuel-agbede/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase, one week after launch. Exophase refreshes its counts in steps, hours apart, so the recorded page is a staircase: the program compresses each case's read history to the distinct page states it shows, weights each state by how long it persisted, and fits a weighted least-squares line of logit(rate) against ln(1 + hours since release) over the last 24 hours, read off at the target. The spread combines the random-walk scale with the fit's residual scale. If a case's history is a single unchanging page (Exophase has not refreshed once, as on launch night), no trend can be fitted, so the program instead projects the coming refresh: the tracked pool multiplies while earners grow far more slowly, diluting every rate except a trophy held by every tracked player. Written in Python 3.13, standard library only, deterministic; on simulated launch-week histories it beats both house baselines (persistence and the fitted curve), and on the launch-night practice reconstruction it scores roughly 0.25–0.45 against the persistence baseline's 1.40–1.69.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=eg1d3bmd&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
