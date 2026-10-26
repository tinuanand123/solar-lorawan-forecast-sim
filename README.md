# Forecast-aware energy management for solar-powered LoRaWAN nodes (simulation code)

Code and result files for the paper "Forecast-Aware Energy Management for Solar-Powered LoRaWAN Sensor Nodes: A Simulation Study of Battery Downsizing" (ICGEST 2026, AIP Conference Proceedings).

**All weather is synthetic (documented generator in `lib.py`). No field or measured data are used.**

## Files
| File | Purpose |
|---|---|
| `core.c` | C core of the hourly node/battery simulator |
| `lib.py` | Constants, weather generator, forecasters, controller wrapper |
| `run1.py` | Trains forecasters, baseline sweep over 16 capacities x 7 controllers -> `res1.pkl`, `models.pkl` |
| `run2.py` | Ablation, noise/bias injection, sensitivity, stress scenario, importance -> `res2.pkl` |
| `run3.py` | Capacity x radio-energy and panel-area grid -> `res3.pkl` |
| `figs2.py` | Regenerates all figures into `figures/` |
| `*.pkl` | Saved results used in the paper (`models.pkl` is rebuilt by `run1.py`) |

## Run
```
pip install -r requirements.txt
gcc -O2 -shared -fPIC -o core.so core.c -lm
python3 run1.py && python3 run2.py && python3 run3.py && python3 figs2.py
```
Gradient-boosting training is not bit-exact between machines, so small numerical differences (last digit) are expected.

## Licence and citation
MIT licence (see `LICENSE`). Please cite the paper when using the code.
