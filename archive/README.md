# Archive

Legacy code and superseded outputs, kept for reference. Nothing here is used by the active pipeline.

## scripts/
| File | Was | Notes |
|---|---|---|
| `main_20sens.py` | `scripts/main 20sens.py` | Older evaluation runner with **linear power-domain** noise (`--power_noise_std`, `--ptx_noise_std`, `--noise_mode`, `--seed`), hard-coded to 20 sensors / window 80, with WCL + Trilateration. Superseded by `scripts/evaluate/eval_noise_grid.py`. **Known issue (left unchanged):** its default data path `Data/20_sensors_test_data/20_sensors/5K_0` no longer exists, so it falls back to the first `*sensor_detection_summary.csv` found under `Data/`, which may be the wrong dataset. |
| `rssi_exp20sensors.bat` | `scripts/rssi_exp20sensors.bat` | Linear-noise sweep (0.00–0.04) for `main_20sens.py`. Writes to `archive/outputs/rssi_exp_20sensors/results_rssi_0X`. Works from any folder. |

## outputs/
| Folder | Was | Notes |
|---|---|---|
| `UavPredictorClass/` | `UavPredictorClass/` | A single old `5K_4_evaluation_comparison_summary.csv`. |
| `rssi_exp_20sensors/` | `scripts/outputs/rssi_exp_20sensors/` | Results of `rssi_exp20sensors.bat`. |
| `old_naming/` | `outputs/` | Earlier noise-grid CSVs using integer noise floors in their names (`-100_samples_1`, `-110_samples_10`). Current runs write `-100.0_samples_1`. |

## docs/
| File | Was | Notes |
|---|---|---|
| `data_preprocessing_pipeline.md` | `docs/data_preprocessing_pipeline.md` | Describes an earlier preprocessing pipeline (`processed_*_normalized.csv` names); out of date. |
