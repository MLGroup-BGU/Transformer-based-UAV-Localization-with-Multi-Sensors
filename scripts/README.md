# Scripts & RSSI Mathematical Documentation

This directory contains execution scripts, training workflows, evaluation pipelines, and experiment suites for the **Transformer-based UAV Localization with Multi-Sensors** project.

This document provides exact mathematical formulations of all RSSI localization algorithms implemented in the repository, a deep breakdown of the linear power domain noising system, and usage guidelines for running experiments.

---

## Table of Contents
1. [Script Directory Structure & Overview](#script-directory-structure--overview)
2. [Exact Mathematical Explanation of RSSI Algorithms](#exact-mathematical-explanation-of-rssi-algorithms)
   - [Weighted Centroid Localization (WCL)](#1-weighted-centroid-localization-wcl)
   - [Log-Distance Path-Loss Multilateration / Trilateration](#2-log-distance-path-loss-multilateration--trilateration)
   - [Spatial Loss Functions & Evaluation Metrics](#3-spatial-loss-functions--evaluation-metrics)
3. [Deep Explanation of the Default Noising System](#deep-explanation-of-the-default-noising-system)
   - [Linear Power Domain vs. Logarithmic Domain Noising](#why-linear-power-domain)
   - [Step-by-Step Noise Injection Pipeline](#step-by-step-noise-injection-pipeline)
   - [Relative vs. Absolute Noise Modes](#relative-vs-absolute-noise-modes)
   - [Transmit Power ($P_{tx}$) Noise Injection](#transmit-power-ptx-noise-injection)
4. [Machine Learning & Hybrid Models Overview](#machine-learning--hybrid-models-overview)
5. [Execution & Command Line Interface](#execution--command-line-interface)

---

## Script Directory Structure & Overview

| Script / File | Description | Source Reference |
| :--- | :--- | :--- |
| `main.py` | Primary evaluation runner for baseline RSSI predictors (WCL, Trilateration) and test datasets with synthetic noise options. | [main.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/main.py) |
| `train_xgboost.py` | Trains XGBoost gradient boosting post-processing regressors to refine $(x, y, z)$ trajectory predictions. | [train_xgboost.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_xgboost.py) |
| `train_hybrid.py` | Trains the Spatiotemporal Hybrid Transformer Encoder model on multi-sensor dataset arrays. | [train_hybrid.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_hybrid.py) |
| `train_lstm.py` | Trains the LSTM sequence decoder model for temporal trajectory prediction. | [train_lstm.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_lstm.py) |
| `build_dataset_cache.py` | Pre-processes raw dataset folders into fast binary HDF5/NPY tensor caches. | [build_dataset_cache.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/build_dataset_cache.py) |
| `rssi_exp.bat` | Windows batch script to sweep noise levels from $\sigma = 0.00$ to $0.10$ in steps of $0.01$. | [rssi_exp.bat](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/rssi_exp.bat) |

---

## Exact Mathematical Explanation of RSSI Algorithms

### 1. Weighted Centroid Localization (WCL)
* **Code Reference**: [rssi_wcl_predictor.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/predictors/rssi_wcl_predictor.py#L17-L97) & [rssi_utils.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/predictors/rssi_utils.py#L8-L25)

Weighted Centroid Localization estimates the 3D UAV coordinate $\hat{\mathbf{p}}_{\text{WCL}} = (\hat{x}, \hat{y}, \hat{z}) \in \mathbb{R}^3$ as a weighted linear combination of ground sensor positions $\mathbf{s}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$ for active sensors $i = 1, \dots, N$.

#### Mathematical Derivation:
1. **Convert RSSI Power Measurements to Linear Weights**:
   Raw RSSI power $P_{rx, i}$ is measured in decibel-milliwatts ($\text{dBm}$). The linear power weight $w_i$ (proportional to power in milliwatts, $\text{mW}$) is given by:
   $$w_i = 10^{\left(\frac{P_{rx, i}}{10}\right)}$$

2. **Weighted Position Estimate**:
   The estimated UAV position is the center of mass of the sensor constellation weighted by received power:
   $$\hat{\mathbf{p}}_{\text{WCL}} = \frac{\sum_{i=1}^{N} w_i \cdot \mathbf{s}_i}{\sum_{i=1}^{N} w_i}$$

3. **Degenerate / Boundary Handling**:
   If the total received power sum is below numerical precision thresholds ($\sum_{i=1}^N w_i < 10^{-12}$) or no active sensors exceed the threshold ($P_{rx} > -120\text{ dBm}$), the position defaults to the unweighted arithmetic mean of ground sensor positions:
   $$\hat{\mathbf{p}}_{\text{fallback}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{s}_i$$

---

### 2. Log-Distance Path-Loss Multilateration / Trilateration
* **Code Reference**: [rssi_trilateration_predictor.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/predictors/rssi_trilateration_predictor.py#L92-L157)

This algorithm converts raw RSSI measurements into estimated 3D Euclidean distances $d_i$ between the UAV and ground sensors using a log-distance propagation model, followed by linear multilateration for initialization and non-linear Gauss-Newton optimization.

#### A. Radio Propagation & Distance Formula
The relationship between transmit power $P_{tx}$ ($\text{dBm}$), transmitter gain $G_{tx}$ ($\text{dBi}$), receiver gain $G_{rx, i}$ ($\text{dBi}$), carrier frequency $f$ ($\text{Hz}$), path loss exponent $n$ (default $n = 2.0$), and 3D separation distance $d_i$ ($\text{meters}$) follows the Log-Distance Path-Loss model:

$$P_{rx, i} = P_{tx} + G_{tx} + G_{rx, i} - \text{PL}(d_i)$$

Where Path Loss $\text{PL}(d_i)$ in $\text{dB}$ is:
$$\text{PL}(d_i) = 10 n \log_{10}(d_i) + 20 \log_{10}(f) + 20 \log_{10}\left(\frac{4\pi}{c}\right)$$

Substituting the speed of light $c \approx 3 \times 10^8\text{ m/s}$:
$$20 \log_{10}\left(\frac{4\pi}{3 \times 10^8}\right) \approx -147.0\text{ dB}$$

$$\text{PL}(d_i) = 10 n \log_{10}(d_i) + 20 \log_{10}(f) - 147.0$$

Setting offset $\text{offset} = 20 \log_{10}(f) - 147.0$, the received power formula becomes:
$$P_{rx, i} = P_{tx} + G_{tx} + G_{rx, i} - 10 n \log_{10}(d_i) - \text{offset}$$

Solving analytically for distance $d_i$:
$$d_i = 10^{\left( \frac{P_{tx} + G_{tx} + G_{rx, i} - P_{rx, i} - \text{offset}}{10 n} \right)}$$

*(Default baseline parameters: $P_{tx} = 20.0\text{ dBm}$, $G_{tx} = 0.0\text{ dBi}$, $f = 4.33 \times 10^8\text{ Hz}$ / $433\text{ MHz}$)*

#### B. Linear Least-Squares Multilateration Initial Guess
For $K \ge 3$ active sensors, the 3D range equations are:
$$\|\mathbf{s}_i - \mathbf{p}\|^2 = d_i^2 \implies (x_i - x)^2 + (y_i - y)^2 + (z_i - z)^2 = d_i^2, \quad i = 0, 1, \dots, K-1$$

Expanding and subtracting the reference sensor equation ($i = 0$) from equation $i$ eliminates the quadratic terms $\|\mathbf{p}\|^2$:
$$2(\mathbf{s}_i - \mathbf{s}_0)^T \mathbf{p} = d_0^2 - d_i^2 + \|\mathbf{s}_i\|^2 - \|\mathbf{s}_0\|^2 \quad \text{for } i = 1, \dots, K-1$$

Expressed as a linear system $A \mathbf{p} = \mathbf{b}$:
$$A = 2 \begin{bmatrix} (x_1 - x_0) & (y_1 - y_0) & (z_1 - z_0) \\ (x_2 - x_0) & (y_2 - y_0) & (z_2 - z_0) \\ \vdots & \vdots & \vdots \\ (x_{K-1} - x_0) & (y_{K-1} - y_0) & (z_{K-1} - z_0) \end{bmatrix} \in \mathbb{R}^{(K-1) \times 3}$$

$$\mathbf{b} = \begin{bmatrix} d_0^2 - d_1^2 + (x_1^2 + y_1^2 + z_1^2) - (x_0^2 + y_0^2 + z_0^2) \\ d_0^2 - d_2^2 + (x_2^2 + y_2^2 + z_2^2) - (x_0^2 + y_0^2 + z_0^2) \\ \vdots \\ d_0^2 - d_{K-1}^2 + (x_{K-1}^2 + y_{K-1}^2 + z_{K-1}^2) - (x_0^2 + y_0^2 + z_0^2) \end{bmatrix} \in \mathbb{R}^{K-1}$$

The linear estimate is obtained via the Moore-Penrose pseudoinverse:
$$\mathbf{p}_{\text{lstsq}} = A^\dagger \mathbf{b} = (A^T A)^{-1} A^T \mathbf{b}$$

#### C. Airborne Altitude (Z-Axis) Initialization Heuristic
Because ground sensors are situated in a nearly co-planar arrangement ($z_i \approx 0\text{ m}$), the vertical component $z_{\text{lstsq}}$ derived from linear algebra is ill-conditioned and can point underground ($z < 0$).
To ensure convergence to the true airborne solution, the initialization applies:
$$z_{\text{init}} = |z_{\text{lstsq}}|$$
$$\text{If } z_{\text{init}} < 200.0\text{ m}, \quad \text{set } z_{\text{init}} = 1000.0\text{ m}$$

If $K < 3$ or linear least squares fails, the initial guess defaults to $\mathbf{p}_{\text{init}} = \hat{\mathbf{p}}_{\text{WCL}} + (0, 0, 1000)^T$.

#### D. Non-Linear Bounded Residual Optimization
The initial estimate is refined by minimizing non-linear distance residual errors:
$$\min_{\mathbf{x} \in [0, 5000]^3} \frac{1}{2} \sum_{i=0}^{K-1} r_i(\mathbf{x})^2$$

where the residual function for sensor $i$ is:
$$r_i(\mathbf{x}) = \|\mathbf{s}_i - \mathbf{x}\|_2 - d_i$$

Solved using SciPy's `least_squares` with `method='trf'` (Trust Region Reflective) bounded within the physical workspace domain $[0\text{ m}, 5000\text{ m}]^3$.

---

### 3. Spatial Loss Functions & Evaluation Metrics
* **Code Reference**: [base_predictor.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/predictors/base_predictor.py#L38-L138)

Given ground truth coordinates $\mathbf{p}_k^* = (x_k^*, y_k^*, z_k^*)$ and predicted coordinates $\hat{\mathbf{p}}_k = (\hat{x}_k, \hat{y}_k, \hat{z}_k)$ across $M$ evaluation timestamps:

1. **3D Euclidean Distance Error (meters)**:
   $$e_{3D}^{(k)} = \|\hat{\mathbf{p}}_k - \mathbf{p}_k^*\|_2 = \sqrt{(\hat{x}_k - x_k^*)^2 + (\hat{y}_k - y_k^*)^2 + (\hat{z}_k - z_k^*)^2}$$
   $$\text{3D Mean Error} = \frac{1}{M} \sum_{k=1}^{M} e_{3D}^{(k)}, \quad \text{3D RMSE} = \sqrt{\frac{1}{M} \sum_{k=1}^{M} \left(e_{3D}^{(k)}\right)^2}$$

2. **2D Planar Error & Z-Axis Error (meters)**:
   $$e_{2D}^{(k)} = \sqrt{(\hat{x}_k - x_k^*)^2 + (\hat{y}_k - y_k^*)^2}, \quad e_{Z}^{(k)} = |\hat{z}_k - z_k^*|$$

3. **Trajectory Step Displacement Loss (meters)**:
   Measures step-to-step relative movement vector error along the trajectory path:
   $$\mathcal{L}_{\text{step}} = \frac{1}{M-1} \sum_{k=1}^{M-1} \left\| (\hat{\mathbf{p}}_{k+1} - \hat{\mathbf{p}}_k) - (\mathbf{p}_{k+1}^* - \mathbf{p}_k^*) \right\|_2$$

4. **PyTorch MSE Loss ($\text{km}^2$)**:
   Calculated in kilometers ($10^{-3}\text{ m}$) to match standard PyTorch `nn.MSELoss()` training objectives:
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{k=1}^{M} \left\| \frac{\hat{\mathbf{p}}_k - \mathbf{p}_k^*}{1000} \right\|_2^2 = \frac{1}{M} \sum_{k=1}^{M} \left[ \left(\frac{\hat{x}_k - x_k^*}{1000}\right)^2 + \left(\frac{\hat{y}_k - y_k^*}{1000}\right)^2 + \left(\frac{\hat{z}_k - z_k^*}{1000}\right)^2 \right]$$

---

## Deep Explanation of the Default Noising System

### Why Linear Power Domain?
* **Code Reference**: [main.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/main.py#L101-L165)

In physical RF systems, environmental thermal noise (AWGN) and multipath propagation fading act additively or multiplicatively on linear power signals ($\text{Watts}$ or $\text{mW}$). Applying Gaussian noise directly to logarithmic power measurements ($\text{dBm}$) produces asymmetrical, non-physical distributions in linear space.

Therefore, the project's noising pipeline converts received power measurements to the **linear power domain**, injects zero-mean Gaussian normal noise, clamps physical lower bounds, and maps the signals back to $\text{dBm}$.

```
  [Raw P_dBm] ---> Convert (10^(P/10)) ---> [P_linear (mW)]
                                                  |
                                          + Noise (Normal Dist)
                                                  |
  [Noisy P_dBm] <-- Log (10*log10) <--- Clamp (>= 1e-15 mW)
```

---

### Step-by-Step Noise Injection Pipeline

For active sensor detections ($P_{\text{dBm}} \ne 0.0$ and $P_{\text{dBm}} > -120.0\text{ dBm}$):

1. **Logarithmic to Linear Transformation**:
   $$P_{\text{linear}} = 10^{\left( \frac{P_{\text{dBm}}}{10} \right)} \quad (\text{mW})$$

2. **Gaussian Normal Noise Sampling**:
   Sample random noise vector $\mathbf{\epsilon}$ from a Gaussian distribution:
   $$\mathbf{\epsilon} \sim \mathcal{N}\left(\mu = 0.0, \, \sigma = \text{\texttt{power\_noise\_std}}\right)$$

3. **Application of Linear Noise (`noise_mode`)**:
   - **Relative Noise Mode** (Default, `--noise_mode relative`):
     Noise acts multiplicatively relative to signal power magnitude:
     $$P_{\text{linear, noisy}} = P_{\text{linear}} \cdot (1 + \mathbf{\epsilon}) = P_{\text{linear}} + P_{\text{linear}} \cdot \mathbf{\epsilon}$$
     *Physical intuition*: Represents signal-dependent proportional fluctuation (e.g. fading / fading variance scaling with signal level).

   - **Absolute Noise Mode** (`--noise_mode absolute`):
     Noise acts additively in linear power space:
     $$P_{\text{linear, noisy}} = P_{\text{linear}} + \mathbf{\epsilon}$$
     *Physical intuition*: Represents continuous additive background thermal noise floor.

4. **Physical Floor Clamping**:
   To prevent non-physical negative linear power or logarithmic domain undefined operations ($\log 0$), power levels are strictly bounded from below:
   $$P_{\text{linear, clamped}} = \max\left( P_{\text{linear, noisy}}, \, 10^{-15} \right)$$
   *(Note: $10^{-15}\text{ mW} = -120\text{ dBm}$, which aligns with receiver sensitivity limits).*

5. **Linear to Logarithmic Transformation**:
   $$P_{\text{dBm, noisy}} = 10 \log_{10}\left( P_{\text{linear, clamped}} \right)$$

---

### Transmit Power ($P_{tx}$) Noise Injection
Similarly, when transmitter power uncertainty is enabled (`--ptx_noise_std > 0.0`), noise is applied to drone transmitter power specs ($P_{tx}$):

$$P_{tx, \text{linear}} = 10^{\left( \frac{P_{tx, \text{dBm}}}{10} \right)}$$
$$\mathbf{\eta} \sim \mathcal{N}(0, \text{\texttt{ptx\_noise\_std}})$$
$$P_{tx, \text{linear, noisy}} = P_{tx, \text{linear}} \cdot (1 + \mathbf{\eta}) \quad (\text{if relative})$$
$$P_{tx, \text{dBm, noisy}} = 10 \log_{10}\left( \max(P_{tx, \text{linear, noisy}}, 10^{-15}) \right)$$

---

## Machine Learning & Hybrid Models Overview

In addition to algebraic RSSI algorithms, the codebase provides data-driven machine learning models:

1. **XGBoost Trajectory Post-Processor**:
   - Source: [train_xgboost.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_xgboost.py) & [online_post_processor.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/postprocessing/online_post_processor.py)
   - Extracted Features: Velocity trends ($\text{grad}(x), \text{grad}(y), \text{grad}(z)$), sensor array inter-distance metrics, spatial centroid offsets, and temporal context window features ($\pm 5$ steps).
   - Target: Refines $(x, y, z)$ coordinates to reduce residual spatial error.

2. **Spatiotemporal Hybrid Transformer Encoder**:
   - Source: [train_hybrid.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_hybrid.py) & [hybrid_encoder.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/models/hybrid_encoder.py)
   - Encodes static sensor parameters (gains, coordinates) and dynamic multi-sector RSSI readings into spatial and temporal attention embeddings.

3. **LSTM Sequence Decoder**:
   - Source: [train_lstm.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/scripts/train_lstm.py) & [lstm_decoder.py](file:///c:/Users/user/Documents/BGU-Master/Transformer-based-UAV-Localization-with-Multi-Sensors/src/models/lstm_decoder.py)
   - Autoregressively models temporal trajectory dynamics from encoder hidden state representations.

---

## Execution & Command Line Interface

### 1. Run Baseline RSSI Evaluation (Main Entry Point)
To run WCL and Trilateration baseline evaluation on test dataset `5K_4`:
```bash
python main.py --max_exps 10
```

### 2. Run Evaluation with Linear Power Noise
To evaluate algorithm robustness under 5% relative linear power noise ($\sigma = 0.05$):
```bash
python main.py --power_noise_std 0.05 --noise_mode relative --max_exps 10
```

To evaluate with both receiver power noise ($\sigma_{\text{power}} = 0.05$) and transmit power noise ($\sigma_{P_{tx}} = 0.02$):
```bash
python main.py --power_noise_std 0.05 --ptx_noise_std 0.02 --max_exps 10
```

### 3. Run Automated Noise Sweep Experiment Suite
Run the batch script to automatically evaluate noise levels from $0.00$ to $0.10$:
```cmd
scripts\rssi_exp.bat
```

### 4. Plot Noise Analysis Graphs
To generate evaluation metric plots across noise levels:
```bash
python -m src.utils.plot_results
```

### 5. Train Machine Learning Models
```bash
# Build dataset binary cache
python scripts/build_dataset_cache.py

# Train XGBoost Post-Processor
python scripts/train_xgboost.py

# Train Hybrid Transformer Encoder
python scripts/train_hybrid.py

# Train LSTM Decoder
python scripts/train_lstm.py
```
