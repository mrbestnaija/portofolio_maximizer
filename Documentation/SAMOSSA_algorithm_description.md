
# Reinforcement Learning for Time Series Forecasting with Interventions

## Project Metadata
- **Author**: Mrbestnaija
- **Framework Type**: Reinforcement Learning with Time Series Interventions
- **Mathematical Convention**:
  - **Indexing**: MIT standard (1 to N)
  - **Precision**: np.float64 throughout
  - **Random Seed**: Fixed for reproducibility
- **Performance Requirements**:
  - **Code Efficiency**: Vectorized operations with numpy/pandas, GPU-accelerated with CuPy
  - **Runtime Logging**: Use loguru for detailed execution logs
  - **Memory Optimization**: Numba for JIT compilation, CuPy for GPU operations
  - **Visualization**: Use Matplotlib, Plotly, and Seaborn for diagnostics

## Core Algorithms

### 1. **Multi-Singular Spectrum Analysis (MSSA) Framework**
#### Mathematical Foundation:
- **Decomposition Model**: Y(t) = Y_stationary + f_non-stationary(t) + ε(t)
- **Matrix Formulation**: Y = F + E where Y, F, E ∈ R^(N×T)
- **Low Rank Assumption**: rank(F) = r << min(N, T)
- **Separable Structure**: f_i(t) = u_i^T ρ(t) = Σ_{k=1}^r u_{ik} ρ_k(t)

#### Implementation Stages:
1. **SSA Decomposition**: Use Page matrix for trend/seasonal decomposition.
2. **Residual Modeling**: Model residuals with AR(p) using statsmodels, optimized with numba.
3. **Forecasting Combination**: Recombine deterministic and stochastic components using Prophet or CuPy.

### 2. **Q-Learning for Time Series Forecasting**
#### Q-Learning Update Rule:
- Formula: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- Description: Q-values are updated using the Bellman equation to optimize the policy.

#### Parameters:
- **alpha**: Learning rate
- **gamma**: Discount factor
- **epsilon**: Exploration rate

### 3. **CUSUM-based Change Point Detection**
#### Mathematical Foundation:
- Detection Score: D(t) = ||(Û₀⊥)ᵀv_t||₂ - C
- CUSUM Framework: y(t+1) = max(y(t) + D(t), 0)
- Change Point Estimation: τ = inf{t : y(t) ≥ h}

#### Implementation Steps:
1. **Input Analysis**: Compute series length T and sum S.
2. **Page Matrix Construction**: Build L×(T₀/L) matrix using block method.
3. **SVD Decomposition**: P = UΣVᵀ with 90% energy criterion.

## Optimization Framework

### 1. **Window Length Selection**:
- Formula: L ≈ min(√(NT), T)
- Error Bound: ImpErr ≈ 1/(√T√min(N, T))

### 2. **Rank Selection**:
- Use TruncatedSVD for optimal rank selection.

### 3. **Missing Data Handling**:
- Use KNNImputer from sklearn for missing data.

## Validation Protocols

### Mathematical Verification:
- **SVD Reconstruction**: Pseudo-inverse via SVD.
- **Rank Validation**: Ensure 90% energy capture.

### Performance Metrics:
- **Reconstruction Error**: ||Ŷ - Y||_F
- **Forecasting Accuracy**: MAPE, RMSE, MAE.

## Implementation Specifications

### Data Structures:
- **Page Matrix**: L×K block-based construction.
- **Tensor Format**: N×L×(T-L+1) for multi-series.

### Computational Requirements:
- **SVD Library**: Use CuPy for GPU-accelerated SVD.
- **Memory Efficiency**: Use CuPy and Numba for parallel processing.
