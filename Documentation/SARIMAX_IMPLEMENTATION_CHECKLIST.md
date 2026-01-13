> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**  
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).  
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.  
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

Here is a checklist for verifying the implementation of the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)** model:

```markdown
# SARIMAX Implementation Checklist

This checklist is designed to ensure that the **SARIMAX** model is correctly implemented and tuned to avoid common errors such as poor fit, non-convergence, and incorrect seasonal components handling.

---

## 1. **Model Structure and Assumptions**

### 1.1 **Model Setup**
- [ ] **Verify Model Components**:
  - Ensure the SARIMAX model includes the following components:
    - **AR** (AutoRegressive) order \( p \).
    - **I** (Integrated) order \( d \) for differencing.
    - **MA** (Moving Average) order \( q \).
    - **Seasonal AR** (SAR) order \( P \).
    - **Seasonal Differencing** order \( D \).
    - **Seasonal MA** (SMA) order \( Q \).
    - **Seasonality period** \( S \) (e.g., 12 for monthly data with yearly seasonality).
    - **Exogenous variables** (X) if any.

### 1.2 **Stationarity Assumptions**
- [ ] **Check for Stationarity**:
  - Verify that the AR and MA components satisfy stationarity conditions. If not, ensure appropriate differencing is applied.
  - Use **ADF (Augmented Dickey-Fuller)** test or **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)** test to check the stationarity of the data.
- [ ] **Frequency / Seasonal Mapping**:
  - Infer the base calendar frequency (B, D, W, M, Q, etc.) using the pandas index and ensure it is attached to the series so statsmodels does not emit `ValueWarning: No frequency information was provided`.
  - Map the inferred frequency to a default seasonal period (e.g., B → 5, D → 7, W → 52, M → 12) before running the grid search so the seasonal block honours the cadence described in `Documentation/QUANT_TIME_SERIES_STACK.md`.

- [ ] **Seasonality Handling**:
  - Ensure that seasonal differences are applied correctly for seasonal data. For non-seasonal data, the seasonal components should be left as zero.

### 1.3 **Model Structure Validation**
- [ ] **AR and MA Process Order**:
  - Ensure that the AR and MA processes are correctly specified based on the autocorrelation and partial autocorrelation function (ACF/PACF) plots.
  - Check for overfitting or underfitting based on these plots and select appropriate orders \( p, d, q \).

---

## 2. **Data Preprocessing**

### 2.1 **Handling Missing Data**
- [ ] **Missing Data Imputation**:
  - Ensure missing values in the time series are handled appropriately using imputation techniques (e.g., forward/backward fill, interpolation, etc.).

- [ ] **Exogenous Variables**:
  - For SARIMAX with exogenous variables (X), ensure that these variables are properly aligned with the target time series.
  - Check for missing values or misalignments in the exogenous data.

### 2.2 **Data Transformation**
- [ ] **Log Transformation (Optional)**:
  - If the data has exponential growth, apply log transformation to stabilize variance.

- [ ] **Scaling (Optional)**:
  - If needed, scale the data or exogenous variables to a standard range (e.g., [0, 1] or mean 0, variance 1).

---

## 3. **Model Fitting and Tuning**

### 3.1 **Model Fitting**
- [ ] **Fit the SARIMAX Model**:
  - Ensure that the model is fit using maximum likelihood estimation (MLE) or conditional sum of squares (CSS), depending on the solver being used.
  - For larger datasets, ensure that the fitting process handles time complexity and resource management appropriately.

### 3.2 **Hyperparameter Tuning**
- [ ] **Grid Search for Orders**:
  - Perform grid search for hyperparameter tuning across different combinations of AR, MA, Seasonal AR, Seasonal MA, and differencing orders.
  - Use **AIC** (Akaike Information Criterion), **BIC** (Bayesian Information Criterion), or cross-validation to select the best set of hyperparameters.

- [ ] **Exogenous Variables**:
  - If using exogenous variables, ensure that the most relevant variables are included, and their effect is properly modeled.

### 3.3 **Non-convergence Issues**
- [ ] **Monitor Convergence**:
  - Check if the model converges to a solution. If not, try adjusting the initial values of parameters or using a different solver.
  - Use diagnostic plots (e.g., residuals plots) to identify potential non-convergence.

- [ ] **Overfitting**:
  - Use cross-validation or out-of-sample tests to ensure the model generalizes well and is not overfitted to the training data.

---

## 4. **Model Validation**

### 4.1 **Residual Diagnostics**
- [ ] **Check Residuals**:
  - Ensure that the residuals are **white noise** (i.e., no autocorrelation, constant mean, and variance). Use:
    - **ACF/PACF plots** of residuals.
    - **Ljung-Box Test** for autocorrelation.
  
- [ ] **Normality of Residuals**:
  - Ensure residuals follow a normal distribution, particularly if forecasting intervals are needed. Use:
    - **Histogram** and **Q-Q plot** for visual checks.
    - **Shapiro-Wilk Test** for normality.

- [ ] **Homoscedasticity**:
  - Check if residuals have constant variance. Use the **Breusch-Pagan Test** or **White Test** for heteroscedasticity.

### 4.2 **Out-of-Sample Forecasting**
- [ ] **Forecasting Accuracy**:
  - Evaluate the model's out-of-sample forecasting accuracy using metrics like:
    - **MAE (Mean Absolute Error)**.
    - **RMSE (Root Mean Squared Error)**.
    - **MAPE (Mean Absolute Percentage Error)**.
    - **R²** for goodness of fit.
  
- [ ] **Compare Baselines**:
  - Compare the SARIMAX model performance against simpler models like ARIMA, Exponential Smoothing, or seasonal naive models.

---

## 5. **Forecasting and Model Deployment**

### 5.1 **Forecasting**
- [ ] **Forecasting for Future Time Steps**:
  - Ensure the model generates forecasts for future time points correctly, using the fitted parameters and incorporating any exogenous variables for future periods.

- [ ] **Uncertainty of Forecasts**:
  - Generate prediction intervals for the forecasts, especially if the model is to be used for decision-making.

### 5.2 **Model Performance Over Time**
- [ ] **Monitor Forecast Drift**:
  - Continuously monitor forecast accuracy and residuals, particularly when there are structural breaks or seasonal shifts.

---

## 6. **Model Refinement**

### 6.1 **Refinement based on Forecast Errors**
- [ ] **Error Analysis**:
  - Analyze large prediction errors and re-tune the model if necessary. Consider adding more features or using a different model if necessary.

- [ ] **Seasonal Pattern Adjustments**:
  - If seasonal patterns change over time, adjust the seasonal parameters (e.g., seasonal order \( P, D, Q \)) or use dynamic seasonal models.

---

## 7. **Implementation Tests**

### 7.1 **Unit Tests**
- [ ] **Check SARIMAX Model Fitting**:
  - Ensure that the SARIMAX model fitting works with various configurations and data lengths.
  
- [ ] **Verify Forecasting Method**:
  - Validate that the model can generate accurate forecasts for both small and large datasets.

### 7.2 **Integration Tests**
- [ ] **Full Pipeline Testing**:
  - Run tests for the entire pipeline: data preprocessing, model fitting, diagnostics, forecasting, and validation.
  
- [ ] **Cross-validation and Hyperparameter Testing**:
  - Ensure that the hyperparameters and training process are validated with cross-validation on time series data.

---

## 8. **Final Validation**

### 8.1 **Empirical Results**
- [ ] **Compare Against Other Models**:
  - Validate the SARIMAX model performance against other time series forecasting models (e.g., ARIMA, Prophet, LSTM).
  
- [ ] **Check for Seasonal Effectiveness**:
  - Validate that the SARIMAX model correctly accounts for seasonal patterns and outperforms baseline models in cases where seasonality plays a significant role.

---

This checklist should help guide the implementation, tuning, and validation of a **SARIMAX** model, ensuring accurate results and minimizing common issues such as poor fitting, non-convergence, and incorrect seasonal handling.
```
