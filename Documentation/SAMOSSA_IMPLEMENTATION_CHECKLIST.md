# SAMoSSA Implementation Checklist

This checklist is designed to ensure that the SAMoSSA algorithm is correctly implemented and tuned to avoid errors such as non-convergence and lack of frequency information.

---

## 1. **Non-Convergence Handling**

### 1.1 **Page Matrix Construction**
- [ ] **Correct matrix stacking**: Ensure the Page matrix is correctly constructed from individual time series.
  - Use the formula: \( Z(f, L, T, t0) \in R^{L \times T/L} \)
  - Ensure that \( T/L \) is an integer for the time series.
  
- [ ] **Singular Value Decomposition (SVD)**:
  - Ensure that the SVD is applied to the Page matrix correctly.
  - Retain the top `k̂` singular components using Hard Singular Value Thresholding (HSVT).
  
### 1.2 **Error Bound for Estimation of Non-Stationary Components**
- [ ] **Estimation error check**: The mean squared estimation error \( \text{EstErr}(N,T, n) \) should scale as \( Õ(\frac{1}{\sqrt{NT}}) \).
  - Verify the error bound for the non-stationary components \( f̂n(t) \).

### 1.3 **AR Process Estimation**
- [ ] **AR Parameter Estimation**:
  - Ensure that OLS is used correctly to estimate AR parameters.
  - The estimation error for the AR model parameters should be bounded using Theorem 4.2.
  - Ensure that the AR parameters are estimated using residuals after subtracting the deterministic components.

### 1.4 **Convergence Check**
- [ ] **Perform validation** using synthetic or known datasets to ensure that the model converges. 
  - Check that the non-stationary components and AR noise processes are properly learned and forecasted.

---

## 2. **Frequency Information Handling**

### 2.1 **Frequency Selection for SSA**
- [ ] **Correct choice of L**:
  - Ensure that \( L \) is selected such that \( 1 < L \leq \sqrt{T} \).
  - Verify that the choice of \( L \) is suitable for capturing the relevant frequencies in the data.

- [ ] **Spectral balancing**:
  - Ensure that the spectra of the stacked Page matrix \( Zf \) are balanced. Use Assumption 4.1 to verify the spectral properties.

### 2.2 **Time Series Decomposition**
- [ ] **Proper decomposition**:
  - After SVD, ensure that the non-stationary component \( f̂n(t) \) is correctly separated from the stationary AR process \( x̂n(t) \).
  - The decomposition of each time series \( y_n(t) = f_n(t) + x_n(t) \) should be accurate.

### 2.3 **Forecasting**
- [ ] **Forecasting of both components**:
  - Forecast both the deterministic component \( f̂n(t) \) and the stationary AR component \( x̂n(t) \) independently before combining them for the final forecast.
  - Verify that AR models for \( x̂n(t) \) use lag-based forecasting properly.

---

## 3. **Algorithm Implementation**

### 3.1 **Univariate Case**
- [ ] **Page matrix transformation**:
  - Ensure that the transformation of the univariate time series into the Page matrix \( Z(y1, L, T, 1) \) is correctly implemented.
  
- [ ] **SVD-based component extraction**:
  - Ensure that the top `k̂` singular components are retained by applying HSVT.
  - Verify that the chosen matrix type (**page** vs **hankel**) matches the experiment requirements; Page matrices preserve spatial slabs across the multivariate stack, while Hankel matrices provide overlapping lags for classic SSA.
  
- [ ] **AR process fitting**:
  - Verify that the AR process is correctly fit to the residual time series after subtracting the deterministic component.
  
### 3.2 **Multivariate Case**
- [ ] **Stacked Page matrix for multivariate time series**:
  - For \( N > 1 \), ensure that the Page matrices are correctly stacked to form the multivariate matrix \( Zy \).
  
- [ ] **Component extraction for each time series**:
  - Ensure that for each time series \( n \in [N] \), the non-stationary component \( f̂n(t) \) is correctly extracted.

- [ ] **AR process identification**:
  - After obtaining the residuals, ensure that AR process parameters \( \alpha_n \) are estimated using OLS and are used for forecasting.

---

## 4. **Model Error and Forecasting Performance**

### 4.1 **Estimation Error Bound for Non-Stationary Components**
- [ ] Verify that the error for the non-stationary component \( f̂n(t) \) has the expected scaling of \( Õ(\frac{1}{\sqrt{NT}}) \).

### 4.2 **AR Process Parameter Estimation Error**
- [ ] **Estimation of AR parameters**:
  - Ensure that the error for AR parameter estimation \( \| \alphâ_n - \alpha_n \|_2 \) is bounded using the theoretical results from Theorem 4.2.

### 4.3 **Forecasting Error Bound**
- [ ] **Out-of-sample forecasting error**: 
  - Check that the forecasting error for SAMoSSA is bounded as \( Õ(\frac{1}{T} + \frac{1}{\sqrt{NT}}) \) with high probability.
  
- [ ] **Performance validation**:
  - Validate the forecasting performance of SAMoSSA on real-world datasets, and ensure that the improvement over baseline methods (e.g., mSSA, ARIMA, Prophet) is consistent with theoretical predictions.

---

## 5. **Practical Implementation Considerations**

### 5.1 **Parameter Tuning**
- [ ] **Grid search for key parameters**:
  - Perform a grid search to select the best values for `k̂`, `L`, and `p` (AR process lag order).
  
- [ ] **Cross-validation**:
  - Use rolling-window cross-validation to select the best model and tune hyperparameters.

### 5.2 **Data Preprocessing**
- [ ] **Scaling and normalization**:
  - Ensure that the time series data is properly scaled and normalized to avoid distortion by large non-stationary components.
- [ ] **Frequency Preservation**:
  - Convert indices to timezone-naive `DatetimeIndex` objects and attach the inferred frequency (or an explicit fallback such as Business-day) so future index extension, Page/Hankel stacking, and residual AutoReg fits honour the cadence documented in `QUANT_TIME_SERIES_STACK.md`.

---

## 6. **Implementation Tests**

### 6.1 **Unit Tests**
- [ ] **Page matrix construction**:
  - Ensure the Page matrix is correctly constructed for both univariate and multivariate time series.
  
- [ ] **SVD and component extraction**:
  - Test that the singular value decomposition (SVD) correctly extracts the non-stationary component and the AR noise component.

### 6.2 **Integration Tests**
- [ ] **Full pipeline**:
  - Ensure the full pipeline works for both small synthetic datasets and large real-world datasets, producing correct forecasts.

---

## 7. **Final Validation**

### 7.1 **Empirical Results**
- [ ] **Performance comparison**:
  - Compare the performance of SAMoSSA against baseline models like mSSA, ARIMA, Prophet, LSTM, and DeepAR on benchmark datasets (e.g., Traffic, Electricity, Exchange).
  - Ensure the improvements in forecasting are in the range of 5% to 37% as indicated in the paper.

### 7.2 **Error Handling and Convergence Check**
- [ ] **Convergence validation**:
  - Ensure that the algorithm converges as expected, particularly for the AR parameter estimation and forecasting accuracy.

---

This checklist should help guide the implementation and testing process for the SAMoSSA algorithm, ensuring that potential issues related to non-convergence, frequency handling, and forecasting performance are addressed.
