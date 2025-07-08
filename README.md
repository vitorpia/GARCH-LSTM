# Hybrid GARCH-LSTM Model for Exchange Rate Volatility Forecasting

This repository presents a hybrid modeling framework that integrates traditional econometric volatility models (from the GARCH family) with deep learning techniques, specifically Recurrent Neural Networks (RNN) via Long Short-Term Memory (LSTM) architectures, to forecast the realized volatility (RV) of the USD/BRL exchange rate. Recurrent Neural Networks are particularly suited for time series forecasting because they retain temporal memory, allowing the model to capture sequential patterns and long-term dependencies in financial data. The LSTM architecture further enhances this ability by mitigating issues of vanishing gradients, making it ideal for modeling volatility dynamics that evolve over time. The Brazilian foreign exchange market is a compelling case study due to its historical susceptibility to external shocks, macroeconomic instability, and structural volatility — characteristics common to many emerging markets. By applying this hybrid GARCH-LSTM framework to Brazil, we not only evaluate the predictive performance of volatility estimators but also demonstrate the value of combining statistical rigor with the adaptive learning power of deep neural networks in a context of high financial uncertainty.
This repository contains a hybrid modeling framework that combines traditional econometric GARCH-family models with deep learning (LSTM) to forecast the realized volatility (RV) of the USD/BRL exchange rate.

## 📈 Objective

To assess the predictive performance of four GARCH variants (GARCH, EGARCH, GJR-GARCH, and APARCH) when their conditional volatility estimates are used as inputs in a univariate LSTM neural network to forecast realized volatility.

## 🔧 Methodology

1. **Data Source**: Daily USD/BRL exchange rate from Yahoo Finance (2010–2025).
2. **Preprocessing**:
   - Compute log returns.
   - Estimate realized volatility (RV) using a 21-day rolling standard deviation.
3. **Volatility Modeling**:
   - Fit GARCH, EGARCH, GJR-GARCH, and APARCH models to the return series.
   - Annualize conditional volatility estimates.
4. **LSTM Training**:
   - Use GARCH-based volatilities as inputs to LSTM networks.
   - Train each model on sequences of 30 days to predict the next-day RV.
5. **Evaluation**:
   - Compare models using RMSE and R².
   - Residual diagnostics and error analysis.

## 📊 Results Summary

| Model        | RMSE     | R²      | GARCH Time (s) |
|--------------|----------|---------|----------------|
| GARCH-LSTM   | 0.01396  | 0.9530  | 0.026          |
| EGARCH-LSTM  | 0.01538  | 0.9429  | 0.025          |
| GJR-LSTM     | 0.01384  | 0.9537  | 0.030          |
| APARCH-LSTM  | 0.01360  | 0.9553  | 0.151          |

APARCH-LSTM showed the best predictive accuracy, with over 95,5%. The superior performance of the APARCH-LSTM model may be attributed to the greater flexibility of the APARCH (Asymmetric Power ARCH) specification in capturing both asymmetries and non-linearities in the conditional volatility process. Unlike traditional GARCH or EGARCH models, APARCH introduces a power term and allows for asymmetry in response to positive and negative shocks, better reflecting the volatility behavior commonly observed in emerging markets like Brazil. This enhanced modeling of volatility dynamics likely provided more informative inputs for the LSTM network, enabling more accurate forecasts of realized volatility. The results suggest that combining a richer volatility structure with deep learning can improve predictive performance in high-noise financial environments.

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- ARCH
- Statsmodels
- Scikit-learn
- Matplotlib

---

## 💡 Author

Developed by the Economist and Data Scientist *Vitor Piagetti Aimi* as part of a research project on exchange rate volatility modeling using hybrid deep learning and econometric models.
