# GARCH-LSTM
# Hybrid GARCH-LSTM Model for Exchange Rate Volatility Forecasting

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

APARCH-LSTM showed the best predictive accuracy, with over 95,5%.

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- ARCH
- Statsmodels
- Scikit-learn
- Matplotlib

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 💡 Author

Developed by the Economist and Data Scientist *Vitor Piagetti Aimi* as part of a research project on exchange rate volatility modeling using hybrid deep learning and econometric models.
