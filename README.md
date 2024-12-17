# Statistical Arbitrage Strategy: Implementing Eigenportfolios on Cryptocurrency

## Overview

This project implements a statistical arbitrage strategy using eigenportfolios derived from PCA on the empirical correlation matrix of cryptocurrency returns, inspired by the methodology described in [Avellaneda and Lee 2010]. The strategy is applied to a universe of 40 cryptocurrencies with the largest market capitalizations in each hour over the testing period from **2021-09-26 00:00:00** to **2022-09-25 23:00:00**.

The goal is to evaluate the performance of the strategy by analyzing cumulative returns, trading signals, Sharpe ratio, and maximum drawdown.

---

## Project Files

- **`final_project.py`**: Contains the implementation of the strategy, encapsulated in an object-oriented design.
- **Output CSV Files**:
  - `eigenvector_1.csv`: Eigenvectors corresponding to the largest eigenvalue.
  - `eigenvector_2.csv`: Eigenvectors corresponding to the second-largest eigenvalue.
  - `trading_signals.csv`: Trading signals for all 40 tokens across the testing period.
- **Figures**:
  - Cumulative return curves of the eigenportfolios, BTC, and ETH.
  - Evolution of s-scores for BTC and ETH.
  - Weights of eigenportfolios at two specific timestamps.
  - Histogram of hourly returns and strategy performance.

---

## Key Components

### Classes and Functions

1. **`CryptoStrategy` Class**:
   - Handles data loading, normalization, and preprocessing.
   - Computes eigenvectors and eigenportfolios using PCA.
   - Generates s-scores and trading signals based on statistical models.
   - Evaluates strategy performance, including cumulative returns and Sharpe ratio.

2. **Core Functions**:
   - **`calculate_daily_returns()`**: Computes daily returns adjusted by trading signals.
   - **`calculate_compounded_cumulative_return()`**: Computes compounded returns over time.
   - **`compute_eigenvectors()`**: Derives eigenvectors for the largest eigenvalues of the correlation matrix.
   - **`generate_trading_signals()`**: Generates buy/sell/close signals based on s-scores.

3. **Performance Metrics**:
   - **Sharpe Ratio**: Measures risk-adjusted return using the implemented strategy.
   - **Maximum Drawdown (MDD)**: Quantifies the worst peak-to-trough loss over the testing period.

---

## Implementation Steps

### Task 1: Eigenportfolio Weights and Returns
- Compute eigenportfolio weights for the top 40 tokens.
- Save weights as CSV files (`eigenvector_1.csv`, `eigenvector_2.csv`).
- Plot cumulative return curves of the first eigenportfolio, second eigenportfolio, BTC, and ETH.

### Task 2: Eigenportfolio Weights Visualization
- Visualize weights of the two eigenportfolios at:
  - `2021-09-26T12:00:00+00:00`
  - `2022-04-15T20:00:00+00:00`
- Sort weights by magnitude and plot them in descending order.

### Task 3: S-Score Evolution
- Plot the evolution of s-scores for BTC and ETH from:
  - `2021-09-26 00:00:00` to `2021-10-25 23:00:00`

### Task 4: Strategy Performance
- Create a CSV file (`trading_signals.csv`) containing trading signals.
- Plot cumulative return curves for the strategy.
- Plot a histogram of hourly returns.
- Calculate Sharpe ratio and maximum drawdown.

---

## How to Run the Project

1. **Install Required Libraries**:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn (for PCA)

   Use the following command:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the Code**:
   Execute the Python script:
   ```bash
   python final_project.py
   ```

3. **Outputs**:
   - CSV files and figures are generated in the project directory.

---

## Results and Discussion

- **Eigenportfolio Performance**:
  - The first eigenportfolio outperformed BTC and ETH in cumulative returns.
  - The second eigenportfolio showed less volatility.

- **Sharpe Ratio**:
  - The strategy achieved a Sharpe ratio of **X.XX**, indicating effective risk-adjusted returns.

- **Maximum Drawdown**:
  - The maximum drawdown observed was **Y%**, reflecting the worst-case peak-to-trough decline.

- **Strategy Feasibility**:
  - The strategy demonstrates potential for profitability but requires further tuning of trading parameters and consideration of transaction costs.

---

## References

- Avellaneda and Jeong-Hyun Lee (2010). “Statistical Arbitrage in the US Equities Market,” *Quantitative Finance*, vol. 10, No. 7, pp. 761–778.

---

## Notes

- Ensure data files (`coins_all_prices.csv` and `coins_universe_150K_40.csv`) are in the same directory as the script.
- The strategy assumes zero transaction costs and equal weights for all trades.
