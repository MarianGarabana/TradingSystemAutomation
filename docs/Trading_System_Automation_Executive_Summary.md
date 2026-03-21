# Trading System Automation – Executive Summary

**Master in Big Data and Data Analytics (MBDS), March 2026**

**Team members:** Omar Ajlouni, Marian Garabana, Marco Ortiz, Lorenz Rösgen, Jorge Vildoso, Yaxin Wu

- **Live app:** https://pythongroupassignment.streamlit.app/
- **Repository:** https://github.com/MarianGarabana/TradingSystemAutomation

---

## 1. The Solution

We built an end-to-end automated trading-signal system in Python. We used historical data to train the model, which then learned to predict price movements of over 30 different securities. There is no need for the user to open any terminal. They access a Streamlit app, pick a stock-ticker, and it delivers the trading signal alongside additional information in charts.

The project operates in two phases. Phase one consisted of the model training (offline phase). That was done locally by feeding the model cleaned historical bulk data, along with some engineered features. In the second phase (online phase), the Streamlit app calls a REST SimFin API that provides fresh data for the already trained model, enabling daily price predictions in the form of one of the three trading signals: **Buy**, **Sell**, or **Hold**.

---

## 2. Data Sources

All the financial data comes from **SimFin** (Simple Financials). SimFin is a free-access platform that provides data on publicly traded companies. We used two methods for the data extraction:

- **Bulk download in the form of CSV files:** The files contained data over five years for publicly US-listed companies, including quarterly income statements, balance sheets, and cash-flow reports. Those files were used for the training of the model.
- **REST API from SimFin:** It accesses the same data in near-real-time with a rate limit of two requests per second. An API wrapper handles the authentication, throttling, and error recovery.

---

## 3. Approach

### Data Processing – ETL Pipeline

The ETL pipeline produces a cleaned dataset, with additional features, ready for model training, and runs only once per stock-ticker. The data pulled from SimFin showed some expected issues such as dates stored as strings, stock-split artefacts (80% gains in one day), and empty dividend fields. Following the handling of those data-quality issues, we engineered **22 different features** organised in 4 groups:

- **Price-based features:** moving averages, momentum indicators (RSI, MACD, Bollinger Bands), market capitalisation, and lagged returns
- **Normalised volatility:** daily returns divided by recent volatility
- **Fundamental ratios:** gross margin, operating margin, net margin, debt-to-equity, and operating cash-flow ratio
- **Target variables:** next day's price direction (up or down)

### Machine Learning Model

Initially we wanted to predict the exact return and not just the direction, framing this as a regression problem, which led to an inefficiency in the prediction signals. Hence, we switched to a **binary classification** of the stock's closing price being either higher or lower compared to the previous day's closing price.

Another issue encountered was that banks and payment providers report their financials in a format that makes some standard ratios meaningless. The initially intended model used all 16 features on 26 of the 31 picked stocks. Instead of dropping those tickers, we built a **second model** for the financial-sector stocks using only 11 features.

We evaluated four classifiers: Logistic Regression, Random Forest, Gradient Boosting, and LightGBM. The data of each ticker was split into an **80/20 training and testing split** after being ordered chronologically. Each ticker's data was split individually before being combined into the pooled training set, ensuring that the most recent 20% of every stock's history ends up in the test set. Additionally, balanced class weights were used to prevent the model from simply predicting "up" for every signal.

| Model | Selected Model | Accuracy | Key Metrics |
|---|---|---|---|
| Standard: 26 stocks, 16 features | Logistic Regression | 50.1% | Down-recall: 0.61 (vs. 0.35 before using balanced class weights) |
| Fallback: 5 stocks (financial sector), 11 features | Gradient Boosting | 54.7% | Strongest performance on financial sector stocks |

Though the 50.1% accuracy might not seem particularly significant, the number itself misses the point. Before rebalancing, the model scored 52% accuracy by predicting "up" almost every day, correctly predicting only 35% of down-movements. After rebalancing, that number jumped to 61%, with the model now predicting a genuine mix of movement signals.

### Signal Generation

The trading signals are based on a binary classifier (0 = Sell, 1 = Buy) and a combination of the respective model confidence. If the model's confidence for the prediction is below the threshold of **0.51**, the result is a Hold prediction (regardless of the initial binary signal). All components import the signal logic from a single central strategy module, guaranteeing that the signal the model was trained to predict is the actual signal displayed in the app, making any kind of drift structurally impossible.

### API Wrapper

We built an object-oriented wrapper around the SimFin REST API that handles authentication, enforces the rate limit, retries automatically on error, and renames the API's response columns to match the format of the CSV files. This ensures that the feature-engineering code works for both historical and live data.

### Web Application

The Streamlit WebApp has four pages:

- **Home:** Introduces the system's functionalities and lists all supported tickers with general information.
- **Go Live:** Lets the user select a ticker and displays the trading signal, interactive charts, and additional metrics after fetching data from the REST API and running it through the feature pipeline.
- **Backtesting:** Displays the models' historical signals over a user-selected date range.
- **Prediction Bet:** Lets the user pick an amount, a level of leverage, and a course direction to simulate a trade in combination with the model's own call.

### Cloud Deployment

The WebApp is deployed on the **Streamlit Community Cloud** and is publicly accessible at [pythongroupassignment.streamlit.app](https://pythongroupassignment.streamlit.app). The platform automatically reads the requirements file with every deployment and installs the respective dependencies. For security, the SimFin API key is stored in the secret dashboard and not in the repository.

---

## 4. Challenges and Solutions

**100% Hold problem**
The initial regression model, designed to predict the exact size of the next day's return, resulted in 100% Hold-signal predictions due to the regularisation levels required to prevent overfitting. Switching to binary classification (simply predicting up or down) solved the problem.

**Financial-sector stocks**
Some engineered features did not work with the way financial services companies report certain ratios in their income statements. Instead of excluding these stocks, a separate model was built for the five financial-sector stocks.

**Sklearn version mismatch on deployment**
The first cloud deployment failed because the installed machine learning library version was newer than the one used during training, causing incompatibility. This was fixed by locking the library version in the requirements file so that the cloud environment matches the local one.

**Identical features in training and serving**
If the WebApp computed a feature differently from the training script, the model would produce senseless predictions without raising an error — a particularly dangerous silent failure. A central strategy module (one function, imported in both the training script and the live prediction page) eliminated this risk.

---

## 5. Conclusions

- The main achievement of this project is not any individual component but making them all work together. The ETL pipeline, the training script, the API wrapper, and the web application each depend on the same feature definitions, data formats, and shared code. The solution only works correctly when all four layers are aligned.
- The model accuracies of 50.1% and 54.7% are modest but expected. Daily stock price movements are inherently difficult to predict, and a model trained on five years of data with publicly available features is unlikely to find a strong edge. What matters is that after six iterations, both models produce genuinely balanced signals rather than defaulting to a single prediction.
- Working on this project showed that the gap between a working prototype and a deployed application is where most real Python skills are built. Problems like dependency management, secure secret handling, and continuous debugging are part of what to expect in a real-world environment.
