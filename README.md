# Personal Investor Assistant

A privacy-focused, local-first dashboard for tracking your investment portfolio against the market. Compare your performance (TWR/MWR) against S&P 500, Nasdaq 100, or any custom benchmark.

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.10+
- (Optional) Docker

### Option A: Local Run
1.  **Clone the repo:**
    ```bash
    git clone https://github.com/donatdermaku/personal-investor-assistant.git
    cd personal-investor-assistant
    ```

2.  **Setup & Run:**
    ```bash
    make setup
    make run
    ```
    The app will open at `http://localhost:8501`.

3.  **Load Data:**
    - On the Dashboard, choose **"🚀 Quick Start"** to see Demo Data.
    - Or upload one of the samples from `data/sample/` via the Sidebar > "Data Uploads".

### Option B: Docker
```bash
docker build -t investor-assistant .
docker run -p 8501:8501 investor-assistant
```

---

## 📊 Features
- **Privacy First**: All data stays on your machine. No external servers receive your financial data.
- **Performance Analytics**: Time-Weighted Return (TWR) and Money-Weighted Return (MWR).
- **Benchmarking**: Compare against SPY, QQQ, or custom tickers.
- **Risk Analysis**: Drawdown charts, Rolling Volatility, Sharpe Ratio, and Factor Tilts.
- **Reports**: Generate offline HTML reports and JSON summaries with Proof-of-Execution (RunManifest).

## 📂 Configuration
- `config.yml`: Core app settings (market hours, update frequency).
- `watchlist.yml`: (Optional) Pre-define a list of tickers to track if better than UI.

## 💾 Exports (Pro Mode)
Switch to **Pro** mode in the sidebar settings to unlock:
- **HTML Report**: A standalone file to share or archive.
- **Summary JSON**: Machine-readable metadata including data hashes (for audit).
- **Raw Data**: CSV dumps of daily values and cashflows.

## 🛠 Developer
- `make verify`: Run all tests and linters.
- `make clean`: Clear local caches.
- `make sample-data`: List location of sample CSVs.

Built with [Streamlit](https://streamlit.io) and [DuckDB](https://duckdb.org).
