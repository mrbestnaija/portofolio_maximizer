````md
# Portfolio Maximizer – Autonomous Profit Engine

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase 7.9 Complete](https://img.shields.io/badge/Phase%207.9-Complete-green.svg)](Documentation/EXIT_ELIGIBILITY_AND_PROOF_MODE.md)
[![Tests: 731](https://img.shields.io/badge/tests-731%20(718%20passing)-success.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-informational.svg)](Documentation/)
[![Research Ready](https://img.shields.io/badge/research-reproducible-purple.svg)](#-research--reproducibility)

> End-to-end quantitative automation that ingests data, forecasts regimes, routes signals, and executes trades hands-free with profit as the north star.

**Version**: 4.3  
**Status**: Phase 7.9 Complete - PnL integrity enforcement, adversarial audit, OpenClaw automation  
**Last Updated**: 2026-02-17

---

## 🎯 Overview

**Portfolio Maximizer** is an automated trading system that marries institutional-grade ETL (Extract, Transform, Load) with autonomous execution. It continuously extracts financial data, validates it, preprocesses it, forecasts market trends, and executes trades, all with minimal human oversight.

### Key Features:
- **End-to-End Automation**: The system automatically handles data extraction, preprocessing, forecasting, and trade execution.
- **PnL Integrity**: The system ensures accurate tracking of trades, preventing double-counting or orphaned positions.
- **Advanced Forecasting**: It uses multiple forecasting models, including **SAMOSSA**, **SARIMAX**, and **GARCH**.
- **High Performance**: With optimized operations and caching, it processes data quickly.

---

## 🎯 Latest Enhancements (Feb 2026)

**Phase 7.9 Achievements**:
- **PnL Integrity Enforcement**: Ensures database-level constraints to avoid errors like double-counting and orphaned positions.
- **Adversarial Audit**: Revealed a 94.2% quant FAIL rate, broken confidence calibration, and ensemble underperformance.
- **OpenClaw Cron Automation**: Runs 9 audit-aligned cron jobs to keep tasks like integrity checks and trading execution up-to-date.
- **Tavily Search Integration**: Adds quota-safe web grounding with fallback to **Tavily** instead of Brave for more robust data scraping.
- **3-Model Local LLM**: deepseek-r1:8b for fast reasoning, deepseek-r1:32b for heavy reasoning, and qwen3:8b for tool orchestration.
- **Cross-session Persistence**: Portfolio state and cash state are preserved between sessions via `--resume`.

---

## 🛠️ Installation and Setup

Let's get **Portfolio Maximizer** running on your machine!

### 1. Clone the Repository

Clone the **Portfolio Maximizer** GitHub repository:

```bash
# Clone the repository to your local machine
git clone https://github.com/mrbestnaija/portfolio_maximizer.git
cd portfolio_maximizer
````

### 2. Set Up a Virtual Environment

Create a virtual environment to manage dependencies.

**For Linux/Mac**:

```bash
python3 -m venv simpleTrader_env
source simpleTrader_env/bin/activate
```

**For Windows**:

```bash
python -m venv simpleTrader_env
simpleTrader_env\Scripts\activate
```

### 3. Install Dependencies

Run this to install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory to store your API keys and cache settings. It is crucial that you do **not** share this file publicly.

Example `.env` configuration:

```bash
# API Keys (if needed)
ALPHA_VANTAGE_API_KEY=your_key_here

# Cache Settings
CACHE_VALIDITY_HOURS=24
```

---

## 🔑 Security and Configuration

### Security Essentials

**Portfolio Maximizer** relies on environment variables for sensitive information like API keys. Follow these practices to ensure the security of your data:

1. **Do not commit your `.env` file** to GitHub.
2. Use **environment variables** for storing sensitive information such as `API_KEYS` and `PASSWORDS`.

**Configuration Tip**:

* **API Keys**: Store all third-party API keys in the `.env` file.
* **Cache Settings**: Adjust the cache validity if you need more frequent updates.

Example `.env`:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
CACHE_VALIDITY_HOURS=24
```

---

## 🚀 Running the System

After setup, you can easily run **Portfolio Maximizer** with a few commands.

### 1. Run the ETL Pipeline

To run the ETL (Extract, Transform, Load) pipeline:

```bash
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2020-01-01 --end 2024-01-01 --execution-mode auto
```

### 2. Launch the Autonomous Trading Loop

Next, launch the **auto-trader** to execute trades automatically:

```bash
python scripts/run_auto_trader.py --tickers AAPL,MSFT --initial-capital 25000 --cycles 5 --sleep-seconds 900
```

This will execute trades every 15 minutes (`900` seconds) until you stop it.

---

## 🔒 Advanced Features

### OpenClaw Integration

**Portfolio Maximizer** integrates with **OpenClaw** to send notifications and automate tasks.

Example command to send a notification:

```bash
openclaw message send --channel whatsapp --target +2347042437712 --message "Daily Portfolio Update"
```

### Autonomous Security Guard

**OpenClaw** provides an autonomous security guard, preventing high-risk actions such as secret exfiltration and irreversible financial decisions.

---

## 🔧 Troubleshooting

### Common Issues and Fixes

**1. Cache Not Working**

```bash
# Check the cache directory permissions
ls -la data/raw/

# Clear cache if corrupted
rm data/raw/*.parquet
```

**2. Import Errors**

```bash
# Ensure the virtual environment is activated
source simpleTrader_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 💡 Tips and Best Practices

* **Test First**: Run the pipeline in **synthetic mode** first to ensure everything is working correctly before going live.
* **Automate Tasks**: Use **cron jobs** for recurring tasks such as portfolio audits.
* **Monitor Performance**: Regularly check performance metrics to ensure the system is functioning as expected.

---

## 🧑‍💻 Contributing

We welcome contributions to **Portfolio Maximizer**! Here's how you can get involved:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Write tests** to ensure functionality
4. **Commit your changes** (`git commit -m 'feat: Add amazing feature'`)
5. **Push your changes** (`git push origin feature/amazing-feature`)
6. **Open a pull request**

---

## 🔖 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.

---

## 📞 Support

If you have any issues or need help:

1. **Check Documentation**: Refer to the `Documentation/` directory.
2. **Search Issues**: Check the [GitHub Issues](https://github.com/mrbestnaija/portfolio_maximizer/issues) section.
3. **Open a New Issue**: If you need further assistance, open a new issue on GitHub.

---

### **Final Thoughts**

This guide simplifies the setup and usage of **Portfolio Maximizer**, helping both newcomers and developers get started quickly. Whether you're just beginning or contributing to the project, the steps are now easier to follow with clear instructions and examples!

```
```
