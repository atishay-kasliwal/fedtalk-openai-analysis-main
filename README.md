# FedTalk: Federal Reserve Announcements Impact Analysis

A comprehensive toolkit for analyzing the relationship between Federal Open Market Committee (FOMC) announcements and stock market reactions using various time intervals and data sources.

## 🏗️ Project Structure

```
fedtalk-openai-analysis-main/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation script
├── src/                     # Source code
│   └── fedtalk/            # Main package
│       ├── __init__.py      # Package initialization
│       ├── analysis/        # Analysis utilities
│       ├── data/            # Data management
│       ├── utils/           # Helper utilities
│       └── pipeline/        # Data processing pipeline
├── data/                    # Data files
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── results/             # Analysis results
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test files
├── config/                  # Configuration files
└── outputs/                 # Generated outputs
```

## 🚀 Features

- **Multi-timeframe Analysis**: Analyze market reactions at 1-minute, 5-minute, and 10-minute intervals
- **Sentiment Analysis**: Process news articles and statements for sentiment scoring
- **Market Data Integration**: Combine price data with news sentiment
- **Visualization Tools**: Generate charts and graphs for analysis
- **Pipeline Automation**: Automated data processing workflows

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fedtalk-openai-analysis-main
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Basic Analysis
```python
from fedtalk.analysis import analysis_util
from fedtalk.pipeline import pipeline

# Run the main analysis pipeline
pipeline.run_analysis()
```

### Custom Analysis
```python
from fedtalk.utils import finance_util, media_util

# Analyze specific time periods
results = finance_util.analyze_market_reactions(
    start_date="2024-07-31",
    end_date="2024-12-31"
)
```

## 📊 Data Sources

- **FOMC Statements**: Official Federal Reserve announcements
- **News Articles**: Financial news coverage
- **Market Data**: Stock price movements and volatility
- **Sentiment Scores**: NLP-based sentiment analysis

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Atishay Kasliwal** - Financial Data Analyst and Researcher

## 🤝 Acknowledgments

- Federal Reserve Economic Data (FRED)
- Financial news sources
- Open-source data science community

---

For questions or support, please open an issue in the repository.
