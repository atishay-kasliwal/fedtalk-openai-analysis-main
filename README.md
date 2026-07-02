# FedTalk: Federal Reserve Announcements Impact Analysis

A research project analyzing whether an LLM (GPT-4o via the OpenAI API) can predict short-horizon
stock market reactions to FOMC (Federal Open Market Committee) statements and press conferences,
using speech transcripts, related news articles, and minute-level price/volatility data around
each announcement.

Full methodology and findings are in [docs/Report.pdf](docs/Report.pdf).

## How it works

1. **Transcribe**: FOMC press conference audio/video is chunked and transcribed (Whisper).
2. **Retrieve context**: For each minute-level window, the most similar sentences from the FOMC
   statement and from contemporaneous news articles are retrieved via embedding search (Pinecone +
   sentence-transformers).
3. **Predict**: The statement/news context is sent to an LLM (OpenAI), which predicts whether the
   market reaction will be Positive or Negative and explains why.
4. **Score**: Predictions are compared against actual SPY price movement and volatility (Alpaca
   market data) at 1-minute, 5-minute, and 10-minute intervals, and scored with
   accuracy/precision/recall/F1.

On the labeled positive-prediction set, the model reached ~0.59 accuracy and ~0.74 weighted F1
against actual market direction (see `data/processed/metrics_positive.csv`). Volatility clustered
noticeably higher in the minutes right after the statement's release across all three interval
granularities (see `data/processed/volatility_overall.csv`).

## Project structure

```
fedtalk-openai-analysis-main/
├── src/fedtalk/            # Main Python package
│   ├── analysis/           # LLM prediction + evaluation (analysis_util.py)
│   ├── pipeline/           # End-to-end pipeline, news retrieval, video chunking
│   ├── utils/              # Finance (Alpaca), media (Whisper/moviepy), Pinecone, viz
│   └── api_keys.py.example # Copy to api_keys.py and fill in your keys (gitignored)
├── data/
│   ├── raw/                # Transcripts, statements, articles, price data (data_1Min/5Min/10Min)
│   └── processed/          # Prediction outputs, filtered/merged CSVs, metrics
├── docs/                   # Report.pdf and project structure notes
├── notebooks/              # Example notebook
├── tests/                  # Basic import tests
└── config/config.yaml      # Timeframes, symbols, thresholds
```

## Installation

```bash
git clone git@github.com:atishay-kasliwal/fedtalk-openai-analysis-main.git
cd fedtalk-openai-analysis-main
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API keys

Copy the template and fill in your own keys — `api_keys.py` is gitignored and must never be committed:

```bash
cp src/fedtalk/api_keys.py.example src/fedtalk/api_keys.py
```

Alpaca market-data credentials are read from environment variables instead:

```bash
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
```

## Usage

```python
from fedtalk.utils import finance_util

# Get SPY price change / volatility for a given window (fetches + caches to data/raw/data_1Min/price/
# if not already present locally)
import datetime
price_change = finance_util.get_price_change(
    datetime.datetime(2024, 7, 31, 18, 30),
    datetime.datetime(2024, 7, 31, 18, 35),
)
```

```python
from fedtalk.analysis import analysis_util

# Get an LLM market-reaction prediction for a batch of statement/news snippets
predictions, metrics = analysis_util.get_market_reaction_predictions(train_batch, test_batch)
```

Run the full end-to-end pipeline (requires all data/API keys in place):

```bash
python -m fedtalk.pipeline.pipeline
# or, after `pip install -e .`:
fedtalk
```

## Testing

```bash
pytest tests/
```

## Data sources

- **FOMC statements & press conference transcripts** — 2024 meeting dates
- **News articles** — contemporaneous financial news coverage
- **Market data** — SPY price and volatility via Alpaca, at 1/5/10-minute intervals
- **LLM predictions & similarity scores** — generated via the pipeline above

## Limitations

- FOMC meetings occur ~8x/year, so despite many per-minute rows, the number of independent events
  is small — treat aggregate accuracy/F1 as indicative, not statistically definitive.
- No trivial baseline (e.g., always-predict-majority-class) is currently reported alongside model
  accuracy; the precision=1.0/recall=0.59 pattern in `metrics_positive.csv` suggests the model may
  be avoiding one class rather than discriminating well — worth investigating further.

## License

MIT — see [LICENSE](LICENSE).

## Author

Atishay Kasliwal
