# FedTalk Project Structure Documentation

## Overview

FedTalk is a comprehensive toolkit for analyzing the relationship between Federal Open Market Committee (FOMC) announcements and stock market reactions. The project is organized using modern Python packaging standards for maintainability and scalability.

## Directory Structure

```
fedtalk-openai-analysis-main/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation script
├── .gitignore                  # Git ignore patterns
├── config/                     # Configuration files
│   └── config.yaml            # Main configuration
├── src/                        # Source code (Python package)
│   └── fedtalk/               # Main package
│       ├── __init__.py         # Package initialization
│       ├── analysis/           # Analysis utilities
│       │   ├── __init__.py
│       │   └── analysis_util.py
│       ├── data/               # Data management
│       │   └── __init__.py
│       ├── utils/              # Helper utilities
│       │   ├── __init__.py
│       │   ├── articles_util.py
│       │   ├── db_util.py
│       │   ├── finance_util.py
│       │   ├── media_util.py
│       │   └── visualizations_util.py
│       └── pipeline/           # Data processing pipeline
│           ├── __init__.py
│           ├── pipeline.py
│           ├── news.py
│           └── video_pro.py
├── data/                       # Data files
│   ├── raw/                    # Raw data files
│   │   ├── data_1Min/         # 1-minute interval data
│   │   ├── data_5Min/         # 5-minute interval data
│   │   ├── data_10Min/        # 10-minute interval data
│   │   ├── *.txt              # Text files
│   │   └── *.mp3              # Audio files
│   ├── processed/              # Processed data files
│   │   ├── *.csv              # CSV data files
│   │   └── *.numbers          # Numbers files
│   └── results/                # Analysis results
├── docs/                       # Documentation
│   ├── PROJECT_STRUCTURE.md    # This file
│   └── Report.pdf              # Original report
├── notebooks/                  # Jupyter notebooks
│   └── 01_quick_start.ipynb   # Quick start guide
├── tests/                      # Test files
│   ├── __init__.py
│   └── test_basic.py          # Basic tests
├── outputs/                    # Generated outputs
└── logs/                       # Log files
```

## Module Organization

### 1. Analysis Module (`src/fedtalk/analysis/`)
- **Purpose**: Core analysis functions for market data and sentiment
- **Key Files**: `analysis_util.py` - Main analysis utilities
- **Responsibilities**: Statistical analysis, sentiment scoring, market reaction analysis

### 2. Utils Module (`src/fedtalk/utils/`)
- **Purpose**: Helper functions and utilities
- **Key Files**:
  - `articles_util.py` - Article processing utilities
  - `db_util.py` - Database operations
  - `finance_util.py` - Financial calculations
  - `media_util.py` - Media file processing
  - `visualizations_util.py` - Chart and graph generation

### 3. Pipeline Module (`src/fedtalk/pipeline/`)
- **Purpose**: Data processing workflows and automation
- **Key Files**:
  - `pipeline.py` - Main pipeline orchestration
  - `news.py` - News data processing
  - `video_pro.py` - Video processing utilities

### 4. Data Module (`src/fedtalk/data/`)
- **Purpose**: Data management and organization
- **Responsibilities**: Data loading, validation, and storage

## Data Organization

### Raw Data (`data/raw/`)
- **Time-based Directories**: Organized by analysis intervals (1Min, 5Min, 10Min)
- **File Types**: Text files, audio files, original data sources
- **Structure**: Each timeframe contains subdirectories for articles, price data, statements, etc.

### Processed Data (`data/processed/`)
- **CSV Files**: Cleaned and processed data ready for analysis
- **Numbers Files**: Spreadsheet data in Numbers format
- **Combined Files**: Merged datasets from multiple sources

### Results (`data/results/`)
- **Analysis Outputs**: Generated charts, graphs, and reports
- **Intermediate Results**: Temporary files from analysis pipeline

## Configuration

The project uses a centralized configuration system:
- **Location**: `config/config.yaml`
- **Purpose**: Centralized settings for data paths, analysis parameters, and output options
- **Benefits**: Easy modification without code changes, environment-specific configurations

## Package Installation

The project can be installed as a Python package:
```bash
# Development installation
pip install -e .

# Production installation
pip install .
```

## Development Workflow

1. **Setup**: Install dependencies with `pip install -r requirements.txt`
2. **Development**: Use `pip install -e .` for development mode
3. **Testing**: Run tests with `pytest tests/`
4. **Documentation**: Update README.md and docs/ as needed

## File Naming Conventions

- **Python Files**: snake_case (e.g., `analysis_util.py`)
- **Directories**: snake_case (e.g., `data_1Min/`)
- **Data Files**: descriptive names with dates when applicable
- **Configuration**: lowercase with descriptive names

## Best Practices

1. **Modular Design**: Each module has a single responsibility
2. **Clear Imports**: Use relative imports within the package
3. **Documentation**: Each module and function has docstrings
4. **Testing**: Unit tests for all major functionality
5. **Configuration**: Externalize configuration parameters
6. **Logging**: Comprehensive logging for debugging and monitoring

## Future Improvements

1. **API Layer**: REST API for data access
2. **Web Interface**: Dashboard for analysis results
3. **Database Integration**: Proper database backend
4. **Real-time Processing**: Streaming data analysis
5. **Machine Learning**: Advanced predictive models
