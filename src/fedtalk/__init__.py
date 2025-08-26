"""
FedTalk: Analysis of Federal Reserve Announcements Impact on Stock Markets

A comprehensive toolkit for analyzing the relationship between FOMC announcements
and market reactions using various time intervals and data sources.
"""

__version__ = "1.0.0"
__author__ = "Atishay Kasliwal"

from . import analysis
from . import data
from . import utils
from . import pipeline

__all__ = ["analysis", "data", "utils", "pipeline"]
