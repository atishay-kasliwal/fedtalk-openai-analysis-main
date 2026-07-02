"""
Basic tests for FedTalk package
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_package_import():
    """Test that the package can be imported"""
    try:
        import fedtalk
        assert fedtalk.__version__ == "1.0.0"
    except ImportError as e:
        pytest.fail(f"Failed to import fedtalk: {e}")

def test_analysis_import():
    """Test that analysis module can be imported"""
    try:
        from fedtalk.analysis import analysis_util
        assert analysis_util is not None
    except ImportError as e:
        pytest.fail(f"Failed to import analysis_util: {e}")

def test_utils_import():
    """Test that utils modules can be imported"""
    try:
        from fedtalk.utils import finance_util, media_util
        assert finance_util is not None
        assert media_util is not None
    except ImportError as e:
        pytest.fail(f"Failed to import utils: {e}")

def test_pipeline_import():
    """Test that pipeline module can be imported"""
    try:
        from fedtalk.pipeline import pipeline
        assert pipeline is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pipeline: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
