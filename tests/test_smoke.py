# Smoke test to check imports are working 
import backtester 

def test_package_imports() -> None:
    assert backtester is not None # Check backtester imported 

