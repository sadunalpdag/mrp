#!/usr/bin/env python3
"""
Test for the order entry issue fix.
This test verifies that open_market_position properly handles API errors.
"""
import sys
import os

# Add the current directory to the path to import ema module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the dependencies before importing ema
class MockRequest:
    def __init__(self):
        self.status_code = 200
    
    def json(self):
        return {}

def test_open_market_position_with_api_error():
    """Test that open_market_position handles API errors gracefully"""
    # Import after setting up path
    import ema
    
    # Save original _signed_request
    original_signed_request = ema._signed_request
    
    # Mock _signed_request to raise an exception
    def mock_signed_request_error(method, path, payload):
        raise RuntimeError("API Error: Mocked failure")
    
    # Replace the function
    ema._signed_request = mock_signed_request_error
    
    try:
        # Call open_market_position - it should NOT raise an exception
        result = ema.open_market_position("BTCUSDT", "UP", 0.001)
        
        # Verify the result
        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dict"
        assert result.get("symbol") == "BTCUSDT", "Symbol should match"
        assert result.get("dir") == "UP", "Direction should match"
        assert result.get("qty") == 0.001, "Quantity should match"
        assert result.get("entry") == 0.0, "Entry should be 0.0 on error"
        assert result.get("pos_side") == "LONG", "Position side should be LONG"
        
        print("✅ Test PASSED: open_market_position handles API errors correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        return False
    finally:
        # Restore original function
        ema._signed_request = original_signed_request


def test_open_market_position_with_success():
    """Test that open_market_position works correctly on success"""
    import ema
    
    # Save original _signed_request
    original_signed_request = ema._signed_request
    
    # Mock _signed_request to return success
    def mock_signed_request_success(method, path, payload):
        return {
            "orderId": 123456,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "avgPrice": "50000.5",
            "executedQty": "0.001"
        }
    
    # Replace the function
    ema._signed_request = mock_signed_request_success
    
    try:
        # Call open_market_position
        result = ema.open_market_position("BTCUSDT", "UP", 0.001)
        
        # Verify the result
        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dict"
        assert result.get("symbol") == "BTCUSDT", "Symbol should match"
        assert result.get("dir") == "UP", "Direction should match"
        assert result.get("qty") == 0.001, "Quantity should match"
        assert result.get("entry") == 50000.5, f"Entry should be 50000.5, got {result.get('entry')}"
        assert result.get("pos_side") == "LONG", "Position side should be LONG"
        
        print("✅ Test PASSED: open_market_position works correctly on success")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        return False
    finally:
        # Restore original function
        ema._signed_request = original_signed_request


if __name__ == "__main__":
    print("Running tests for order entry fix...\n")
    
    # Run tests
    test1_passed = test_open_market_position_with_api_error()
    test2_passed = test_open_market_position_with_success()
    
    # Summary
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
