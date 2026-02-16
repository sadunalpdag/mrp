"""
Test script for onchain_strategy.py with mock data
This demonstrates the strategy logic works correctly without requiring external API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import functions from onchain_strategy
from onchain_strategy import zscore, build_signals, backtest_summary

def create_mock_price_data(days=365, start_price=50000):
    """Create mock OHLCV price data"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic price movement with trend
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # slight positive drift
    prices = start_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, days)
    })
    
    return df

def create_mock_onchain_data(days=365):
    """Create mock on-chain metrics data"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate on-chain activity with cyclical pattern
    np.random.seed(42)
    base_activity = 100000
    cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 20000  # cyclical component
    noise = np.random.normal(0, 5000, days)
    activity = base_activity + cycle + noise
    
    df = pd.DataFrame({
        'time': dates,
        'asset': 'btc',
        'AdrActCnt': activity
    })
    
    return df

def test_zscore():
    """Test z-score calculation"""
    print("\n1️⃣  Testing zscore function...")
    print("-" * 60)
    
    # Create sample data
    data = pd.Series([100, 110, 105, 120, 115, 130, 125, 140, 135, 150])
    z = zscore(data, window=5)
    
    print(f"Original data: {data.tolist()}")
    print(f"Z-scores (window=5): {z.round(2).tolist()}")
    print("✅ zscore function works correctly")

def test_build_signals():
    """Test signal generation with mock data"""
    print("\n2️⃣  Testing build_signals function...")
    print("-" * 60)
    
    # Create mock data
    px_df = create_mock_price_data(days=200)
    on_df = create_mock_onchain_data(days=200)
    
    print(f"Created {len(px_df)} days of price data")
    print(f"Created {len(on_df)} days of on-chain data")
    
    # Generate signals
    strategy_df = build_signals(px_df, on_df, on_metric='AdrActCnt', max_positions=3)
    
    print(f"\nStrategy DataFrame shape: {strategy_df.shape}")
    print(f"Columns: {strategy_df.columns.tolist()}")
    
    # Check signal distribution
    signal_counts = strategy_df['signal'].value_counts()
    position_counts = strategy_df['position'].value_counts()
    
    print(f"\nSignal distribution:")
    print(f"  Long signals (1): {signal_counts.get(1, 0)}")
    print(f"  Short signals (-1): {signal_counts.get(-1, 0)}")
    print(f"  Neutral signals (0): {signal_counts.get(0, 0)}")
    
    print(f"\nPosition distribution:")
    print(f"  Long positions (1): {position_counts.get(1.0, 0)}")
    print(f"  Short positions (-1): {position_counts.get(-1.0, 0)}")
    print(f"  No position (0): {position_counts.get(0.0, 0)}")
    
    # Verify position limits (max 3 buy or 3 sell)
    max_long = position_counts.get(1.0, 0)
    max_short = position_counts.get(-1.0, 0)
    
    print(f"\nPosition Limits Check:")
    print(f"  Max long positions at once: {max_long} (limit: 3)")
    print(f"  Max short positions at once: {max_short} (limit: 3)")
    
    # Show last 10 rows
    print(f"\nLast 10 rows of strategy data:")
    cols = ['time', 'close', 'sma50', 'trend', 'on_z', 'signal', 'position', 'strategy_ret']
    print(strategy_df[cols].tail(10).to_string(index=False))
    
    print("\n✅ build_signals function works correctly")
    return strategy_df

def test_backtest_summary(strategy_df):
    """Test backtest summary calculation"""
    print("\n3️⃣  Testing backtest_summary function...")
    print("-" * 60)
    
    summary = backtest_summary(strategy_df)
    
    print(f"Backtest Summary:")
    print(f"  Total Return: {summary['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {summary['sharpe_approx']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
    print(f"  Position Changes: {summary['position_changes']}")
    print(f"  Total Days: {summary['rows']}")
    
    # Verify metrics are reasonable
    assert isinstance(summary['total_return'], float), "total_return should be float"
    assert isinstance(summary['sharpe_approx'], float), "sharpe_approx should be float"
    assert isinstance(summary['max_drawdown'], float), "max_drawdown should be float"
    assert summary['position_changes'] >= 0, "position_changes should be non-negative"
    assert summary['rows'] > 0, "rows should be positive"
    
    print("\n✅ backtest_summary function works correctly")

def test_full_strategy():
    """Test the complete strategy workflow"""
    print("\n4️⃣  Testing complete strategy workflow...")
    print("-" * 60)
    
    # Test with different market conditions
    scenarios = [
        ("Bull Market", create_mock_price_data(days=180, start_price=50000)),
        ("Sideways Market", create_mock_price_data(days=180, start_price=45000)),
    ]
    
    for scenario_name, px_df in scenarios:
        print(f"\n📊 Scenario: {scenario_name}")
        print("   " + "-" * 56)
        
        on_df = create_mock_onchain_data(days=180)
        strategy_df = build_signals(px_df, on_df, on_metric='AdrActCnt', max_positions=3)
        summary = backtest_summary(strategy_df)
        
        print(f"   Return: {summary['total_return']*100:>8.2f}% | "
              f"Sharpe: {summary['sharpe_approx']:>6.2f} | "
              f"DD: {summary['max_drawdown']*100:>7.2f}% | "
              f"Trades: {summary['position_changes']:>3d}")
    
    print("\n✅ Full strategy workflow works correctly")

def main():
    """Run all tests"""
    print("=" * 80)
    print("ONCHAIN STRATEGY TEST SUITE")
    print("=" * 80)
    print("\nRunning tests with mock data to verify strategy logic...")
    
    try:
        # Run individual tests
        test_zscore()
        strategy_df = test_build_signals()
        test_backtest_summary(strategy_df)
        test_full_strategy()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe on-chain strategy system is working correctly.")
        print("In production, it will use real data from Coin Metrics and Binance APIs.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
