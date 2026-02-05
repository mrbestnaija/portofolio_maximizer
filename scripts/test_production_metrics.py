#!/usr/bin/env python3
"""Quick test of production_only filtering"""

from etl.database_manager import DatabaseManager

with DatabaseManager() as db:
    # Test with production_only=False (all data)
    all_data = db.get_performance_summary(production_only=False)
    print("=== ALL DATA (including test/synthetic) ===")
    print(f"Total trades: {all_data['total_trades']}")
    print(f"Total profit: ${all_data['total_profit'] or 0:.2f}")
    print(f"Win rate: {all_data['win_rate']:.1%}")
    print(f"Profit factor: {all_data['profit_factor']}")
    print(f"Table used: {all_data['table_used']}")
    print()

    # Test with production_only=True (exclude test data)
    prod_data = db.get_performance_summary(production_only=True)
    print("=== PRODUCTION DATA ONLY ===")
    print(f"Total trades: {prod_data['total_trades']}")
    print(f"Total profit: ${prod_data['total_profit'] or 0:.2f}")
    print(f"Win rate: {prod_data['win_rate']:.1%}")
    print(f"Profit factor: {prod_data['profit_factor']}")
    print(f"Table used: {prod_data['table_used']}")
    print()

    # Show the difference
    print("=== DIFFERENCE (test/synthetic data) ===")
    print(f"Test trades: {all_data['total_trades'] - prod_data['total_trades']}")
    print(f"Test profit: ${(all_data['total_profit'] or 0) - (prod_data['total_profit'] or 0):.2f}")
