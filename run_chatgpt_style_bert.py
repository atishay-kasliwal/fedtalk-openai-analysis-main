#!/usr/bin/env python3
"""
Runner script for ChatGPT-Style BERT Financial Predictor
Matches ChatGPT implementation logic exactly
"""

import sys
import os
sys.path.append('.')

from chatgpt_style_bert_predictor import predict_multi_interval_price_change_chatgpt
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_combination(train_interval, pred_interval, smoke_test=False):
    """Run single train-predict combination"""
    print(f"🚀 Running BERT (ChatGPT-style): Train on {train_interval}Min, Predict {pred_interval}Min")
    
    smoke_limit = 10 if smoke_test else None
    
    try:
        results = predict_multi_interval_price_change_chatgpt(
            train_interval=train_interval,
            prediction_intervals=[pred_interval],
            price_change_threshold=0.0001,
            num_statement_matches="5",
            num_news_matches="5",
            smoke_limit_test_rows=smoke_limit
        )
        
        if results:
            key = f"{train_interval}Min_train_{pred_interval}Min_pred"
            if key in results:
                metrics = results[key]['metrics']
                print(f"✅ Results for {train_interval}Min → {pred_interval}Min:")
                print(f"   Direction Accuracy: {metrics['Direction_Accuracy']:.4f}")
                print(f"   Price Change MAE: {metrics.get('Price_Change_MAE', 0):.6f}%")
                print(f"   Total Predictions: {metrics['Total_Predictions']}")
                print(f"   Valid Predictions: {metrics['Valid_Predictions']}")
                return True
            else:
                print(f"❌ No results found for {train_interval}Min → {pred_interval}Min")
                return False
        else:
            print(f"❌ Failed to generate results for {train_interval}Min → {pred_interval}Min")
            return False
            
    except Exception as e:
        print(f"❌ Error running {train_interval}Min → {pred_interval}Min: {e}")
        return False

def run_all_combinations(smoke_test=False):
    """Run all train-predict combinations"""
    print("🚀 Running ALL ChatGPT-Style BERT combinations...")
    
    intervals = ["1", "5", "10", "15", "20", "25", "30"]
    success_count = 0
    total_count = 0
    
    for train_interval in intervals:
        for pred_interval in intervals:
            total_count += 1
            if run_single_combination(train_interval, pred_interval, smoke_test):
                success_count += 1
            print("-" * 60)
    
    print(f"\n🎉 Completed: {success_count}/{total_count} combinations successful")
    return success_count, total_count

def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ChatGPT-Style BERT Financial Predictor')
    parser.add_argument('--train-interval', type=str, help='Training interval (e.g., "1", "5")')
    parser.add_argument('--pred-interval', type=str, help='Prediction interval (e.g., "1", "5")')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test with limited data')
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_combinations(smoke_test=args.smoke_test)
    elif args.train_interval and args.pred_interval:
        run_single_combination(args.train_interval, args.pred_interval, smoke_test=args.smoke_test)
    else:
        print("Usage examples:")
        print("  python run_chatgpt_style_bert.py --train-interval 1 --pred-interval 5")
        print("  python run_chatgpt_style_bert.py --all")
        print("  python run_chatgpt_style_bert.py --all --smoke-test")

if __name__ == "__main__":
    main()
