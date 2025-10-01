#!/bin/bash

# Push Everything to Different Terminals
# This script helps distribute all running processes across multiple terminals

echo "🚀 Fedtalk Analysis - Push Everything to Different Terminals"
echo "=========================================================="

# Set working directory
cd "/Users/atishaykasliwal/untitled folder 2/fedtalk-openai-analysis-main"

# Check current running processes
echo "📊 Current running processes:"
ps aux | grep python | grep -v grep

echo ""
echo "🎯 Available options to push to different terminals:"
echo ""
echo "1. 🔄 ChatGPT Predictions (Multi-interval)"
echo "2. 🤖 BERT Predictions (Cross-interval)"
echo "3. 📈 Parallel Interval Processing"
echo "4. 🔍 Similarity Analysis"
echo "5. 📊 Comprehensive Analysis"
echo "6. 🎯 Single Interval Processing"
echo "7. 🚀 All Processing (Full Pipeline)"
echo "8. 📝 Create Individual Terminal Scripts"
echo "9. 🛑 Stop All Running Processes"
echo ""

read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo "🚀 Pushing ChatGPT Predictions to different terminals..."
        echo ""
        echo "Terminal 1 - 1-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 1"
        echo ""
        echo "Terminal 2 - 5-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 5"
        echo ""
        echo "Terminal 3 - 10-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 10"
        echo ""
        echo "Terminal 4 - 15-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 15"
        echo ""
        echo "Terminal 5 - 20-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 20"
        echo ""
        echo "Terminal 6 - 25-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 25"
        echo ""
        echo "Terminal 7 - 30-minute intervals:"
        echo "python3 run_single_interval_chatgpt.py --interval 30"
        ;;
    2)
        echo "🤖 Pushing BERT Predictions to different terminals..."
        echo ""
        echo "Terminal 1 - Cross-interval BERT:"
        echo "python3 corrected_cross_interval_bert_predictor.py"
        echo ""
        echo "Terminal 2 - Final BERT Predictor:"
        echo "python3 final_corrected_bert_predictor.py"
        echo ""
        echo "Terminal 3 - Advanced BERT:"
        echo "python3 advanced_bert_financial_predictor.py"
        ;;
    3)
        echo "📈 Pushing Parallel Interval Processing to different terminals..."
        echo ""
        echo "Terminal 1 - Independent Parallel Runner:"
        echo "python3 parallel_interval_runner_independent.py"
        echo ""
        echo "Terminal 2 - Standard Parallel Runner:"
        echo "python3 parallel_interval_runner.py"
        echo ""
        echo "Terminal 3 - Clean Parallel Runner:"
        echo "python3 clean_parallel_runner.py"
        ;;
    4)
        echo "🔍 Pushing Similarity Analysis to different terminals..."
        echo ""
        echo "Terminal 1 - Statements News Analysis:"
        echo "python3 run_statements_news_analysis.py"
        echo ""
        echo "Terminal 2 - Standalone Similarity:"
        echo "python3 standalone_similarity_analysis.py"
        echo ""
        echo "Terminal 3 - Simple Analysis:"
        echo "python3 run_simple_analysis.py"
        ;;
    5)
        echo "📊 Pushing Comprehensive Analysis to different terminals..."
        echo ""
        echo "Terminal 1 - Comprehensive Predictions:"
        echo "python3 comprehensive_predictions.py"
        echo ""
        echo "Terminal 2 - Comprehensive Verification:"
        echo "python3 comprehensive_verification.py"
        echo ""
        echo "Terminal 3 - Run Analysis:"
        echo "python3 run_analysis.py"
        ;;
    6)
        echo "🎯 Pushing Single Interval Processing to different terminals..."
        echo ""
        echo "Available intervals: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45"
        read -p "Enter interval (minutes): " interval
        echo ""
        echo "Terminal 1 - $interval-minute interval:"
        echo "python3 run_single_interval_chatgpt.py --interval $interval"
        echo ""
        echo "Terminal 2 - BERT for $interval-minute:"
        echo "python3 bert_financial_predictor.py --interval $interval"
        ;;
    7)
        echo "🚀 Pushing All Processing to different terminals..."
        echo ""
        echo "Terminal 1 - ChatGPT Predictions:"
        echo "python3 run_all_chatgpt_predictions_parallel.py"
        echo ""
        echo "Terminal 2 - BERT Predictions:"
        echo "python3 run_bert_predictions.py"
        echo ""
        echo "Terminal 3 - Multi-interval Processing:"
        echo "python3 run_multi_interval_processing.py"
        echo ""
        echo "Terminal 4 - Cross-interval Analysis:"
        echo "python3 run_cross_interval_bert.py"
        echo ""
        echo "Terminal 5 - Similarity Analysis:"
        echo "python3 run_similarity_analysis.py"
        ;;
    8)
        echo "📝 Creating individual terminal scripts..."
        python3 parallel_interval_runner_independent.py --create-scripts --intervals 1 5 10 15 20 25 30
        echo ""
        echo "✅ Individual scripts created!"
        echo ""
        echo "🚀 To run in separate terminals:"
        echo "Terminal 1: python3 run_1min_independent.py"
        echo "Terminal 2: python3 run_5min_independent.py"
        echo "Terminal 3: python3 run_10min_independent.py"
        echo "Terminal 4: python3 run_15min_independent.py"
        echo "Terminal 5: python3 run_20min_independent.py"
        echo "Terminal 6: python3 run_25min_independent.py"
        echo "Terminal 7: python3 run_30min_independent.py"
        ;;
    9)
        echo "🛑 Stopping all running Python processes..."
        pkill -f "python.*run_chatgpt_predictions"
        pkill -f "python.*bert"
        pkill -f "python.*parallel"
        pkill -f "python.*analysis"
        echo "✅ All processes stopped!"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "💡 Instructions:"
echo "   1. Open new terminal windows/tabs"
echo "   2. Copy and paste the commands above into each terminal"
echo "   3. Press Enter to start each process"
echo ""
echo "📊 Monitor progress with:"
echo "   python3 monitor_progress.py"
echo ""
echo "✅ Done!"
