#!/bin/bash

# Quick Terminal Launcher - Immediately start processes in different terminals
# This script opens new terminal windows and starts specific processes

echo "🚀 Quick Terminal Launcher"
echo "========================="

# Set working directory
cd "/Users/atishaykasliwal/untitled folder 2/fedtalk-openai-analysis-main"

echo ""
echo "🎯 Choose what to launch in different terminals:"
echo ""
echo "1. 🔄 ChatGPT Predictions (All intervals)"
echo "2. 🤖 BERT Predictions (All types)"
echo "3. 📈 Parallel Processing (All runners)"
echo "4. 🔍 Analysis Suite (All analysis)"
echo "5. 🚀 Everything (Full pipeline)"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Launching ChatGPT Predictions in different terminals..."
        # Terminal 1 - 1-minute
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_single_interval_chatgpt.py --interval 1"'
        sleep 1
        
        # Terminal 2 - 5-minute
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_single_interval_chatgpt.py --interval 5"'
        sleep 1
        
        # Terminal 3 - 10-minute
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_single_interval_chatgpt.py --interval 10"'
        sleep 1
        
        # Terminal 4 - 15-minute
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_single_interval_chatgpt.py --interval 15"'
        ;;
    2)
        echo "🤖 Launching BERT Predictions in different terminals..."
        # Terminal 1 - Cross-interval BERT
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 corrected_cross_interval_bert_predictor.py"'
        sleep 1
        
        # Terminal 2 - Final BERT
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 final_corrected_bert_predictor.py"'
        sleep 1
        
        # Terminal 3 - Advanced BERT
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 advanced_bert_financial_predictor.py"'
        ;;
    3)
        echo "📈 Launching Parallel Processing in different terminals..."
        # Terminal 1 - Independent Runner
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 parallel_interval_runner_independent.py"'
        sleep 1
        
        # Terminal 2 - Standard Runner
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 parallel_interval_runner.py"'
        sleep 1
        
        # Terminal 3 - Clean Runner
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 clean_parallel_runner.py"'
        ;;
    4)
        echo "🔍 Launching Analysis Suite in different terminals..."
        # Terminal 1 - Statements News Analysis
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_statements_news_analysis.py"'
        sleep 1
        
        # Terminal 2 - Similarity Analysis
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 standalone_similarity_analysis.py"'
        sleep 1
        
        # Terminal 3 - Comprehensive Analysis
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 comprehensive_predictions.py"'
        ;;
    5)
        echo "🚀 Launching Everything in different terminals..."
        # Terminal 1 - ChatGPT Predictions
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_all_chatgpt_predictions_parallel.py"'
        sleep 2
        
        # Terminal 2 - BERT Predictions
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_bert_predictions.py"'
        sleep 2
        
        # Terminal 3 - Multi-interval Processing
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_multi_interval_processing.py"'
        sleep 2
        
        # Terminal 4 - Cross-interval Analysis
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_cross_interval_bert.py"'
        sleep 2
        
        # Terminal 5 - Similarity Analysis
        osascript -e 'tell application "Terminal" to do script "cd /Users/atishaykasliwal/untitled\\ folder\\ 2/fedtalk-openai-analysis-main && python3 run_similarity_analysis.py"'
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Processes launched in different terminals!"
echo ""
echo "📊 Monitor progress with:"
echo "   python3 monitor_progress.py"
echo ""
echo "🛑 To stop all processes:"
echo "   ./push_to_different_terminals.sh"
echo "   (Choose option 9)"
