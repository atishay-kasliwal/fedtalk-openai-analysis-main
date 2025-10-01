#!/bin/bash

echo "🚀 Pushing code to git repository..."

# Navigate to project directory
cd "/Users/atishaykasliwal/untitled folder 2/fedtalk-openai-analysis-main"

# Remove any git lock files
rm -f .git/index.lock

# Add all changes
echo "📁 Adding all changes..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Add ChatGPT-style BERT implementation with matching naming conventions and rolling window logic

- Created chatgpt_style_bert_predictor.py with identical logic to ChatGPT
- Added rolling window context (5-record temporal window)
- Matched all naming conventions (functions, variables, file paths)
- Added cross-interval training approach
- Created run_chatgpt_style_bert.py runner script
- Added comprehensive documentation and comparison files
- All edge cases handled identically to ChatGPT implementation"

# Push to remote
echo "⬆️ Pushing to remote repository..."
git push

echo "✅ Code successfully pushed to repository!"
