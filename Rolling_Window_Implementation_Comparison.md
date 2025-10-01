# Rolling Window Implementation Comparison
## ChatGPT vs BERT (ChatGPT-Style) Approaches

## 🎯 **Key Implementation Differences**

### **ChatGPT Rolling Window Approach**
```python
# ChatGPT: Temporal Context Window
def create_windowed_test_prompt(data_df, text_col, window_size=5):
    records = []
    df_sorted = data_df.sort_values('start_time').reset_index(drop=True)
    for idx, row in df_sorted.iterrows():
        # Build context from previous rows within window (no future leakage)
        start_idx = max(0, idx - window_size)
        context_rows = df_sorted.iloc[start_idx:idx]
        prior_context = []
        for _, prow in context_rows.iterrows():
            prior_context.append({
                'Prev_Time': str(prow['start_time']),
                'Prev_Speech': str(prow.get('speech', ''))[:160] + "...",
                'Prev_Text': str(pc_text)[:120] + "...",
                'Prev_Similarity': float(prow.get('average_similarity_score', 0))
            })
```

### **BERT Rolling Window Approach (New Implementation)**
```python
# BERT: Same Temporal Context Window Logic
def create_chatgpt_style_prompts(self, data_df, text_col, is_training=True):
    data_sorted = data_df.sort_values('start_time').reset_index(drop=True)
    
    for idx, row in data_sorted.iterrows():
        # Build context from previous rows within window (same as ChatGPT)
        window_size = 5  # Same as ChatGPT
        start_idx = max(0, idx - window_size)
        context_rows = data_sorted.iloc[start_idx:idx]
        
        prior_context = []
        for _, prow in context_rows.iterrows():
            prior_context.append({
                'Prev_Time': str(prow['start_time']),
                'Prev_Speech': str(prow.get('speech', ''))[:160] + "...",
                'Prev_Text': str(pc_text)[:120] + "...",
                'Prev_Similarity': float(prow.get('average_similarity_score', 0))
            })
```

## 📊 **Rolling Window Characteristics**

| Aspect | ChatGPT | BERT (ChatGPT-Style) |
|--------|---------|----------------------|
| **Window Size** | 5 records | 5 records |
| **Context Type** | Temporal (previous records) | Temporal (previous records) |
| **Future Leakage** | Prevented | Prevented |
| **Context Building** | `start_idx = max(0, idx - window_size)` | `start_idx = max(0, idx - window_size)` |
| **Data Sorting** | `sort_values('start_time')` | `sort_values('start_time')` |
| **Context Fields** | Prev_Time, Prev_Speech, Prev_Text, Prev_Similarity | Prev_Time, Prev_Speech, Prev_Text, Prev_Similarity |

## 🔄 **Rolling Window Flow Comparison**

### **ChatGPT Flow**
```
1. Sort data by start_time
2. For each record at index idx:
   - Get window: data[max(0, idx-5):idx]
   - Build Prior_Context from window
   - Create prompt with current + prior context
   - Send to GPT API
   - Parse response
```

### **BERT Flow (ChatGPT-Style)**
```
1. Sort data by start_time
2. For each record at index idx:
   - Get window: data[max(0, idx-5):idx]  # SAME LOGIC
   - Build Prior_Context from window      # SAME LOGIC
   - Create combined text with context    # SAME LOGIC
   - Tokenize and predict with BERT       # DIFFERENT: Local model
```

## 🎯 **Key Similarities (Now Matching)**

### ✅ **Temporal Context Window**
- Both use 5-record rolling window
- Both prevent future data leakage
- Both sort by `start_time` before processing
- Both build `Prior_Context` from previous records

### ✅ **Data Processing**
- Both convert price changes to percentages
- Both use same date filtering (2021-2023 train, 2024 test)
- Both implement warmup logic (skip first timestamp per date)
- Both handle missing data with fallbacks

### ✅ **Edge Case Handling**
- Both handle NaN values in text columns
- Both truncate text to fixed lengths
- Both implement same column selection logic
- Both handle smoke test limits

### ✅ **Prediction Logic**
- Both predict future intervals (not first interval)
- Both use same train/test split approach
- Both implement same validation methodology
- Both calculate same metrics

## 🔧 **Implementation Differences**

### **ChatGPT Approach**
```python
# API-based prediction
response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
# Parse JSON response
```

### **BERT Approach (ChatGPT-Style)**
```python
# Local model prediction
combined_text = f"Current Speech: {current_speech} "
combined_text += f"News Context: {news_context} "
combined_text += f"Statement Context: {statement_context} "
combined_text += f"Prior Context: {prior_text}"

# Tokenize and predict
encoding = self.tokenizer(combined_text, ...)
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
direction_pred = torch.argmax(outputs.logits, dim=1).item()
```

## 📈 **Rolling Window Benefits**

### **Temporal Context**
- **ChatGPT**: Uses natural language understanding of temporal relationships
- **BERT**: Uses learned embeddings from temporal context

### **Pattern Recognition**
- **ChatGPT**: Recognizes patterns through few-shot examples
- **BERT**: Learns patterns through gradient descent training

### **Context Integration**
- **ChatGPT**: Integrates context through prompt engineering
- **BERT**: Integrates context through tokenization and attention

## 🎯 **Rolling Window Edge Cases (Both Handle)**

### ✅ **Empty Context**
```python
# Both handle when no prior context exists
if idx < window_size:
    context_rows = data_sorted.iloc[0:idx]  # Get available history
```

### ✅ **Missing Data**
```python
# Both handle missing text with fallbacks
pc_text = prow.get(text_col, None)
if pc_text is None or (isinstance(pc_text, float) and pd.isna(pc_text)):
    pc_text = prow.get('speech', '')
```

### ✅ **Text Truncation**
```python
# Both truncate text to fixed lengths
'Prev_Speech': str(prow.get('speech', ''))[:160] + "...",
'Prev_Text': str(pc_text)[:120] + "...",
```

### ✅ **Date Boundaries**
```python
# Both handle date boundaries in warmup logic
pred_sorted['date'] = pred_sorted['start_time'].dt.date
first_per_date_idx = pred_sorted.groupby('date', as_index=False).head(1).index
```

## 🚀 **Usage Examples**

### **ChatGPT Style**
```bash
# Run ChatGPT predictions
python run_chatgpt_predictions.py
```

### **BERT (ChatGPT-Style)**
```bash
# Run BERT with same logic as ChatGPT
python run_chatgpt_style_bert.py --train-interval 1 --pred-interval 5

# Run all combinations
python run_chatgpt_style_bert.py --all

# Smoke test
python run_chatgpt_style_bert.py --all --smoke-test
```

## 📊 **Expected Results Comparison**

| Metric | ChatGPT | BERT (ChatGPT-Style) |
|--------|---------|----------------------|
| **Rolling Window Size** | 5 records | 5 records |
| **Context Integration** | Natural language | Learned embeddings |
| **Prediction Speed** | ~2-5 seconds per batch | ~0.1 seconds per batch |
| **Accuracy** | 60-70% (estimated) | 55-65% (estimated) |
| **Interpretability** | High (explanations) | Low (black box) |
| **Cost** | ~$0.01-0.05 per prediction | Free after training |

## 🎉 **Conclusion**

The new BERT implementation now **exactly matches** the ChatGPT rolling window approach:

- ✅ **Same rolling window logic** (5-record temporal context)
- ✅ **Same edge case handling** (missing data, truncation, boundaries)
- ✅ **Same prediction methodology** (future intervals, warmup logic)
- ✅ **Same data processing** (date filtering, percentage conversion)
- ✅ **Same evaluation metrics** (direction accuracy, price MAE)

The only differences are:
- **ChatGPT**: API-based with natural language reasoning
- **BERT**: Local model with learned embeddings

Both approaches now use **identical rolling window implementations** for fair comparison!
