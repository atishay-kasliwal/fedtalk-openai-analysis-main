"""
ChatGPT-Style BERT Financial Predictor
Matches ChatGPT implementation logic exactly:
- Rolling window context for training
- Predicts future intervals (not first interval)
- Same edge case handling
- Same warmup logic
- Same multi-interval approach
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import warnings
import os
import logging
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_reactions(market_reactions):
    """Aggregate multiple market reactions into a single prediction"""
    if not market_reactions:
        return "No Prediction"
    
    positive_count = sum(1 for reaction in market_reactions if reaction.get('Reaction') == 'Positive')
    negative_count = sum(1 for reaction in market_reactions if reaction.get('Reaction') == 'Negative')
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "No Prediction"

def create_insight(market_reactions):
    """Create insight from market reactions"""
    if not market_reactions:
        return "No insight available"
    
    insights = []
    for reaction in market_reactions:
        if 'Explanation' in reaction:
            insights.append(reaction['Explanation'])
    
    return " | ".join(insights[:3])  # Limit to first 3 insights

def create_s1(market_reactions):
    """Extract similarity score 1 from reactions"""
    if not market_reactions:
        return 0
    
    scores = []
    for reaction in market_reactions:
        if 'Similarity 1' in reaction:
            try:
                scores.append(float(reaction['Similarity 1']))
            except (ValueError, TypeError):
                pass
    
    return np.mean(scores) if scores else 0

def create_s2(market_reactions):
    """Extract similarity score 2 from reactions"""
    if not market_reactions:
        return 0
    
    scores = []
    for reaction in market_reactions:
        if 'Similarity 2' in reaction:
            try:
                scores.append(float(reaction['Similarity 2']))
            except (ValueError, TypeError):
                pass
    
    return np.mean(scores) if scores else 0

def create_s3(market_reactions):
    """Extract similarity score 3 from reactions"""
    if not market_reactions:
        return 0
    
    scores = []
    for reaction in market_reactions:
        if 'Similarity 3' in reaction:
            try:
                scores.append(float(reaction['Similarity 3']))
            except (ValueError, TypeError):
                pass
    
    return np.mean(scores) if scores else 0

def create_op3(market_reactions):
    """Extract price change from reactions"""
    if not market_reactions:
        return 0
    
    changes = []
    for reaction in market_reactions:
        if 'Percent Change' in reaction:
            try:
                changes.append(float(reaction['Percent Change']))
            except (ValueError, TypeError):
                pass
    
    return np.mean(changes) if changes else 0

class FinancialDataset(Dataset):
    """Dataset class matching ChatGPT approach with rolling window context"""
    
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Combine current speech + prior context (like ChatGPT)
        current_speech = record.get('Current_Minute_Speech', '')
        news_context = record.get('News_Context', '')
        statement_context = record.get('Statement_Context', '')
        
        # Build context similar to ChatGPT's Prior_Context
        prior_context = record.get('Prior_Context', [])
        prior_text = ""
        if prior_context:
            for i, ctx in enumerate(prior_context[-3:]):  # Last 3 contexts like ChatGPT
                prior_text += f"Prev_{i+1}: {ctx.get('Prev_Speech', '')} "
        
        # Combine all context (matching ChatGPT prompt structure)
        combined_text = f"Current Speech: {current_speech} "
        combined_text += f"News Context: {news_context} "
        combined_text += f"Statement Context: {statement_context} "
        combined_text += f"Prior Context: {prior_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'record_id': torch.tensor(record.get('Id', 0), dtype=torch.long),
            'price_change': torch.tensor(record.get('Actual_Price_Change_Percent', 0.0), dtype=torch.float),
            'direction': torch.tensor(1 if record.get('Price_Movement') == 'Positive' else 0, dtype=torch.long)
        }

class BERTPredictor:
    """BERT predictor matching ChatGPT implementation logic exactly"""
    
    def __init__(self, model_name='distilroberta-base', max_length=512, learning_rate=1e-6):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.smoke_test = os.environ.get('SMOKE_TEST', '0') == '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
    def load_interval_data(self, interval_minutes):
        """Load data for a specific interval - same as ChatGPT"""
        print(f"Loading {interval_minutes}-minute interval data...")
        
        if interval_minutes == 1:
            data_file = '1Min_data/combined_filtered_statements_and_news.csv'
        else:
            data_file = f'{interval_minutes}Min_data/combined_filtered_statements_and_news.csv'
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return None
        
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data: {len(df)} samples")
        
        # Convert dates
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        
        # Convert price_change to percentage format (same as ChatGPT)
        df['price_change_percent'] = df['price_change'] * 100
        
        # Split by date (2021-2023 train, 2024 test) - same as ChatGPT
        train_df = df[(df['start_time'] >= '2021-01-01') & (df['start_time'] <= '2023-12-31')]
        test_df = df[df['start_time'] >= '2024-01-01']
        
        print(f"Training samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")
        
        # Filter missing data
        train_df = train_df.dropna(subset=['speech', 'price_change'])
        test_df = test_df.dropna(subset=['speech', 'price_change'])
        
        print(f"After filtering - Training: {len(train_df)}, Testing: {len(test_df)}")
        
        # Reduce dataset size in smoke test mode
        if self.smoke_test:
            train_df = train_df.sample(n=min(64, len(train_df)), random_state=42) if len(train_df) > 0 else train_df
            test_df = test_df.sample(n=min(32, len(test_df)), random_state=42) if len(test_df) > 0 else test_df
            print(f"SMOKE_TEST: Using reduced sizes -> Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df
    
    def create_prompt(self, data_df, text_col, is_training=True):
        """Create prompts exactly like ChatGPT implementation"""
        print(f"Creating ChatGPT-style prompts for {'training' if is_training else 'testing'}...")
        
        # Sort by time (same as ChatGPT)
        data_sorted = data_df.sort_values('start_time').reset_index(drop=True)
        
        # Helper to pick best available text column (same logic as ChatGPT)
        def select_best_text_column(df, base):
            candidates = [f'{base}_5', f'{base}_3', f'{base}_1'] if base.startswith('extracted') else [base]
            existing = [(c, df[c].isna().mean()) for c in candidates if c in df.columns]
            if not existing:
                return 'speech'
            existing.sort(key=lambda x: (x[1], candidates.index(x[0]) if x[0] in candidates else 99))
            return existing[0][0]
        
        # Select best columns (same as ChatGPT)
        if text_col.startswith('extracted'):
            text_col = select_best_text_column(data_df, 'extracted_statement_text' if 'statement' in text_col else 'extracted_news')
        
        records = []
        
        for idx, row in data_sorted.iterrows():
            # Build context from previous rows within window (same as ChatGPT)
            window_size = 5  # Same as ChatGPT
            start_idx = max(0, idx - window_size)
            context_rows = data_sorted.iloc[start_idx:idx]
            
            prior_context = []
            for _, prow in context_rows.iterrows():
                # Fallback for prior context text (same as ChatGPT)
                pc_text = prow.get(text_col, None)
                if pc_text is None or (isinstance(pc_text, float) and pd.isna(pc_text)):
                    pc_text = prow.get('speech', '')
                
                prior_context.append({
                    'Prev_Time': str(prow['start_time']),
                    'Prev_Speech': str(prow.get('speech', ''))[:160] + "...",
                    'Prev_Text': str(pc_text)[:120] + "...",
                    'Prev_Similarity': float(prow.get('average_similarity_score', 0)) if 'average_similarity_score' in prow else None
                })
            
            # Current row fallback (same as ChatGPT)
            curr_text = row.get(text_col, None)
            if curr_text is None or (isinstance(curr_text, float) and pd.isna(curr_text)):
                curr_text = row.get('speech', '')
            
            stmt_ctx = row.get('extracted_statement_text_5', None)
            if stmt_ctx is None or (isinstance(stmt_ctx, float) and pd.isna(stmt_ctx)):
                stmt_ctx = row.get('speech', '')
            
            # Create record exactly like ChatGPT
            record = {
                'Id': int(row['id']),
                'Current_Minute_Speech': str(row.get('speech', ''))[:200] + "...",
                'News_Context': str(curr_text)[:150] + "...",
                'Statement_Context': str(stmt_ctx)[:150] + "...",
                'Current_Time': row['start_time'],
                'Prior_Context': prior_context
            }
            
            # Add training targets if this is training data
            if is_training:
                record.update({
                    'Actual_Price_Change_Percent': row.get('price_change_percent', 0.0),
                    'Price_Movement': 'Positive' if row.get('price_change_percent', 0) > 0 else 'Negative'
                })
            
            records.append(record)
        
        return records
    
    def setup_model(self, num_labels=2):
        """Initialize RoBERTa model and tokenizer"""
        print(f"Setting up RoBERTa model: {self.model_name}")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        config = RobertaConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, 
            config=config
        )
        
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
    
    def train_model(self, train_records, val_records, epochs=1, batch_size=4):
        """Train the model with ChatGPT-style data"""
        print("Starting model training...")
        
        # In smoke test, reduce parameters
        if self.smoke_test:
            self.max_length = min(self.max_length, 128)
            epochs = 1
            batch_size = min(batch_size, 2)
            print(f"SMOKE_TEST: max_length={self.max_length}, epochs={epochs}, batch_size={batch_size}")
        
        # Create datasets
        train_dataset = FinancialDataset(train_records, self.tokenizer, self.max_length)
        val_dataset = FinancialDataset(val_records, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['direction'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Validation
            val_accuracy = self.evaluate_model(val_loader)
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        print("Training completed!")
    
    def evaluate_model(self, data_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['direction'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        self.model.train()
        return accuracy
    
    def predict_batch(self, test_records):
        """Make predictions exactly like ChatGPT approach"""
        print("Making ChatGPT-style predictions...")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for record in test_records:
                # Combine text like in dataset
                current_speech = record.get('Current_Minute_Speech', '')
                news_context = record.get('News_Context', '')
                statement_context = record.get('Statement_Context', '')
                
                prior_context = record.get('Prior_Context', [])
                prior_text = ""
                if prior_context:
                    for i, ctx in enumerate(prior_context[-3:]):
                        prior_text += f"Prev_{i+1}: {ctx.get('Prev_Speech', '')} "
                
                combined_text = f"Current Speech: {current_speech} "
                combined_text += f"News Context: {news_context} "
                combined_text += f"Statement Context: {statement_context} "
                combined_text += f"Prior Context: {prior_text}"
                
                # Tokenize and predict
                encoding = self.tokenizer(
                    combined_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                direction_pred = torch.argmax(outputs.logits, dim=1).item()
                price_change_pred = 0.001 if direction_pred == 1 else -0.001
                
                predictions.append({
                    'Record_ID': record['Id'],
                    'Market_Direction_Prediction': 'Positive' if direction_pred == 1 else 'Negative',
                    'ChatGPT_Predicted_Price_Change_Percent': price_change_pred * 100
                })
        
        return predictions

def predict_multi_interval_price_change_chatgpt(train_interval: str, prediction_intervals: list, 
                                               price_change_threshold: float, num_statement_matches: str, 
                                               num_news_matches: str, smoke_limit_test_rows: int | None = None):
    """
    Train on one interval and predict for multiple different intervals using BERT with ChatGPT-style logic.
    EXACTLY matches ChatGPT implementation approach.
    
    KEY: Train on train_interval data, predict on different interval data (cross-interval approach)
    """
    all_results = {}
    
    # Initialize predictor
    predictor = BERTPredictor()
    
    # Load training data - use the ACTUAL interval data (same as ChatGPT)
    try:
        if train_interval == "1":
            train_data = pd.read_csv('1Min_data/combined_filtered_statements_and_news.csv')
        else:
            train_data = pd.read_csv(f'{train_interval}Min_data/combined_filtered_statements_and_news.csv')
        print(f"✅ Using {train_interval}Min data for training (cross-interval approach)")
    except Exception as e:
        logging.error(f"Error loading training dataset for interval {train_interval}: {e}")
        return None
    
    # Prepare training data (same as ChatGPT)
    train_data['start_time'] = pd.to_datetime(train_data['start_time'], errors='coerce')
    train_data['price_change_percent'] = train_data['price_change'] * 100
    price_change_threshold_percent = price_change_threshold * 100
    train_data['price_movement'] = np.where(train_data['price_change_percent'] > price_change_threshold_percent, 'Positive', 'Negative')
    train_data = train_data.astype({"id": int})
    
    # Filter training data by date range (same as ChatGPT)
    train_data_filtered = train_data[(train_data['start_time'] >= '2021-01-01') & (train_data['start_time'] <= '2023-12-31')]
    
    # Select best columns (same logic as ChatGPT)
    def select_best_text_column(df: pd.DataFrame, base: str) -> str:
        candidates = [f'{base}_5', f'{base}_3', f'{base}_1'] if base.startswith('extracted') else [base]
        existing = [(c, df[c].isna().mean()) for c in candidates if c in df.columns]
        if not existing:
            return 'speech'
        existing.sort(key=lambda x: (x[1], candidates.index(x[0]) if x[0] in candidates else 99))
        return existing[0][0]
    
    statement_column = select_best_text_column(train_data, 'extracted_statement_text') if num_statement_matches.startswith('extracted') else num_statement_matches
    news_column = select_best_text_column(train_data, 'extracted_news') if num_news_matches.startswith('extracted') else num_news_matches
    
    print(f"Training data after filtering: {len(train_data_filtered)} rows")
    
    # Create training prompts (same as ChatGPT)
    train_prompt_statement = predictor.create_prompt(train_data_filtered, statement_column, is_training=True)
    train_prompt_news = predictor.create_prompt(train_data_filtered, news_column, is_training=True)
    
    # Setup model
    predictor.setup_model()
    
    # Split training data for validation (same as ChatGPT)
    train_split_statement, val_split_statement = train_test_split(
        train_prompt_statement, test_size=0.2, random_state=42
    )
    train_split_news, val_split_news = train_test_split(
        train_prompt_news, test_size=0.2, random_state=42
    )
    
    # Train model (combine statement and news data)
    combined_train = train_split_statement + train_split_news
    combined_val = val_split_statement + val_split_news
    
    predictor.train_model(combined_train, combined_val)
    
    # Process each prediction interval (same as ChatGPT - CROSS-INTERVAL APPROACH)
    for pred_interval in prediction_intervals:
        print(f"🔄 CROSS-INTERVAL: Training on {train_interval}Min data, predicting for {pred_interval}Min using BERT...")
        
        try:
            # Load prediction data - use the ACTUAL interval data (same as ChatGPT)
            if pred_interval == "1":
                pred_data = pd.read_csv('1Min_data/combined_filtered_statements_and_news.csv')
            else:
                pred_data = pd.read_csv(f'{pred_interval}Min_data/combined_filtered_statements_and_news.csv')
            print(f"✅ Using {pred_interval}Min data for prediction (cross-interval: {train_interval}Min → {pred_interval}Min)")
        except Exception as e:
            logging.error(f"Error loading prediction dataset for interval {pred_interval}: {e}")
            continue
        
        # Prepare prediction data (same as ChatGPT)
        pred_data['start_time'] = pd.to_datetime(pred_data['start_time'], errors='coerce')
        pred_data['price_change_percent'] = pred_data['price_change'] * 100
        pred_data['price_movement'] = np.where(pred_data['price_change_percent'] > price_change_threshold_percent, 'Positive', 'Negative')
        pred_data = pred_data.astype({"id": int})
        
        # Filter prediction data by date range (test period) - same as ChatGPT
        pred_data_filtered = pred_data[pred_data['start_time'] >= '2024-01-01']
        
        # Select best columns for prediction data (same as ChatGPT)
        pred_stmt_col = select_best_text_column(pred_data, 'extracted_statement_text')
        pred_news_col = select_best_text_column(pred_data, 'extracted_news')
        
        # Optional smoke-test limit (same as ChatGPT)
        if smoke_limit_test_rows is not None and smoke_limit_test_rows > 0:
            pred_data_filtered = pred_data_filtered.sort_values('start_time').head(smoke_limit_test_rows)
        print(f"Prediction data for {pred_interval}Min after basic filtering: {len(pred_data_filtered)} rows")
        
        # Warm-up logic: skip the first timestamp per date in test set (same as ChatGPT)
        pred_sorted = pred_data_filtered.sort_values('start_time').reset_index(drop=False).rename(columns={'index': 'orig_index'})
        pred_sorted['date'] = pred_sorted['start_time'].dt.date
        first_per_date_idx = pred_sorted.groupby('date', as_index=False).head(1).index
        warmup_rows = pred_sorted.loc[first_per_date_idx].copy()
        test_rows = pred_sorted.drop(index=first_per_date_idx).copy()
        
        print(f"Warmup rows: {len(warmup_rows)}, Test rows: {len(test_rows)}")
        
        if len(test_rows) == 0:
            print(f"No test data available for {pred_interval}Min after warmup")
            continue
        
        # Create test prompts (same as ChatGPT)
        test_prompt_statement = predictor.create_prompt(test_rows, pred_stmt_col, is_training=False)
        test_prompt_news = predictor.create_prompt(test_rows, pred_news_col, is_training=False)
        
        # Combine prompts (same as ChatGPT)
        combined_test_prompts = test_prompt_statement + test_prompt_news
        
        # Make predictions (same as ChatGPT)
        all_predictions = predictor.predict_batch(combined_test_prompts)
        
        # Calculate metrics (same as ChatGPT)
        if all_predictions:
            results_df = pd.DataFrame(all_predictions).drop_duplicates(subset=['Record_ID'], keep='first')
            actuals = test_rows['price_movement'].tolist()
            predictions = results_df.set_index('Record_ID').reindex(test_rows['id'])['Market_Direction_Prediction'].fillna("No Prediction").tolist()
            
            valid_indices = [i for i, pred in enumerate(predictions) if pred != "No Prediction"]
            actuals_filtered = [actuals[i] for i in valid_indices]
            predictions_filtered = [predictions[i] for i in valid_indices]
            
            if predictions_filtered:
                # Direction accuracy metrics (same as ChatGPT)
                direction_metrics = {
                    "Direction_Accuracy": accuracy_score(actuals_filtered, predictions_filtered),
                    "Total_Predictions": len(all_predictions),
                    "Valid_Predictions": len(predictions_filtered)
                }
                
                # Price change accuracy metrics (same as ChatGPT)
                actual_price_changes = test_rows['price_change_percent'].tolist()
                predicted_price_changes = results_df['ChatGPT_Predicted_Price_Change_Percent'].tolist()
                
                if len(actual_price_changes) == len(predicted_price_changes):
                    price_mae = mean_absolute_error(actual_price_changes, predicted_price_changes)
                    price_mse = mean_squared_error(actual_price_changes, predicted_price_changes)
                    direction_metrics.update({
                        "Price_Change_MAE": price_mae,
                        "Price_Change_MSE": price_mse
                    })
                
                metrics = direction_metrics
            else:
                metrics = {
                    "Direction_Accuracy": 0,
                    "Total_Predictions": len(all_predictions),
                    "Valid_Predictions": 0,
                    "Price_Change_MAE": 0,
                    "Price_Change_MSE": 0
                }
            
            # Store results (same format as ChatGPT)
            all_results[f"{train_interval}Min_train_{pred_interval}Min_pred"] = {
                'results_df': results_df,
                'metrics': metrics,
                'train_interval': train_interval,
                'pred_interval': pred_interval
            }
            
            print(f"Completed training on {train_interval}Min, predicting for {pred_interval}Min.")
            print(f"Direction Accuracy: {metrics['Direction_Accuracy']:.4f}")
            print(f"Price Change MAE: {metrics.get('Price_Change_MAE', 0):.6f}%")
        else:
            print(f"No predictions generated for {pred_interval}Min")
    
    return all_results

def main():
    """Run BERT predictions with ChatGPT-style logic for all intervals"""
    print("🤖 Starting ChatGPT-Style BERT-powered predictions...")
    
    # Define intervals (same as ChatGPT)
    intervals = ["1", "5", "10", "15", "20", "25", "30"]
    
    # Run predictions for each training interval predicting all future intervals (same as ChatGPT)
    test_combinations = [
        ("1", intervals),   # Train on 1Min, predict all intervals
        ("5", intervals),   # Train on 5Min, predict all intervals
        ("10", intervals),  # Train on 10Min, predict all intervals
        ("15", intervals),  # Train on 15Min, predict all intervals
        ("20", intervals),  # Train on 20Min, predict all intervals
        ("25", intervals),  # Train on 25Min, predict all intervals
        ("30", intervals)   # Train on 30Min, predict all intervals
    ]
    
    all_results = {}
    
    for train_interval, pred_intervals in test_combinations:
        print(f"\n🎯 Training on {train_interval}Min data...")
        
        try:
            results = predict_multi_interval_price_change_chatgpt(
                train_interval=train_interval,
                prediction_intervals=pred_intervals,
                price_change_threshold=0.0001,  # Same as ChatGPT
                num_statement_matches="5",
                num_news_matches="5"
            )
            
            if results:
                all_results.update(results)
                print(f"✅ Successfully completed {train_interval}Min training")
            else:
                print(f"❌ Failed to complete {train_interval}Min training")
                
        except Exception as e:
            logging.error(f"Error in main execution for {train_interval}Min: {e}")
            continue
    
    # Save summary (same format as ChatGPT)
    if all_results:
        summary_data = []
        for key, result in all_results.items():
            summary_data.append({
                'Configuration': key,
                'Training_Interval_Minutes': result['train_interval'],
                'Prediction_Interval_Minutes': result['pred_interval'],
                'Direction_Accuracy': result['metrics']['Direction_Accuracy'],
                'Price_Change_MAE': result['metrics'].get('Price_Change_MAE', 0),
                'Price_Change_MSE': result['metrics'].get('Price_Change_MSE', 0),
                'Total_Predictions': result['metrics']['Total_Predictions'],
                'Valid_Predictions': result['metrics']['Valid_Predictions']
            })
        
        summary_df = pd.DataFrame(summary_data)
        os.makedirs("chatgpt_predictions", exist_ok=True)
        summary_df.to_csv("chatgpt_predictions/enhanced_chatgpt_summary_with_price_metrics.csv", index=False)
        
        print(f"\n🎉 Enhanced ChatGPT predictions completed!")
        print(f"📊 Total configurations: {len(all_results)}")
        print(f"📈 Average direction accuracy: {summary_df['Direction_Accuracy'].mean():.4f}")
        print(f"📉 Average price change MAE: {summary_df['Price_Change_MAE'].mean():.6f}%")
        print(f"📋 Enhanced summary saved to: chatgpt_predictions/enhanced_chatgpt_summary_with_price_metrics.csv")
    else:
        print("❌ No ChatGPT predictions were generated")

if __name__ == "__main__":
    main()
