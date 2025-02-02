import os
import media_util
import pandas as pd
import finance_util
import datetime
import analysis_util
import articles_util
import db_util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import sent_tokenize
import json
import logging
import datetime
import math
import math
import matplotlib.pyplot as plt
import ast
from collections import Counter
import nltk

# nltk.download('punkt_tab')

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    filename="pipeline.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

ANALYSIS_PATH = "data_1Min/analysis"
THRESHOLD = 0.001
STATEMENTS_THRESHOLD = 0
# file_names = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01",
#               "2023-12-13", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14", "2021-01-27", "2021-03-17",
#                 "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15"]
# file_names = ["2024Jan", "2023Feb"]
ARTICLE_FILES_ENCODING = 'utf8'
STATEMENTS_FILES_ENCODING = 'utf8'
# file_names = ["2024Jan"]

def get_speech_and_price_data():
    # In UTC
    # start_ts = ["2024-01-31 19:30:00"]
    start_ts = ["2024-01-31 18:30:00", "2022-01-26 19:30:00"]
    combined_data = pd.DataFrame()
    for ts in start_ts:
        file = ts[:10]
        file_data = pd.DataFrame()
        audio_folder = media_util.AUDIO_BASE_PATH + file
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
        media_util.split_video(file)
        video_partitions_path = media_util.VIDEO_BASE_PATH + media_util.PARTITIONS_SUBDIRECTORY_NAME + "/" + file
        for filename in os.listdir(video_partitions_path):
            video_file_path = video_partitions_path + "/" + filename
            audio_file_name = filename.split(".")[0] + ".wav"
            audio_file_path = audio_folder + "/" + audio_file_name
            media_util.extract_audio_from_video(video_file_path, audio_file_path)
            speech_text = media_util.extract_speech(audio_file_path)
            if not speech_text:
                continue
            speech_seconds = filename.split(".")[0].split("-")
            timestamp = datetime.datetime.strptime(ts, finance_util.DATETIME_FORMAT)
            start_time = timestamp + datetime.timedelta(seconds = int(speech_seconds[0]))
            end_time = timestamp + datetime.timedelta(seconds = int(speech_seconds[1]))
            print(file, filename, start_time, end_time)
            price_change = finance_util.get_price_change(start_time, end_time)
            volatility = finance_util.get_price_volatility(start_time, end_time)
            file_data = pd.concat([file_data, pd.DataFrame([[ts, start_time, end_time, speech_text, price_change, volatility]], columns = ['original_time', 'start_time', 'end_time', 'speech', 'price_change', 'volatility'], index = [len(file_data)])])
        file_data.sort_values(by = 'start_time').to_csv(f'{ANALYSIS_PATH}/{file}.csv', index = False)
        combined_data = pd.concat([combined_data, file_data])
    combined_data.sort_values(by = 'start_time').to_csv(f'data_1Min/combined.csv', index = False)

# def store_in_db(type):
#     db_util.init_db(index_name = type)
#     if type == "statement":
#         base_dir = articles_util.STATEMENTS_BASE_PATH
#     else:
#         base_dir = articles_util.ARTICLES_BASE_PATH + "news/"
#     for filename in os.listdir(base_dir):
#         if not filename.endswith(".txt"):
#             continue
#         date = filename.split(".")[0]
#         with open(base_dir + filename, encoding = 'utf-8', mode = 'r') as f:
#             content = f.readlines()
#         f.close()
#         text = " ".join(content)
#         text_sentences = sent_tokenize(text, language = 'english')
#         sentence_data = pd.DataFrame({'id': [i for i in range(len(text_sentences))], 'sentence': text_sentences})
#         sentence_data_file = base_dir + date + '.csv'
#         sentence_data.to_csv(sentence_data_file, index = False)
#         db_util.upsert_data(index_name = type, namespace = date, data_file = sentence_data_file)

def compare_speech_with_statements_and_news(num_matching_statement_sentences, num_matching_news_sentences):
    logging.info("Starting compare_speech_with_statements_and_news function")

    # Load dataset
    try:
        data = pd.read_csv('data_10Min/combined.csv', parse_dates=['start_time', 'original_time', 'end_time'])
        logging.info(f"Loaded dataset with {len(data)} rows")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # Add unique ID and new columns
    data['id'] = range(len(data))
    data['start_date'] = data['start_time'].dt.date
    data['threshold_0.1perc'] = data['price_change'].abs() >= 0.001
    data['threshold_0.25perc'] = data['price_change'].abs() >= 0.0025
    data['threshold_0.5perc'] = data['price_change'].abs() >= 0.005
    data['threshold_0.75perc'] = data['price_change'].abs() >= 0.0075

    logging.info("Added calculated threshold columns")

    # Compute statement price change
    try:
        data['statement_price_change'] = data.apply(
            lambda row: finance_util.get_price_change(
                row['original_time'] - datetime.timedelta(seconds=30*60),
                row['original_time'] - datetime.timedelta(seconds=25*60)
            ), axis=1
        )
        logging.info("Calculated statement_price_change")
    except Exception as e:
        logging.error(f"Error computing statement_price_change: {e}")

    data['statement_price_movement'] = np.where(data['statement_price_change'] > 0, 'Positive', 'Negative')
    data['interval_price_movement'] = np.where(data['price_change'] > 0, 'Positive', 'Negative')

    # Debugging lengths of input lists
    logging.info(f"num_matching_statement_sentences: {num_matching_statement_sentences}")
    logging.info(f"num_matching_news_sentences: {num_matching_news_sentences}")

    # Ensure input lists are not empty
    if not num_matching_statement_sentences or not num_matching_news_sentences:
        logging.error("num_matching_statement_sentences and num_matching_news_sentences cannot be empty")
        return

    # Loop through matching sentence counts
    for index in range(len(num_matching_statement_sentences)):
        try:
            logging.info(f"Processing statement matches: {num_matching_statement_sentences[index]}")
            
            # Extract statements
            data[
                [f"extracted_statement_text_{num_matching_statement_sentences[index]}", 
                 f"extracted_statement_text_{num_matching_statement_sentences[index]}_score"]
            ] = data.apply(
                lambda row: pd.Series(
                    db_util.query(
                        index_name="statement", 
                        date=row['start_date'], 
                        query_text=row['speech'], 
                        num_matches=num_matching_statement_sentences[index]
                    )
                ), 
                axis=1
            )
            
            logging.info(f"Extracted statement matches for {num_matching_statement_sentences[index]}")
        
        except Exception as e:
            logging.error(f"Error processing statement matches {num_matching_statement_sentences[index]}: {e}")

        try:
            logging.info(f"Processing news matches: {num_matching_news_sentences[index]}")

            # Extract news
            data[
                [f"extracted_news_{num_matching_news_sentences[index]}", 
                 f"extracted_news_{num_matching_news_sentences[index]}_score"]
            ] = data.apply(
                lambda row: pd.Series(
                    db_util.query(
                        index_name="news", 
                        date=row['start_date'], 
                        query_text=row['speech'], 
                        num_matches=num_matching_news_sentences[index]
                    )
                ), 
                axis=1
            )

            logging.info(f"Extracted news matches for {num_matching_news_sentences[index]}")
        
        except Exception as e:
            logging.error(f"Error processing news matches {num_matching_news_sentences[index]}: {e}")

    # Compute similarities and differences
    try:
        data['similarities_and_differences_with_statement_30_matching_sentences'] = data.apply(
            lambda row: analysis_util.get_similar_and_different_terms(
                row['speech'], 
                row[f'extracted_statement_text_{num_matching_statement_sentences[-1]}']
            ), axis=1
        )
        logging.info("Computed similarities_and_differences_with_statement_30_matching_sentences")
    except Exception as e:
        logging.error(f"Error computing similarities with statements: {e}")

    try:
        data['similarities_and_differences_with_news_35_matching_sentences'] = data.apply(
            lambda row: analysis_util.get_similar_and_different_terms(
                row['speech'], 
                row[f'extracted_news_{num_matching_news_sentences[-1]}']
            ), axis=1
        )
        logging.info("Computed similarities_and_differences_with_news_35_matching_sentences")
    except Exception as e:
        logging.error(f"Error computing similarities with news: {e}")

    # Split data based on price movement
    try:
        data_positive_statement = data[data['statement_price_movement'] == 'Positive']
        data_negative_statement = data[data['statement_price_movement'] == 'Negative']

        # Save filtered data
        data_positive_statement.sort_values(by='start_time').to_csv('data_10Min/combined_filtered_positive.csv', index=False)
        data_negative_statement.sort_values(by='start_time').to_csv('data_10Min/combined_filtered_negative.csv', index=False)

        logging.info("Saved filtered data files")

    except Exception as e:
        logging.error(f"Error saving filtered data: {e}")

    logging.info("Finished compare_speech_with_statements_and_news function")


import logging
import os
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

  

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict_price_change_using_score(interval: str, statement_type: str, test_size: float, 
                                     price_change_threshold: float, num_statement_matches: str, 
                                     num_news_matches: str):
    # Load the data
    try:
        data = pd.read_csv(f'data_{interval}/combined_filtered_{statement_type}.csv')
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Create a binary label (Positive/Negative) based on the price change threshold
    data['price_movement'] = np.where(data['price_change'] > price_change_threshold, 'Positive', 'Negative')
    data = data.astype({"id": int})
    
    # (Optional) Train/test split by ID if needed later in the pipeline
    X = data['id'].tolist()
    y = data['price_movement'].tolist()
    _, _ , _, _ = train_test_split(X, y, test_size=test_size, stratify=y)
    
    # Convert the 'original_time' column to datetime
    data['original_time'] = pd.to_datetime(data['original_time'], errors='coerce')
    
    # Filter training and testing data by date range
    train_data = data[(data['original_time'] >= '2021-01-01') & (data['original_time'] <= '2023-12-31')]
    test_data = data[data['original_time'] >= '2024-01-01']
    pd.set_option('display.max_columns', None)
    
    # Define column names for statement and news based on the given match numbers
    statement_column = f'extracted_statement_text_{num_statement_matches}'
    news_column = f'extracted_news_{num_news_matches}'
    if statement_column not in data.columns or news_column not in data.columns:
        print(f"Missing required columns: {statement_column}, {news_column}")
        return

    # Prepare prompt dictionaries for training and testing
    train_prompt_statement = [{
        'Id': row['id'],
        'Average Similarity Score': row[statement_column],
        'Price Movement': row['price_movement']
    } for _, row in train_data.iterrows()]
    
    train_prompt_news = [{
        'Id': row['id'],
        'Average Similarity Score': row[news_column],
        'Price Movement': row['price_movement']
    } for _, row in train_data.iterrows()]
    
    test_prompt_statement = [{
        'Id': row['id'],
        'Average Similarity Score': row[statement_column]
    } for _, row in test_data.iterrows()]
    
    test_prompt_news = [{
        'Id': row['id'],
        'Average Similarity Score': row[news_column]
    } for _, row in test_data.iterrows()]
    
    # Helper: Chunk a list into batches
    def chunk_data(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]
    
    batch_size = 10  
    train_batches_statement = list(chunk_data(train_prompt_statement, batch_size))
    test_batches_statement = list(chunk_data(test_prompt_statement, batch_size))
    train_batches_news = list(chunk_data(train_prompt_news, batch_size))
    test_batches_news = list(chunk_data(test_prompt_news, batch_size))
    
    # Helper: Aggregate market reactions via majority vote
    def aggregate_reactions(market_reactions):
        reactions = [item.get('Reaction') for item in market_reactions if item.get('Reaction') is not None]
        if not reactions:
            return None  # or a default value
        most_common = Counter(reactions).most_common(1)[0][0]
        return most_common
    
    # Helper: Create a detailed insight per test ID by joining explanation texts
    def create_insight(market_reactions):
        explanations = []
        for item in market_reactions:
            # Concatenate time and explanation if available
            if 'Time' in item and 'Explanation' in item:
                explanations.append(f"{item['Time']}: {item['Explanation']}")
        return " | ".join(explanations)
    
    # Process batches for statement-based predictions and insights
    all_predictions_statement = []
    for train_batch, test_batch in zip(train_batches_statement, test_batches_statement):
        try:
            predictions, _ = analysis_util.get_market_reaction_predictions(train_batch, test_batch)
            all_predictions_statement.extend(predictions)
        except Exception as e:
            logging.error(f"Error processing a batch for statements: {e}")
    
    # Process batches for news-based predictions and insights
    all_predictions_news = []
    for train_batch, test_batch in zip(train_batches_news, test_batches_news):
        try:
            predictions, _ = analysis_util.get_market_reaction_predictions(train_batch, test_batch)
            all_predictions_news.extend(predictions)
        except Exception as e:
            logging.error(f"Error processing a batch for news: {e}")
    
    # Build dictionaries mapping test ID to actual outcome
    test_data_dict = test_data.set_index('id')['price_movement'].to_dict()
    
    # Build per-ID prediction and insight dictionaries for statements
    predictions_statement_dict = {}
    insights_statement_dict = {}
    for pred in all_predictions_statement:
        if isinstance(pred, dict):
            _id = pred.get('Id')
            if _id in test_data_dict:
                predictions_statement_dict[_id] = aggregate_reactions(pred['Market Reaction'])
                insights_statement_dict[_id] = create_insight(pred['Market Reaction'])
    
    # Build per-ID prediction and insight dictionaries for news
    predictions_news_dict = {}
    insights_news_dict = {}
    for pred in all_predictions_news:
        if isinstance(pred, dict):
            _id = pred.get('Id')
            if _id in test_data_dict:
                predictions_news_dict[_id] = aggregate_reactions(pred['Market Reaction'])
                insights_news_dict[_id] = create_insight(pred['Market Reaction'])
    
    # Prepare lists to accumulate per-ID results for the CSV
    results_list = []
    pred_y_statement = []
    actual_y_statement = []
    pred_y_news = []
    actual_y_news = []
    
    for _, row in test_data.iterrows():
        _id = row['id']
        actual = row['price_movement']
        # time_str = row['original_time'].strftime("%H:%M:%S") if pd.notnull(row['original_time']) else "N/A"
        
        # Get statement predictions and insights for this ID (if available)
        stmt_pred = predictions_statement_dict.get(_id, "No Prediction")
        stmt_insight = insights_statement_dict.get(_id, "No Insight")
        # Get news predictions and insights for this ID (if available)
        news_pred = predictions_news_dict.get(_id, "No Prediction")
        news_insight = insights_news_dict.get(_id, "No Insight")
        
        # Append to the results list
        results_list.append({
            "ID": _id,
            
            "Statement_Prediction": stmt_pred,
            "Statement_Actual": actual,
            "Statement_Insight": stmt_insight,
            "News_Prediction": news_pred,
            "News_Actual": actual,
            "News_Insight": news_insight
        })
        
        # Collect values for metric calculation
        if stmt_pred != "No Prediction":
            pred_y_statement.append(stmt_pred)
            actual_y_statement.append(actual)
        if news_pred != "No Prediction":
            pred_y_news.append(news_pred)
            actual_y_news.append(actual)
    
    # Compute evaluation metrics for both statement and news predictions
    def compute_metrics(actual, predicted):
        if not predicted:
            return 0, 0, 0, 0
        return (
            accuracy_score(actual, predicted),
            f1_score(actual, predicted, average='weighted', zero_division=0),
            precision_score(actual, predicted, average='weighted', zero_division=0),
            recall_score(actual, predicted, average='weighted', zero_division=0)
        )
    
    metrics_statement = compute_metrics(actual_y_statement, pred_y_statement)
    metrics_news = compute_metrics(actual_y_news, pred_y_news)
    
    # Create DataFrame for per-ID predictions and insights
    results_df = pd.DataFrame(results_list, columns=["ID", 
                                                     "Statement_Prediction", "Statement_Actual", "Statement_Insight", 
                                                     "News_Prediction", "News_Actual", "News_Insight"])
    
    # Create DataFrame for overall metrics (rows for Statement and News models)
    metrics_data = [
        {
            "Model": "Statement",
            "Accuracy": metrics_statement[0],
            "F1_Score": metrics_statement[1],
            "Precision": metrics_statement[2],
            "Recall": metrics_statement[3]
        },
        {
            "Model": "News",
            "Accuracy": metrics_news[0],
            "F1_Score": metrics_news[1],
            "Precision": metrics_news[2],
            "Recall": metrics_news[3]
        }
    ]
    metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Accuracy", "F1_Score", "Precision", "Recall"])
    
    # Create output directory if it does not exist
    output_dir = "data_5Min/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the detailed per-ID predictions and insights CSV
    predictions_file = os.path.join(output_dir, "predictions_insights.csv")
    results_df.to_csv(predictions_file, index=False)
    logging.info(f"Predictions and insights saved to {predictions_file}")
    
    # Save the overall metrics CSV
    metrics_file = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logging.info(f"Metrics summary saved to {metrics_file}")
    
    # Optionally, print a short summary report on screen
    print("Detailed predictions and insights have been saved to:", predictions_file)
    print("Overall metrics summary has been saved to:", metrics_file)
    
    return results_df, metrics_df




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# # def predict_price_change_using_text(statement_type: str, test_size: int, price_change_threshold: int, num_statement_matches: str, num_news_matches: str):
# #     # Load the data
# #     data = pd.read_csv(f'data_10Min/combined_filtered_{statement_type}.csv')
    
# #     # Create the 'price_movement' column based on the threshold
# #     data['price_movement'] = np.where(data['price_change'] > price_change_threshold, 'Positive', 'Negative')
    
# #     # Convert 'id' to int for consistency
# #     data = data.astype({"id": int})
    
# #     # Prepare features and labels
# #     X = data['id'].tolist()
# #     y = data['price_movement'].tolist()
    
# #     # Split the data into training and testing sets
# #     X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, stratify=y)

# #     print("Training IDs:", X_train)
# #     print("Testing IDs:", X_test)
# #     print(data.columns)

# #     # Convert 'original_time' to datetime
# #     data['original_time'] = pd.to_datetime(data['original_time'], errors='coerce')

# #     # Filter data into training and testing sets
# #     train_data = data[(data['original_time'] >= '2021-01-01') & (data['original_time'] <= '2023-12-31')]
# #     test_data = data[data['original_time'] >= '2024-01-01']

# #     # Prepare the prompts for statements and news
# #     train_prompt_statement, train_prompt_news = [], []
# #     for _, row in train_data.iterrows():
# #         train_prompt_statement.append({
# #             'Id': row['id'],
# #             'First Text': row['speech'],
# #             'Second Text': row.get(f'extracted_statement_text_{num_statement_matches}', ''),
# #             'Price Movement': row['price_movement']
# #         })
# #         train_prompt_news.append({
# #             'Id': row['id'],
# #             'First Text': row['speech'],
# #             'Second Text': row.get(f'extracted_news_{num_news_matches}', ''),
# #             'Price Movement': row['price_movement']
# #         })
    
# #     test_prompt_statement, test_prompt_news = [], []
# #     for _, row in test_data.iterrows():
# #         test_prompt_statement.append({
# #             'Id': row['id'],
# #             'First Text': row['speech'],
# #             'Second Text': row.get(f'extracted_statement_text_{num_statement_matches}', '')
# #         })
# #         test_prompt_news.append({
# #             'Id': row['id'],
# #             'First Text': row['speech'],
# #             'Second Text': row.get(f'extracted_news_{num_news_matches}', '')
# #         })
    
# #     # Fetch predictions using statements and news
# #     try:
# #         predictions_statement, insights_statement = analysis_util.get_price_change_predictions_using_statements(train_prompt_statement, test_prompt_statement)
# #     except Exception as e:
# #         print(f"Error getting predictions using statements: {e}")
# #         predictions_statement, insights_statement = [], []

# #     try:
# #         predictions_news, insights_news = analysis_util.get_price_change_predictions_using_news(train_prompt_news, test_prompt_news)
# #     except Exception as e:
# #         print(f"Error getting predictions using news: {e}")
# #         predictions_news, insights_news = [], []

# #     # Initialize lists to store results
# #     pred_y_statement, pred_y_news = [], []
# #     actual_y_statement, actual_y_news = [], []

# #     # Set the index of test data to 'id' for easier access
# #     if 'id' in test_data.columns:
# #         test_data.set_index('id', inplace=True)

# #     # Process statement predictions
# #     for pred in predictions_statement:
# #         if isinstance(pred, dict) and 'Id' in pred and 'Market Reaction' in pred:
# #             try:
# #                 # Extract the final timepoint reaction
# #                 final_reaction = pred['Market Reaction'][-1]['Reaction']
# #                 pred_y_statement.append(final_reaction)
# #                 actual_y_statement.append(test_data.at[pred['Id'], 'price_movement'])
# #             except Exception as e:
# #                 print(f"Error processing statement prediction {pred}: {e}")
# #         else:
# #             print(f"Invalid prediction format for statement: {pred}")

# #     # Process news predictions
# #     for pred in predictions_news:
# #         if isinstance(pred, dict) and 'Id' in pred and 'Market Reaction' in pred:
# #             try:
# #                 # Extract the final timepoint reaction
# #                 final_reaction = pred['Market Reaction'][-1]['Reaction']
# #                 pred_y_news.append(final_reaction)
# #                 actual_y_news.append(test_data.at[pred['Id'], 'price_movement'])
# #             except Exception as e:
# #                 print(f"Error processing news prediction {pred}: {e}")
# #         else:
# #             print(f"Invalid prediction format for news: {pred}")

# #     # Calculate performance metrics for statements
# #     accuracy_statement = accuracy_score(actual_y_statement, pred_y_statement) if pred_y_statement else 0
# #     f1_statement = f1_score(actual_y_statement, pred_y_statement, average='weighted') if pred_y_statement else 0
# #     precision_statement = precision_score(actual_y_statement, pred_y_statement, average='weighted', zero_division=0) if pred_y_statement else 0
# #     recall_statement = recall_score(actual_y_statement, pred_y_statement, average='weighted', zero_division=0) if pred_y_statement else 0

# #     # Calculate performance metrics for news
# #     accuracy_news = accuracy_score(actual_y_news, pred_y_news) if pred_y_news else 0
# #     f1_news = f1_score(actual_y_news, pred_y_news, average='weighted') if pred_y_news else 0
# #     precision_news = precision_score(actual_y_news, pred_y_news, average='weighted', zero_division=0) if pred_y_news else 0
# #     recall_news = recall_score(actual_y_news, pred_y_news, average='weighted', zero_division=0) if pred_y_news else 0

# #     # Store results
# #     results = [
# #         [
# #             "Statement Text", statement_type, test_size, num_statement_matches, 
# #             str(accuracy_statement),
# #             str(f1_statement),
# #             str(precision_statement),
# #             str(recall_statement),
# #             insights_statement
# #         ],
# #         [
# #             "News Text", statement_type, test_size, num_news_matches,
# #             str(accuracy_news),
# #             str(f1_news),
# #             str(precision_news),
# #             str(recall_news),
# #             insights_news
# #         ]
# #     ]

# #     return results


# # Configure logging to print to console and optionally to a file
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



# # def predict_price_change_using_score(interval: str, statement_type: str, test_size: float, price_change_threshold: float, num_statement_matches: str, num_news_matches: str):
# #     logging.info(f"Loading data for interval: {interval}, statement_type: {statement_type}")
    
# #     # Load the dataset
# #     try:
# #         data = pd.read_csv(f'data_{interval}/combined_filtered_{statement_type}.csv')
# #     except Exception as e:
# #         logging.error(f"Error loading dataset: {e}")
# #         return
    
# #     # Define price movement
# #     data['price_movement'] = np.where(data['price_change'] > price_change_threshold, 'Positive', 'Negative')
# #     data = data.astype({"id": int})
    
# #     # Split the data into training and testing sets
# #     X = data['id'].tolist()
# #     y = data['price_movement'].tolist()
# #     X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, stratify=y)
    
# #     logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
# #     # Convert 'original_time' to datetime
# #     data['original_time'] = pd.to_datetime(data['original_time'], errors='coerce')
    
# #     # Filter training and testing data
# #     train_data = data[(data['original_time'] >= '2021-01-01') & (data['original_time'] <= '2023-12-31')]
# #     test_data = data[data['original_time'] >= '2024-01-01']
    
# #     # Validate required columns
# #     statement_column = f'extracted_statement_text_{num_statement_matches}_score'
# #     news_column = f'extracted_news_{num_news_matches}_score'
    
# #     if statement_column not in data.columns or news_column not in data.columns:
# #         logging.error(f"Missing required columns: {statement_column}, {news_column}")
# #         return
    
# #     # Create training and testing prompts
# #     train_prompt_statement = [{'Id': row['id'], 'Average Similarity Score': row[statement_column], 'price_movement': row['price_movement']} for _, row in train_data.iterrows()]
# #     train_prompt_news = [{'Id': row['id'], 'Average Similarity Score': row[news_column], 'price_movement': row['price_movement']} for _, row in train_data.iterrows()]
# #     test_prompt_statement = [{'Id': row['id'], 'Average Similarity Score': row[statement_column]} for _, row in test_data.iterrows()]
# #     test_prompt_news = [{'Id': row['id'], 'Average Similarity Score': row[news_column]} for _, row in test_data.iterrows()]
    
# #     # Get predictions
# #     try:
# #         predictions_statement, insights_statement = analysis_util.get_market_reaction_predictions(train_prompt_statement, test_prompt_statement)
# #         predictions_news, insights_news = analysis_util.get_market_reaction_predictions(train_prompt_news, test_prompt_news)
# #     except Exception as e:
# #         logging.error(f"Error in prediction generation: {e}")
# #         return
    
# #     # Convert test_data into a dictionary for quick lookup
# #     test_data_dict = test_data.set_index('id')['price_movement'].to_dict()
    
# #     # Extract predictions
# #     pred_y_statement = [pred['price_movement'] for pred in predictions_statement if isinstance(pred, dict) and pred['Id'] in test_data_dict]
# #     actual_y_statement = [test_data_dict[pred['Id']] for pred in predictions_statement if isinstance(pred, dict) and pred['Id'] in test_data_dict]
# #     pred_y_news = [pred['price_movement'] for pred in predictions_news if isinstance(pred, dict) and pred['Id'] in test_data_dict]
# #     actual_y_news = [test_data_dict[pred['Id']] for pred in predictions_news if isinstance(pred, dict) and pred['Id'] in test_data_dict]
    
# #     # Compute performance metrics
# #     def compute_metrics(actual, predicted):
# #         if not predicted:
# #             return 0, 0, 0, 0  # Return zero if no predictions were made
# #         return (
# #             accuracy_score(actual, predicted),
# #             f1_score(actual, predicted, average='weighted', zero_division=0),
# #             precision_score(actual, predicted, average='weighted', zero_division=0),
# #             recall_score(actual, predicted, average='weighted', zero_division=0)
# #         )
    
# #     metrics_statement = compute_metrics(actual_y_statement, pred_y_statement)
# #     metrics_news = compute_metrics(actual_y_news, pred_y_news)
    
# #     # Store results
# #     results = [
# #         ["Statement Similarity Score", statement_type, test_size, num_statement_matches, *map(str, metrics_statement), insights_statement],
# #         ["News Similarity Score", statement_type, test_size, num_news_matches, *map(str, metrics_news), insights_news]
# #     ]
    
# #     return results

# def predict_price_volatility_using_score(statement_type: str, test_size: int, price_threshold: int, num_statement_matches: str, num_news_matches: str):
#     data = pd.read_csv('/combined_filtdata_5Minered_' + statement_type + '.csv', parse_dates = ['start_time', 'end_time'])
#     data['volatility'] = data.apply(lambda row: finance_util.get_price_volatility(row['start_time'], row['end_time']), axis = 1)
#     data['price_volatility'] = np.where(data['volatility'] > price_threshold, 'High', 'Low')
#     data = data.astype({"id": int})
#     X = data['id'].tolist()
#     y = data['price_volatility'].tolist()
#     X_train, X_test, _, _ = train_test_split(X, y, test_size = test_size, stratify = y)
#     print(X_train)
#     print(X_test)
#    # Ensure the column is in DateTime format
#     data['original_time'] = pd.to_datetime(data['original_time'], infer_datetime_format=True, errors='coerce')

# # Filter data correctly
#     train_data = data[(data['original_time'] >= '2021-01-01') & (data['original_time'] <= '2023-12-31')]
#     test_data = data[data['original_time'] >= '2024-01-01']

    

#     train_prompt_statement = list()
#     train_prompt_news = list()
#     for _, row in train_data.iterrows():
#         train_prompt_statement.append({'Id':  row['id'], 'Average Similarity Score': row['extracted_statement_text_' + num_statement_matches + '_score'], 'Price Volatility': row['price_volatility']})
#         train_prompt_news.append({'Id':  row['id'], 'Average Similarity Score': row['extracted_news_' + num_news_matches + '_score'], 'Price Volatility': row['price_volatility']})
#     test_prompt_statement = list()
#     test_prompt_news = list()
#     for _, row in test_data.iterrows():
#         test_prompt_statement.append({'Id':  row['id'], 'Average Similarity Score': row['extracted_statement_text_' + num_statement_matches + '_score']})
#         test_prompt_news.append({'Id':  row['id'], 'Average Similarity Score': row['extracted_news_' + num_news_matches + '_score']})
#     predictions_statement, insights_statement = analysis_util.get_market_reaction_predictions(train_prompt_statement, test_prompt_statement)
#     predictions_news, insights_news = analysis_util.get_market_reaction_predictions(train_prompt_news, test_prompt_news)
#     # print("Statement: ", predictions_statement)
#     # print("News :", predictions_news)
#     # statement_results_file_name = f'predictions_using_statement_score_{num_statement_matches}_matches_test_size_{test_size}_threshold_{price_change_threshold}.txt'
#     # news_results_file_name = f'predictions_using_news_score_{num_news_matches}_matches_test_size_{test_size}_threshold_{price_change_threshold}.txt'
#     pred_y_statement = list()
#     pred_y_news = list()
#     actual_y_statement = list()
#     actual_y_news = list()
#     test_data.set_index('id', inplace = True)
#     for pred in predictions_statement:
#         pred_y_statement.append(pred['Price Volatility'])
#         actual_y_statement.append(test_data.at[pred['Id'], 'price_volatility'])
#     for pred in predictions_news:
#         pred_y_news.append(pred['Price Volatility'])
#         actual_y_news.append(test_data.at[pred['Id'], 'price_volatility'])
#     results.append(["Statement Similarity Score", statement_type, test_size, num_statement_matches, 
#                     str(accuracy_score(actual_y_statement, pred_y_statement)),
#                     str(f1_score(actual_y_statement, pred_y_statement, pos_label = 'weighted')),
#                     str(precision_score(actual_y_statement, pred_y_statement, pos_label = 'weighted')),
#                     str(recall_score(actual_y_statement, pred_y_statement, pos_label = 'weighted')),
#                     insights_statement])
#     results.append(["News Similarity Score", statement_type, test_size, num_news_matches,
#                     str(accuracy_score(actual_y_news, pred_y_news)),
#                     str(f1_score(actual_y_news, pred_y_news, pos_label = 'weighted')),
#                     str(precision_score(actual_y_news, pred_y_news, pos_label = 'weighted')),
#                     str(recall_score(actual_y_news, pred_y_news, pos_label = 'weighted')),
#                     insights_news])
#     # with open(media_util.RESULTS_BASE_PATH + statement_results_file_name, 'w') as f:
#     #     f.write(str(predictions_statement) + "\n")
#     #     f.write("Accuracy: " + str(accuracy_score(actual_y_statement, pred_y_statement)) + "\n")
#     #     f.write("F1 Score: " + str(f1_score(actual_y_statement, pred_y_statement, pos_label = 'Positive')) + "\n")
#     #     f.write("Precision: " + str(precision_score(actual_y_statement, pred_y_statement, pos_label = 'Positive')) + "\n")
#     #     f.write("Recall: " + str(recall_score(actual_y_statement, pred_y_statement, pos_label = 'Positive')) + "\n")
#     # f.close()
#     # with open(media_util.RESULTS_BASE_PATH + news_results_file_name, 'w') as f:
#     #     f.write(str(predictions_news) + "\n")
#     #     f.write("Accuracy: " + str(accuracy_score(actual_y_news, pred_y_news)) + "\n")
#     #     f.write("F1 Score: " + str(f1_score(actual_y_news, pred_y_news, pos_label = 'Positive')) + "\n")
#     #     f.write("Precision: " + str(precision_score(actual_y_news, pred_y_news, pos_label = 'Positive')) + "\n")
#     #     f.write("Recall: " + str(recall_score(actual_y_news, pred_y_news, pos_label = 'Positive')) + "\n")
#     # f.close()

# def calc_volatility_measures(interval: str, statement_type: str):
#     data_statement = pd.read_csv('data_' + interval + '/combined_filtered_' + statement_type + '.csv', parse_dates = ['start_time', 'end_time'])
#     if 'volatility' not in data_statement.columns:
#         print(interval)
#         data_statement['volatility'] = data_statement.apply(lambda row: finance_util.get_price_volatility(row['start_time'], row['end_time']), axis = 1)
#     data_statement_positive_intervals = data_statement[data_statement['price_change'] >= 0]
#     data_statement_negative_intervals = data_statement[data_statement['price_change'] < 0]
#     data_statement_positive_intervals['volatility_squared'] = data_statement_positive_intervals.apply(lambda row: row['volatility']**2, axis = 1)
#     data_statement_negative_intervals['volatility_squared'] = data_statement_negative_intervals.apply(lambda row: row['volatility']**2, axis = 1)
#     volatility_measure_for_positive = math.sqrt(data_statement_positive_intervals['volatility_squared'].sum())
#     volatility_measure_for_negative = math.sqrt(data_statement_negative_intervals['volatility_squared'].sum())
#     return volatility_measure_for_positive, volatility_measure_for_negative



# def calc_price_movement_distribution(interval: str, statement_type: str):
#     # Read data
#     data = pd.read_csv(f'data_{interval}/combined_filtered_{statement_type}.csv')

#     # Convert date formats
#     data['original_time'] = pd.to_datetime(data['original_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
#     data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
#     data['end_time'] = pd.to_datetime(data['end_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

#     # Extract unique and sorted values
#     original_times = sorted(data['original_time'].dropna().unique())
#     dates = sorted(data['start_date'].dropna().unique())
#     end_times = sorted(data[['start_date', 'end_time']].groupby(by=['start_date']).max()['end_time'].dropna().unique())

#     # Convert to formatted strings
#     original_times = [t.strftime('%Y-%m-%d %H:%M') for t in original_times]
#     dates = [d.strftime('%Y-%m-%d') for d in dates]
#     end_times = [e.strftime('%Y-%m-%d %H:%M') for e in end_times]

#     print(original_times)
#     print(dates)
#     print(end_times)

#     volatility_df = pd.DataFrame()

#     for i in range(len(dates)):
#         # Read price data
#         price_file = f'data_{interval}/price/{dates[i]}.csv'
#         try:
#             price_data = pd.read_csv(price_file, parse_dates=['timestamp'])
#         except FileNotFoundError:
#             print(f"Warning: Price file {price_file} not found. Skipping...")
#             continue

#         # Ensure 'calculated_volatility' column exists
#         if 'calculated_volatility' not in price_data.columns:
#             price_data['calculated_volatility'] = price_data.apply(finance_util.calculate_volatility, axis=1)

#         # Convert timestamps to datetime
#         price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], errors='coerce')

#         # Convert start_time and end_time
#         start_time = datetime.datetime.strptime(original_times[i], '%Y-%m-%d %H:%M') - datetime.timedelta(hours=1)
#         end_time = datetime.datetime.strptime(end_times[i], '%Y-%m-%d %H:%M')

#         # Filter price data within the time range
#         price_data = price_data[(price_data['timestamp'] >= start_time) & (price_data['timestamp'] <= end_time)]

#         if price_data.empty:
#             print(f"Warning: No data for date {dates[i]}. Skipping...")
#             continue

#         # Compute relative time
#         price_data['relative_time'] = (price_data['timestamp'] - datetime.datetime.strptime(original_times[i], '%Y-%m-%d %H:%M')).dt.total_seconds() / 60.0

#         # Compute volatility squared
#         price_data['volatility_squared'] = price_data['calculated_volatility'].fillna(0).apply(lambda x: x * x)

#         # Append to volatility_df
#         volatility_df = pd.concat([volatility_df, price_data[['relative_time', 'volatility_squared']]])

#     # Compute standard deviation of volatility
#     if not volatility_df.empty:
#         volatility_df = volatility_df.groupby(by='relative_time').sum().reset_index()
#         volatility_df['volatility_std'] = volatility_df['volatility_squared'].apply(lambda x: math.sqrt(x))

#         # Save results
#         volatility_df[['relative_time', 'volatility_std']].to_csv(f'data_{interval}/volatility_{statement_type}.csv', index=False)
#     else:
#         print("No volatility data to save.")
# def visualize_box_plot_price_change(interval: str, statement_type: str):
#     data = pd.read_csv('data_' + interval + '/combined_filtered_' + statement_type + '.csv')
#     data_pos = data[data['interval_price_movement'] == 'Positive'][['price_change']]
#     fig_pos = plt.figure()
#     data_pos.boxplot()
#     fig_pos.savefig('data_' + interval + '/results/positive_price_change_box_plot_' + statement_type + '_statement_movement.svg', format = "svg")
#     data_neg = data[data['interval_price_movement'] == 'Negative'][['price_change']]
#     fig_neg = plt.figure()
#     data_neg.boxplot()
#     fig_neg.savefig('data_' + interval + '/results/negative_price_change_box_plot_' + statement_type + '_statement_movement.svg', format = "svg")


# def visualize_histogram_terms(interval: str, statement_type: str):

#     def split_into_list_similar(x):
#         try:
#             x = x.replace('("', '["').replace('")', '"]')
#             json_content = json.loads(x)
#             return ast.literal_eval(str(json_content.get("Similar", "[]")))
#         except (json.JSONDecodeError, ValueError, AttributeError) as e:
#             print(f"JSON parsing error: {e}")
#             return []

#     def split_into_list_different(x, index):
#         try:
#             x = x.replace('("', '["').replace('")', '"]')
#             json_content = json.loads(x)
#             different_list = json_content.get("Different", [])
#             parsed_list = ast.literal_eval(str(different_list))
#             return [ele[index] for ele in parsed_list if len(ele) > index]
#         except (json.JSONDecodeError, ValueError, AttributeError, IndexError) as e:
#             print(f"JSON parsing error: {e}")
#             return []

#     # Read the data
#     data = pd.read_csv(f'data_{interval}/combined_filtered_{statement_type}.csv')

#     for direction in ["Positive", "Negative"]:
#         for type in ["statement", "news"]:
#             data_selected = data[data['interval_price_movement'] == direction].copy()  # Avoid chained indexing

#             similar_words = []
#             different_words_baselines = []
#             different_words_compared = []

#             # Define the correct column based on type
#             column_map = {
#                 "statement": "similarities_and_differences_with_statement_30_matching_sentences",
#                 "news": "similarities_and_differences_with_news_35_matching_sentences"
#             }
#             column = column_map.get(type)

#             # Check if the column exists
#             if column not in data_selected.columns:
#                 print(f"Warning: Column '{column}' not found in data. Skipping {type} processing.")
#                 continue

#             # Apply functions safely
#             data_selected["similar"] = data_selected[column].apply(split_into_list_similar)
#             data_selected["different_baseline"] = data_selected[column].apply(lambda x: split_into_list_different(x, 0))
#             data_selected["different_compared"] = data_selected[column].apply(lambda x: split_into_list_different(x, 1))

#             # Collect words
#             for value in data_selected['similar']:
#                 similar_words.extend(value)
#             for value in data_selected['different_baseline']:
#                 different_words_baselines.extend(value)
#             for value in data_selected['different_compared']:
#                 different_words_compared.extend(value)

#             # File naming based on index
#             file_names = ["similar", "different_speech", "different_compared"]
#             words_lists = [similar_words, different_words_baselines, different_words_compared]

#             for file_name, words_list in zip(file_names, words_lists):
#                 if not words_list:  # Skip empty lists
#                     print(f"Skipping {file_name} histogram due to no data.")
#                     continue

#                 counts = dict(Counter(words_list).most_common(20))
#                 labels, values = zip(*counts.items()) if counts else ([], [])

#                 # Sorting indices
#                 ind_sort = np.argsort(values)[::-1]
#                 labels, values = np.array(labels)[ind_sort], np.array(values)[ind_sort]
#                 indexes = np.arange(len(labels))

#                 # Create plot directory if not exists
#                 plot_dir = f'data_{interval}/results/{type}/{direction} Interval & {statement_type} Statement Price Movement'
#                 os.makedirs(plot_dir, exist_ok=True)

#                 # Plot histogram
#                 plt.figure(figsize=(20, 10))
#                 plt.bar(indexes, values)
#                 plt.xticks(indexes, labels, rotation='vertical')
#                 plt.xlabel("Top 20 Terms/Phrases")
#                 plt.ylabel("Frequency")
#                 plt.title(f"{direction} Interval & {statement_type} Statement - {file_name}")
#                 plt.margins(x=0.2)
                
#                 plt.savefig(f'{plot_dir}/{direction}_interval_{statement_type}_statement_{file_name}_words_of_{type}_histogram.jpg')
#                 plt.close()

#                 # Save text file
#                 with open(f'{plot_dir}/{direction}_interval_{statement_type}_statement_{file_name}_words_of_{type}_histogram.txt', 'w', encoding='utf-8') as f:
#                     f.write(str(counts) + "\n")

# # Extract audio, speech text and get price data
get_speech_and_price_data()

# # Store statement and news data in Pinecone
# store_in_db("statement")
# store_in_db("news")

# # # Interval-Level Text and Similarity Score Extraction and Analysis
statement_matches = [15, 20, 25, 30]
news_matches = [20, 25, 30, 35]
# compare_speech_with_statements_and_news(statement_matches, news_matches)

# # # Price Movement Prediction using Text
# # Example usage
ans = predict_price_change_using_score(
    interval="5Min",
    statement_type="positive",
    test_size=0.2,
    price_change_threshold=0.3,
    num_statement_matches="15",
    num_news_matches="20"
)
predict_price_change_using_score(
    interval="5Min",
    statement_type="negativeß",
    test_size=0.2,
    price_change_threshold=0.3,
    num_statement_matches="3",
    num_news_matches="3"
)
# # Example usage
# predict_price_change_using_score(
#     interval="5Min",
#     statement_type="negative",
#     test_size=0.2,
#     price_change_threshold=0.3,
#     num_statement_matches="3",
#     num_news_matches="3"
# )
# results = list()
# intervals = ['1Min', '5Min', '10Min']

# for interval in intervals:
#     for i in range(len(statement_matches)):
#         res_positive_02 = predict_price_change_using_score(
#     interval="5Min",
#     statement_type="positive",
#     test_size=0.2,
#     price_change_threshold=0.3,
#     num_statement_matches="15",
#     num_news_matches="20"
# )
#         if res_positive_02:
#             results.extend(res_positive_02)
        
#         res_positive_033 = predict_price_change_using_score(
#     interval="5Min",
#     statement_type="positive",
#     test_size=0.2,
#     price_change_threshold=0.3,
#     num_statement_matches="15",
#     num_news_matches="20"
# )
#         if res_positive_033:
#             results.extend(res_positive_033)
        
#         res_negative_02 = predict_price_change_using_score(
#     interval="5Min",
#     statement_type="negative",
#     test_size=0.2,
#     price_change_threshold=0.3,
#     num_statement_matches="15",
#     num_news_matches="20"
# )
#         if res_negative_02:
#             results.extend(res_negative_02)
        
#         res_negative_033 = predict_price_change_using_score(
#     interval="5Min",
#     statement_type="negative",
#     test_size=0.2,
#     price_change_threshold=0.3,
#     num_statement_matches="15",
#     num_news_matches="20"
# )
#         if res_negative_033:
#             results.extend(res_negative_033)

# # # Create a DataFrame with the desired column names and save the results to a CSV file.
# results_df = pd.DataFrame(
#     results, 
#     columns=["Id","Time","Reaction","Explanation","Insights"]
# )
# results_df.to_csv("data_5Min/results/results_of_score_analysis_using_price_movement_with_statement_separation1.csv", index=False)

# # Volatility Results
# results_volatility = list()
# for interval in intervals:
#     pos, neg = calc_volatility_measures(interval, 'positive')
#     results_volatility.append([interval, 'positive', 'positive', pos])
#     results_volatility.append([interval, 'positive', 'negative', neg])
#     pos, neg = calc_volatility_measures(interval, 'negative')
#     results_volatility.append([interval, 'negative', 'positive', pos])
#     results_volatility.append([interval, 'negative', 'negative', neg])
# results_volatility_df = pd.DataFrame(results_volatility, columns = ["Interval", "Statement Price Movement", "Interval Price Movement", "Volatility (Standard Deviation)"])
# results_volatility_df.to_csv("volatility_results.csv", index = False)

# for interval in intervals:
#     calc_price_movement_distribution(interval, 'positive')
#     calc_price_movement_distribution(interval, 'negative')

# # Price Change Box Plots
# for interval in intervals:
#     visualize_box_plot_price_change(interval, 'positive')
#     visualize_box_plot_price_change(interval, 'negative')

# # Similar and Different Terms Histograms
# for interval in intervals:
#     visualize_histogram_terms(interval, 'positive')
#     visualize_histogram_terms(interval, 'negative')

# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score, classification_report
# # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # def analyze_sentiment(text):
# #     """
# #     Analyze the sentiment of a given text using VADER.

# #     Parameters:
# #     text (str): The input text to analyze.

# #     Returns:
# #     float: The compound sentiment score ranging from -1 (negative) to 1 (positive).
# #     """
# #     analyzer = SentimentIntensityAnalyzer()
# #     sentiment_score = analyzer.polarity_scores(text)['compound']
# #     return sentiment_score

# # def train_model(data):
# #     """
# #     Train a predictive model to forecast market reactions based on speech sentiment and other features.

# #     Parameters:
# #     data (pd.DataFrame): The training data containing features and target variables.

# #     Returns:
# #     model: The trained logistic regression model.
# #     vectorizer: The TF-IDF vectorizer fitted on the training data.
# #     """
# #     # Ensure the 'speech' column is of type string
# #     data['speech'] = data['speech'].astype(str)

# #     # Extract features and target variable
# #     X = data['speech']
# #     y = data['price_change']  # Assuming 'price_change' is the target variable

# #     # Split the data into training and validation sets
# #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Vectorize the speech text using TF-IDF
# #     vectorizer = TfidfVectorizer(max_features=5000)
# #     X_train_tfidf = vectorizer.fit_transform(X_train)
# #     X_val_tfidf = vectorizer.transform(X_val)

# #     # Initialize and train the logistic regression model
# #     model = LogisticRegression()
# #     model.fit(X_train_tfidf, y_train)

# #     # Evaluate the model on the validation set
# #     y_pred = model.predict(X_val_tfidf)
# #     accuracy = accuracy_score(y_val, y_pred)
# #     print(f"Validation Accuracy: {accuracy:.2f}")
# #     print("Classification Report:")
# #     print(classification_report(y_val, y_pred))

# #     return model, vectorizer

# # def test_model(model, vectorizer, test_data):
# #     """
# #     Test the trained model on new data and generate predictions.

# #     Parameters:
# #     model: The trained logistic regression model.
# #     vectorizer: The TF-IDF vectorizer fitted on the training data.
# #     test_data (pd.DataFrame): The test data containing features.

# #     Returns:
# #     list: A list of predictions.
# #     dict: Insights or evaluation metrics.
# #     """
# #     # Ensure the 'speech' column is of type string
# #     test_data['speech'] = test_data['speech'].astype(str)

# #     # Transform the test data using the fitted TF-IDF vectorizer
# #     X_test_tfidf = vectorizer.transform(test_data['speech'])

# #     # Generate predictions
# #     predictions = model.predict(X_test_tfidf)

# #     # Compile insights or evaluation metrics
# #     insights = {
# #         'predictions': predictions,
# #         'prediction_count': len(predictions),
# #         # Add more metrics or insights as needed
# #     }

# #     return predictions, insights
# # import os
# # import pandas as pd
# # import datetime
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_squared_error

# # def process_fomc_data_and_predict():
# #     # Define the timestamps for the FOMC press conferences in UTC
# #     start_ts = ["2024-03-20 18:30:00", "2022-05-04 18:30:00"]
# #     combined_data = pd.DataFrame()

# #     for ts in start_ts:
# #         file_date = ts[:10]
# #         file_data = pd.DataFrame()
# #         audio_folder = os.path.join('audio', file_date)
# #         os.makedirs(audio_folder, exist_ok=True)

# #         # Ensure the video partitions path exists
# #         video_partitions_path = os.path.join('video_partitions', file_date)
# #         if not os.path.exists(video_partitions_path):
# #             print(f"Directory {video_partitions_path} does not exist. Skipping this date.")
# #             continue

# #         # Split the video into segments
# #         media_util.split_video(file_date)

# #         for filename in os.listdir(video_partitions_path):
# #             video_file_path = os.path.join(video_partitions_path, filename)
# #             audio_file_name = f"{os.path.splitext(filename)[0]}.wav"
# #             audio_file_path = os.path.join(audio_folder, audio_file_name)

# #             # Extract audio and transcribe speech
# #             media_util.extract_audio_from_video(video_file_path, audio_file_path)
# #             speech_text = media_util.extract_speech(audio_file_path)
# #             if not speech_text:
# #                 continue

# #             # Determine the start and end times of the speech segment
# #             speech_seconds = filename.split(".")[0].split("-")
# #             timestamp = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
# #             start_time = timestamp + datetime.timedelta(seconds=int(speech_seconds[0]))
# #             end_time = timestamp + datetime.timedelta(seconds=int(speech_seconds[1]))

# #             # Retrieve market data
# #             price_change = finance_util.get_price_change(start_time, end_time)
# #             volatility = finance_util.get_price_volatility(start_time, end_time)

# #             # Compile the data
# #             file_data = pd.concat([file_data, pd.DataFrame([{
# #                 'original_time': ts,
# #                 'start_time': start_time,
# #                 'end_time': end_time,
# #                 'speech': speech_text,
# #                 'price_change': price_change,
# #                 'volatility': volatility,
# #             }])])

# #         if not file_data.empty:
# #             # Save the processed data for the specific date
# #             file_data.sort_values(by='start_time').to_csv(f'analysis/{file_date}.csv', index=False)
# #             combined_data = pd.concat([combined_data, file_data])

# #     if combined_data.empty:
# #         print("No data processed. Exiting function.")
# #         return None, None

# #     # Save the combined data
# #     combined_data.sort_values(by='start_time').to_csv('data_1Min/combined.csv', index=False)

# #     # Split into training and testing datasets
# #     combined_data['original_time'] = pd.to_datetime(combined_data['original_time'])
# #     train_data = combined_data[
# #         (combined_data['original_time'] >= '2021-01-01') & (combined_data['original_time'] <= '2023-12-31')
# #     ]
# #     test_data = combined_data[combined_data['original_time'] >= '2024-01-01']

# #     if train_data.empty or test_data.empty:
# #         print("Insufficient data for training or testing. Exiting function.")
# #         return None, None

# #     # Train the predictive model
# #     model = train_model(train_data)

# #     # Test the model and get predictions
# #     predictions, insights = test_model(model, test_data)

# #     # Save predictions to CSV
# #     predictions_df = pd.DataFrame(predictions)
# #     predictions_df.to_csv('predictions.csv', index=False)

# #     return predictions, insights

# # def train_model(train_data):
# #     # Example training function using Linear Regression
# #     features = ['price_change', 'volatility']
# #     target = 'price_change'  # Assuming we're predicting price change

# #     X = train_data[features]
# #     y = train_data[target]

# #     model = LinearRegression()
# #     model.fit(X, y)

# #     return model

# # def test_model(model, test_data):
# #     # Example testing function
# #     features = ['price_change', 'volatility']
# #     target = 'price_change'

# #     X_test = test_data[features]
# #     y_test = test_data[target]

# #     predictions = model.predict(X_test)
# #     mse = mean_squared_error(y_test, predictions)

# #     insights = {
# #         'mean_squared_error': mse,
# #         'predictions': predictions,
# #         'actuals': y_test.values
# #     }

# #     return predictions, insights

# # process_fomc_data_and_predict()




# def chunk_data(data_list, batch_size):
#     """Yield successive chunks from data_list."""
#     for i in range(0, len(data_list), batch_size):
#         yield data_list[i:i + batch_size]
