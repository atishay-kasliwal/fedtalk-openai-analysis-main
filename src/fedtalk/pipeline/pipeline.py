import os
import logging
import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ..utils import media_util
from ..utils import finance_util
from ..analysis import analysis_util
from ..utils import articles_util
from ..utils import db_util
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.basicConfig(
#     filename="pipeline.log",
#     filemode="w",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )
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

# Helper: Chunk a list into batches
def chunk_data(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

# Helper: Aggregate market reactions via majority vote
def aggregate_reactions(market_reactions):
    reactions = [item.get('Reaction') for item in market_reactions if item.get('Reaction') is not None]
    if not reactions:
        return None
    most_common = Counter(reactions).most_common(1)[0][0]
    return most_common

# Helper: Create a detailed insight per test ID by joining explanation texts
def create_insight(market_reactions):
    explanations = []
    for item in market_reactions:
        explanations.append(str(item['Explanation']))
    return " | ".join(explanations)

def create_s1(market_reactions):
    return " ".join(str(item['Similarity 1']) for item in market_reactions if 'Similarity 1' in item)

def create_s2(market_reactions):
    return " ".join(str(item['Similarity 2']) for item in market_reactions if 'Similarity 2' in item)

def create_s3(market_reactions):
    return " ".join(str(item['Similarity 3']) for item in market_reactions if 'Similarity 3' in item)

def create_op3(market_reactions):
    return " ".join(str(item['Percent Change']) for item in market_reactions if 'Percent Change' in item)







ANALYSIS_PATH = "data/raw/data_1Min/analysis"
THRESHOLD = 0.001
STATEMENTS_THRESHOLD = 0
file_names = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01",
              "2023-12-13", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14", "2021-01-27", "2021-03-17",
                "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15"]
file_names = ["2024Jan", "2023Feb"]
ARTICLE_FILES_ENCODING = 'utf8'
STATEMENTS_FILES_ENCODING = 'utf8'


def get_speech_and_price_data():
    # In UTC
    # start_ts = ["2024-01-31 19:30:00"]
    start_ts = [ "2023-02-01 18:30:00", "2023-03-22 18:30:00", "2023-05-03 18:30:00", 
              "2023-06-14 18:30:00", "2023-07-26 18:30:00",
              "2023-09-20 18:30:00", "2023-11-01 18:30:00",
              "2023-12-13 18:30:00", "2022-01-26 18:30:00",
              "2022-03-16 18:30:00", 
              "2022-05-04 18:30:00", "2022-06-15 18:30:00", 
              "2022-07-27 18:30:00",
              "2022-09-21 18:30:00", "2022-11-02 18:30:00", 
              "2022-12-14 18:30:00",
              "2021-01-27 18:30:00", "2021-03-17 18:30:00",
              "2024-01-31 18:30:00", "2024-03-20 18:30:00", 
              "2024-05-01 18:30:00",
              "2024-06-12 18:30:00","2024-07-31 18:30:00",
              "2024-09-18 18:30:00","2024-11-07 18:30:00",
              "2024-12-18 18:30:00", 
              "2021-04-28 18:30:00", "2021-06-16 18:30:00", 
              "2021-07-28 18:30:00",
              "2021-09-22 18:30:00", "2021-11-03 18:30:00", 
              "2021-12-15 18:30:00"]
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
    combined_data.sort_values(by = 'start_time').to_csv(f'data/raw/data_1Min/combined.csv', index = False)

# get_speech_and_price_data()



def compare_speech_with_statements_and_news(num_matching_statement_sentences, num_matching_news_sentences):
    data = pd.read_csv('data/raw/data_1Min/combined.csv', parse_dates=['start_time', 'original_time', 'end_time'])
    data['id'] = range(len(data))
    data['start_date'] = data['start_time'].dt.date
    data['threshold_0.1perc'] = data['price_change'].abs() >= 0.001
    data['threshold_0.25perc'] = data['price_change'].abs() >= 0.0025
    data['threshold_0.5perc'] = data['price_change'].abs() >= 0.005
    data['threshold_0.75perc'] = data['price_change'].abs() >= 0.0075
    data['statement_price_change'] = data.apply(lambda row: finance_util.get_price_change(row['original_time'] - datetime.timedelta(seconds=60),
                                                                                           row['original_time']), axis=1)
    data['statement_price_movement'] = np.where(data['statement_price_change'] > 0, 'Positive', 'Negative')
    data['interval_price_movement'] = np.where(data['price_change'] > 0, 'Positive', 'Negative')
    print(2)

    for index in range(len(num_matching_statement_sentences)):
        data[[f'extracted_statement_text_{num_matching_statement_sentences[index]}',
              f'extracted_statement_text_{num_matching_statement_sentences[index]}_score']] = data.apply(lambda row: db_util.query(index_name="statement", date=row['start_date'], 
                                                                                                                         query_text=row['speech'], num_matches=num_matching_statement_sentences[index]), axis=1, result_type='expand')
        print(4)

        data[[f'extracted_news_{num_matching_news_sentences[index]}',
              f'extracted_news_{num_matching_news_sentences[index]}_score']] = data.apply(lambda row: db_util.query(index_name="news", date=row['start_date'], 
                                                                                                              query_text=row['speech'], num_matches=num_matching_news_sentences[index]), axis=1, result_type='expand')
        print(6)

    data.sort_values(by='start_time').to_csv('data/raw/data_1Min/combined_filtered.csv', index=False)
    print(3)







# # # Interval-Level Text and Similarity Score Extraction and Analysis
statement_matches = [25,30,35,40,45,50,55]
# news_matches = [25, 30, 35, 40,45,50,55]
# compare_speech_with_statements_and_news(statement_matches, news_matches)

# # # Price Movement Prediction using Text


def predict_next_minute_price_change(interval: str, test_size: float, 
                                      price_change_threshold: float, num_statement_matches: str, 
                                      num_news_matches: str):
    try:
        data = pd.read_csv('data/raw/data_1Min/combined_filtered.csv')
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')
    data['next_minute_time'] = data['start_time'] + datetime.timedelta(minutes=1)
    
    data['price_movement'] = np.where(data['price_change'] > price_change_threshold, 'Positive', 'Negative')
    data = data.astype({"id": int})
    
    X = data['id'].tolist()
    y = data['price_movement'].tolist()
    train_test_split(X, y, test_size=test_size, stratify=y)
    
    train_data = data[(data['start_time'] >= '2021-01-01') & (data['start_time'] <= '2023-12-31')]
    test_data = data[data['next_minute_time'] >= '2024-01-01']
    
    statement_column = f'extracted_statement_text_{num_statement_matches}'
    news_column = f'extracted_news_{num_news_matches}'
    
    if statement_column not in data.columns or news_column not in data.columns:
        print(f"Missing required columns: {statement_column}, {news_column}")
        return
    
    def create_prompt(data_chunk, text_col):
        return [{
            'Id': row['id'],
            'Average Similarity Score': row[text_col],
            'Price Movement': row.get('price_movement', None),
            'Percent Change': row['price_change']
        } for _, row in data_chunk.iterrows()]

    train_prompt_statement = create_prompt(train_data, statement_column)
    test_prompt_statement = create_prompt(test_data, statement_column)
    train_prompt_news = create_prompt(train_data, news_column)
    test_prompt_news = create_prompt(test_data, news_column)

    def chunk_data(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]

    batch_size = 15
    train_batches_statement = list(chunk_data(train_prompt_statement, batch_size))
    test_batches_statement = list(chunk_data(test_prompt_statement, batch_size))
    train_batches_news = list(chunk_data(train_prompt_news, batch_size))
    test_batches_news = list(chunk_data(test_prompt_news, batch_size))

    all_predictions_combined = []
    for train_batch_statement, test_batch_statement, train_batch_news, test_batch_news in zip(train_batches_statement, test_batches_statement, train_batches_news, test_batches_news):
        try:
            combined_predictions, _ = analysis_util.get_market_reaction_predictions(
                train_batch_statement + train_batch_news, 
                test_batch_statement + test_batch_news
            )
            print(combined_predictions)
            for pred in combined_predictions:
                _id = pred.get('Id')
                all_predictions_combined.append({
                    'Id': _id,
                    'Combined_Prediction': aggregate_reactions(pred['Market Reaction']),
                    'Combined_Insight': create_insight(pred['Market Reaction']),
                    'Similarity 1': create_s1(pred['Market Reaction']),
                    'Similarity 2': create_s2(pred['Market Reaction']),
                    'Similarity 3': create_s3(pred['Market Reaction']),
                    'Price Change': create_op3(pred['Market Reaction'])
                })
        except Exception as e:
            logging.error(f"Error processing a batch: {e}")
    
    results_df = pd.DataFrame(all_predictions_combined).drop_duplicates(subset=['Id'], keep='first')
    actuals = test_data['price_movement'].tolist()
    predictions = results_df.set_index('Id').reindex(test_data['id'])['Combined_Prediction'].fillna("No Prediction").tolist()
    print(predictions)
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "No Prediction"]
    actuals_filtered = [actuals[i] for i in valid_indices]
    predictions_filtered = [predictions[i] for i in valid_indices]
    
    if predictions_filtered:
        metrics = {
            "Accuracy": accuracy_score(actuals_filtered, predictions_filtered),
            "F1_Score": f1_score(actuals_filtered, predictions_filtered, average='weighted', zero_division=0),
            "Precision": precision_score(actuals_filtered, predictions_filtered, average='weighted', zero_division=0),
            "Recall": recall_score(actuals_filtered, predictions_filtered, average='weighted', zero_division=0)
        }
        conf_matrix = confusion_matrix(actuals_filtered, predictions_filtered, labels=['Positive', 'Negative'])
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics.update({
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    else:
        metrics = {"Accuracy": 0, "F1_Score": 0, "Precision": 0, "Recall": 0, "Sensitivity": 0, "Specificity": 0}

    output_dir = "data/raw/data_1Min/analysis"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "combined_predictions45.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "combined_metrics_summary45.csv"), index=False)
    
    return results_df, metrics

def main():
    predict_next_minute_price_change(
        interval="1Min",
        test_size=0.2,
        price_change_threshold=0.3,
        num_statement_matches="45",
        num_news_matches="45")


if __name__ == "__main__":
    main()
