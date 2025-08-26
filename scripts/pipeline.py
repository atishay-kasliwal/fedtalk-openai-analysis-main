import os
import media_util
import pandas as pd
import finance_util
import datetime
import analysis_util
import articles_util

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import sent_tokenize
import logging
import datetime

import matplotlib.pyplot as plt
from collections import Counter
import db_util
import nltk

# nltk.download('punkt_tab')

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
            return None  # or a default value
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

# def get_speech_and_price_data():
#     # In UTC
#     # start_ts = ["2024-01-31 19:30:00"]
#     start_ts = [ "2023-02-01 18:30:00", "2023-03-22 18:30:00", "2023-05-03 18:30:00", 
#               "2023-06-14 18:30:00", "2023-07-26 18:30:00",
#               "2023-09-20 18:30:00", "2023-11-01 18:30:00",
#               "2023-12-13 18:30:00", "2022-01-26 18:30:00",
#               "2022-03-16 18:30:00", 
#               "2022-05-04 18:30:00", "2022-06-15 18:30:00", 
#               "2022-07-27 18:30:00",
#               "2022-09-21 18:30:00", "2022-11-02 18:30:00", 
#               "2022-12-14 18:30:00",
#               "2021-01-27 18:30:00", "2021-03-17 18:30:00",
#               "2024-01-31 18:30:00", "2024-03-20 18:30:00", 
#               "2024-05-01 18:30:00",
#               "2024-06-12 18:30:00","2024-07-31 18:30:00",
#               "2024-09-18 18:30:00","2024-11-07 18:30:00",
#               "2024-12-18 18:30:00", 
#               "2021-04-28 18:30:00", "2021-06-16 18:30:00", 
#               "2021-07-28 18:30:00",
#               "2021-09-22 18:30:00", "2021-11-03 18:30:00", 
#               "2021-12-15 18:30:00"]
#     combined_data = pd.DataFrame()
#     for ts in start_ts:
#         file = ts[:10]
#         file_data = pd.DataFrame()
#         audio_folder = media_util.AUDIO_BASE_PATH + file
#         if not os.path.exists(audio_folder):
#             os.makedirs(audio_folder)
#         media_util.split_video(file)
#         video_partitions_path = media_util.VIDEO_BASE_PATH + media_util.PARTITIONS_SUBDIRECTORY_NAME + "/" + file
#         for filename in os.listdir(video_partitions_path):
#             video_file_path = video_partitions_path + "/" + filename
#             audio_file_name = filename.split(".")[0] + ".wav"
#             audio_file_path = audio_folder + "/" + audio_file_name
#             media_util.extract_audio_from_video(video_file_path, audio_file_path)
#             speech_text = media_util.extract_speech(audio_file_path)
#             if not speech_text:
#                 continue
#             speech_seconds = filename.split(".")[0].split("-")
#             timestamp = datetime.datetime.strptime(ts, finance_util.DATETIME_FORMAT)
#             start_time = timestamp + datetime.timedelta(seconds = int(speech_seconds[0]))
#             end_time = timestamp + datetime.timedelta(seconds = int(speech_seconds[1]))
#             print(file, filename, start_time, end_time)
#             price_change = finance_util.get_price_change(start_time, end_time)
#             volatility = finance_util.get_price_volatility(start_time, end_time)
#             file_data = pd.concat([file_data, pd.DataFrame([[ts, start_time, end_time, speech_text, price_change, volatility]], columns = ['original_time', 'start_time', 'end_time', 'speech', 'price_change', 'volatility'], index = [len(file_data)])])
#         file_data.sort_values(by = 'start_time').to_csv(f'{ANALYSIS_PATH}/{file}.csv', index = False)
#         combined_data = pd.concat([combined_data, file_data])
#     combined_data.sort_values(by = 'start_time').to_csv(f'data_1Min/combined.csv', index = False)





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


import pandas as pd
import numpy as np
import datetime

import pandas as pd
import numpy as np
import datetime

# def compare_speech_with_statements_and_news(num_matching_statement_sentences, num_matching_news_sentences):
#     print("Loading data...")
#     # Load the dataset and parse date columns
#     data = pd.read_csv('data_1Min/combined.csv', parse_dates=['start_time', 'original_time', 'end_time'])
#     print("Data loaded successfully.")

#     # Add a unique identifier for each row
#     data['id'] = range(len(data))
    
#     # Extract the date from the start_time column for further processing
#     data['start_date'] = data['start_time'].dt.date
    
#     print("Defining thresholds for price changes...")
#     # Define thresholds for price change percentages (absolute value)
#     data['threshold_0.1perc'] = data['price_change'].abs() >= 0.001
#     data['threshold_0.25perc'] = data['price_change'].abs() >= 0.0025
#     data['threshold_0.5perc'] = data['price_change'].abs() >= 0.005
#     data['threshold_0.75perc'] = data['price_change'].abs() >= 0.0075
#     print("Thresholds defined successfully.")

#     print("Calculating price changes for the same minute...")
#     # Calculate the price change for the same minute (60 seconds before original_time to original_time)
#     data['statement_price_change'] = data.apply(
#         lambda row: finance_util.get_price_change(row['original_time'] - datetime.timedelta(seconds=60), 
#                                                   row['original_time']), axis=1)
#     print("Same-minute price changes calculated.")

#     print("Calculating price changes for the next minute...")
#     # Calculate the price change for the next minute (original_time to 60 seconds after original_time)
#     data['next_minute_price_change'] = data.apply(
#         lambda row: finance_util.get_price_change(row['original_time'], 
#                                                   row['original_time'] + datetime.timedelta(seconds=60)), axis=1)
#     print("Next-minute price changes calculated.")

#     print("Determining price movements (positive/negative) for both intervals...")
#     # Determine whether the price movement for the same minute is positive or negative
#     data['statement_price_movement'] = np.where(data['statement_price_change'] > 0, 'Positive', 'Negative')
    
#     # Determine whether the price movement for the next minute is positive or negative
#     data['next_minute_price_movement'] = np.where(data['next_minute_price_change'] > 0, 'Positive', 'Negative')
#     print("Price movements determined successfully.")

#     print("Processing extracted statements and news matches...")
#     # Process extracted statements and news matches based on user-defined parameters
#     for index in range(len(num_matching_statement_sentences)): 
#         print(f"Extracting matching statements for {num_matching_statement_sentences[index]} sentences...")
#         # Extract matching statements from the database using db_util.query()
#         data[['extracted_statement_text_' + str(num_matching_statement_sentences[index]), 
#               'extracted_statement_text_' + str(num_matching_statement_sentences[index]) + '_score']] = \
#             data.apply(lambda row: db_util.query(index_name="statement", date=row['start_date'], 
#                                                  query_text=row['speech'], num_matches=num_matching_statement_sentences[index]), 
#                        axis=1, result_type='expand')
#         print(f"Matching statements extracted for {num_matching_statement_sentences[index]} sentences.")
        
#         print(f"Extracting matching news articles for {num_matching_news_sentences[index]} sentences...")
#         # Extract matching news articles from the database using db_util.query()
#         data[['extracted_news_' + str(num_matching_news_sentences[index]), 
#               'extracted_news_' + str(num_matching_news_sentences[index]) + '_score']] = \
#             data.apply(lambda row: db_util.query(index_name="news", date=row['start_date'], 
#                                                  query_text=row['speech'], num_matches=num_matching_news_sentences[index]), 
#                        axis=1, result_type='expand')
#         print(f"Matching news articles extracted for {num_matching_news_sentences[index]} sentences.")
    
#     print("Filtering positive and negative movements...")
#     # Filter rows where the same-minute price movement is positive or negative
#     same_minute_positive = data[data['statement_price_movement'] == 'Positive']
#     same_minute_negative = data[data['statement_price_movement'] == 'Negative']
    
#     # Filter rows where the next-minute price movement is positive or negative
#     next_minute_positive = data[data['next_minute_price_movement'] == 'Positive']
#     next_minute_negative = data[data['next_minute_price_movement'] == 'Negative']
    
#     print("Saving filtered results to CSV files...")
#     # Save filtered results to CSV files for further analysis or reporting
#     same_minute_positive.sort_values(by='start_time').to_csv('data_1Min/combined_filtered_same_minute_positive.csv', index=False)
#     same_minute_negative.sort_values(by='start_time').to_csv('data_1Min/combined_filtered_same_minute_negative.csv', index=False)
    
#     next_minute_positive.sort_values(by='start_time').to_csv('data_1Min/combined_filtered_next_minute_positive.csv', index=False)
#     next_minute_negative.sort_values(by='start_time').to_csv('data_1Min/combined_filtered_next_minute_negative.csv', index=False)
    
#     print("Filtered results saved successfully.")


# def compare_speech_with_statements_and_news(num_matching_statement_sentences, num_matching_news_sentences):
#     data = pd.read_csv('data_1Min/combined.csv', parse_dates = ['start_time', 'original_time', 'end_time'])
#     # data = data[data['price_change'].abs() >= THRESHOLD]
#     data['id'] = range(len(data))
#     data['start_date'] = data['start_time'].dt.date
#     data['threshold_0.1perc'] = data['price_change'].abs() >= 0.001
#     data['threshold_0.25perc'] = data['price_change'].abs() >= 0.0025
#     data['threshold_0.5perc'] = data['price_change'].abs() >= 0.005
#     data['threshold_0.75perc'] = data['price_change'].abs() >= 0.0075
#     data['statement_price_change'] = data.apply(lambda row: finance_util.get_price_change(row['original_time'] - datetime.timedelta(seconds = 60),
#                                                                                       row['original_time']), axis = 1)
#     data['statement_price_movement'] = np.where(data['statement_price_change'] > 0, 'Positive', 'Negative')
#     data['interval_price_movement'] = np.where(data['price_change'] > 0, 'Positive', 'Negative')
#     print(2)
#     for index in range(len(num_matching_statement_sentences)): 
#         data[['extracted_statement_text_' + str(num_matching_statement_sentences[index]), 'extracted_statement_text_' + str(num_matching_statement_sentences[index]) + '_score']] = data.apply(lambda row: db_util.query(index_name = "statement", date = row['start_date'], 
#                                                                                 query_text = row['speech'], num_matches = num_matching_statement_sentences[index]), axis = 1, result_type='expand')
#         print(4)
#         data[['extracted_news_' + str(num_matching_news_sentences[index]), 'extracted_news_' + str(num_matching_news_sentences[index]) + '_score']] = data.apply(lambda row: db_util.query(index_name = "news", date = row['start_date'], 
#                                                                                 query_text = row['speech'], num_matches = num_matching_news_sentences[index]), axis = 1, result_type='expand')
#         print(6)

#     # data['similarities_and_differences_with_statement_30_matching_sentences'] = data.apply(lambda row: analysis_util.get_similar_and_different_terms(row['speech'], row['extracted_statement_text_' + str(num_matching_statement_sentences[-1])]), axis = 1)
#     # data[['Topic', 'Keywords']] = pd.DataFrame(data['text'].apply(analysis_util.get_topic_and_keywords).tolist(), index = data.index)
#     # data[['Sentiment', 'Strength']] = pd.DataFrame(data['text'].apply(analysis_util.rate_news).tolist(), index = data.index)
#     # data['sentiment_comparison_with_statement'] =  data.apply(lambda row: analysis_util.compare_sentiment_and_certainity_level(row['speech'], row['extracted_statement_text_' + str(num_matching_statement_sentences[-1])]), axis = 1)
#     # data['similarities_and_differences_with_news_35_matching_sentences'] = data.apply(lambda row: analysis_util.get_similar_and_different_terms(row['speech'], row['extracted_news_' + str(num_matching_news_sentences[-1])]), axis = 1)
#     # data['sentiment_comparison_with_news'] =  data.apply(lambda row: analysis_util.compare_sentiment_and_certainity_level(row['speech'], row['extracted_news_' + str(num_matching_news_sentences[-1])]), axis = 1)
#     data_positive_statement = data[data['statement_price_movement'] == 'Positive']
#     data_negative_statement = data[data['statement_price_movement'] == 'Negative']
#     print(3)
#     data_positive_statement.sort_values(by = 'start_time').to_csv(f'data_1Min/combined_filtered_positive.csv', index = False)
#     data_negative_statement.sort_values(by = 'start_time').to_csv(f'data_1Min/combined_filtered_negative.csv', index = False)

def compare_speech_with_statements_and_news(num_matching_statement_sentences, num_matching_news_sentences):
    data = pd.read_csv('data_1Min/combined.csv', parse_dates=['start_time', 'original_time', 'end_time'])
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

    data.sort_values(by='start_time').to_csv('data_1Min/combined_filtered.csv', index=False)
    print(3)


import pandas as pd
import numpy as np
import datetime
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def predict_next_minute_price_change(interval: str, statement_type: str, test_size: float, 
                                      price_change_threshold: float, num_statement_matches: str, 
                                      num_news_matches: str):
    try:
        data = pd.read_csv(f'data_1Min/combined_filtered.csv')
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')
    data['next_minute_time'] = data['start_time'] + datetime.timedelta(minutes=1)
    
    # Create a binary label (Positive/Negative) based on the next-minute price change
    data['price_movement'] = np.where(data['price_change'] > price_change_threshold, 'Positive', 'Negative')
    data = data.astype({"id": int})
    
    X = data['id'].tolist()
    y = data['price_movement'].tolist()
    _, _ , _, _ = train_test_split(X, y, test_size=test_size, stratify=y)
    
    train_data = data[(data['start_time'] >= '2021-01-01') & (data['start_time'] <= '2023-12-31')]
    test_data = data[data['next_minute_time'] >= '2024-01-01']
    pd.set_option('display.max_columns', None)
    
    statement_column = f'extracted_statement_text_{num_statement_matches}'
    news_column = f'extracted_news_{num_news_matches}'
    
    if statement_column not in data.columns or news_column not in data.columns:
        print(f"Missing required columns: {statement_column}, {news_column}")
        return
    
    train_prompt_statement = [{
        'Id': row['id'],
        'Average Similarity Score': row[statement_column],
        'Price Movement': row['price_movement'],
        'Percent Change': row['price_change']
    } for _, row in train_data.iterrows()]
    
    train_prompt_news = [{
        'Id': row['id'],
        'Average Similarity Score': row[news_column],
        'Price Movement': row['price_movement'],
        'Percent Change': row['price_change']
    } for _, row in train_data.iterrows()]
    
    test_prompt_statement = [{
        'Id': row['id'],
        'Average Similarity Score': row[statement_column],
        'Percent Change': row['price_change']
    } for _, row in test_data.iterrows()]
    
    test_prompt_news = [{
        'Id': row['id'],
        'Average Similarity Score': row[news_column],
        'Percent Change': row['price_change']
    } for _, row in test_data.iterrows()]
    
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
                combined_reaction = aggregate_reactions(pred['Market Reaction'])
                combined_insight = create_insight(pred['Market Reaction'])
                s1 = create_s1(pred['Market Reaction'])
                s2 = create_s2(pred['Market Reaction'])
                s3 = create_s3(pred['Market Reaction'])
                p = create_op3(pred['Market Reaction'])
                all_predictions_combined.append({
                    'Id': _id,
                    'Combined_Prediction': combined_reaction,
                    'Combined_Insight': combined_insight,
                    'Similarity 1': s1,
                    'Similarity 2': s2,
                    'Similarity 3': s3,
                    'Price Change': p
                })
        except Exception as e:
            logging.error(f"Error processing a batch for statements and news: {e}")
    
    results_df = pd.DataFrame(all_predictions_combined)
    results_df = results_df.drop_duplicates(subset=['Id'], keep='first')
    
    # # Group only numerical columns and compute mean
    # numeric_cols = ["Similarity 1", "Similarity 2", "Similarity 3"]
    # results_df[numeric_cols] = results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # grouped_df = results_df.groupby("Id")[numeric_cols].mean().reset_index()

    # # Handle Reaction separately (use mode to keep the most common reaction for each Id)
    # reaction_df = results_df.groupby("Id")["Market Reaction"].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown").reset_index()

    # # Merge both results
    # final_results = grouped_df.merge(reaction_df, on="Id", how="left")

    # print(final_results)
    
    actuals = test_data['price_movement'].tolist()
    predictions = results_df.set_index('Id').reindex(test_data['id'])['Combined_Prediction'].fillna("No Prediction").tolist()
    
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
        
        # Calculate confusion matrix to derive sensitivity and specificity
        conf_matrix = confusion_matrix(actuals_filtered, predictions_filtered, labels=['Positive', 'Negative'])
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.update({
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })
    else:
        metrics = {"Accuracy": 0, "F1_Score": 0, "Precision": 0, "Recall": 0, "Sensitivity": 0, "Specificity": 0}
    
    output_dir = "data_1Min/analysis"
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, "positive_same_time_negative_predictions_insights_minu55.csv")
    results_df.to_csv(predictions_file, index=False)
    logging.info(f"Next-minute predictions saved to {predictions_file}")
    
    metrics_file = os.path.join(output_dir, "positive_same_time_negative_metrics_summary_minut55.csv")
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    logging.info(f"Metrics summary saved to {metrics_file}")
    
    return results_df, metrics







# # # Interval-Level Text and Similarity Score Extraction and Analysis
statement_matches = [25,30,35,40,45,50,55]
news_matches = [25, 30, 35, 40,45,50,55]


# compare_speech_with_statements_and_news(statement_matches, news_matches)

# # # Price Movement Prediction using Text
# # Example usage

def predict_next_minute_price_change(interval: str, test_size: float, 
                                      price_change_threshold: float, num_statement_matches: str, 
                                      num_news_matches: str):
    try:
        data = pd.read_csv('data_1Min/combined_filtered.csv')
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

    output_dir = "data_1Min/analysis"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "combined_predictions45.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "combined_metrics_summary45.csv"), index=False)
    
    return results_df, metrics

predict_next_minute_price_change(
    interval="1Min",
    test_size=0.2,
    price_change_threshold=0.3,
    num_statement_matches="45",
    num_news_matches="45")
