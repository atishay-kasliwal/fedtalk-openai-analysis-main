ARTICLES_BASE_PATH = "data/raw/data_1Min/articles/"
STATEMENTS_BASE_PATH = "data/raw/data_1Min/statements/"
# STATEMENTS_ANALYSIS_TIME_PERIOD = 15 * 60 # considering time period of 15 mins for stock price open to close percentage change

def read_file_content(file, encoding):
    with open(file, encoding = encoding, mode = 'r') as f:
        content = f.readlines()
    return " ".join(content)




import pandas as pd

def merge_positive_predictions(
    predictions_file="data/processed/Positive_Prediction.csv",
    filtered_file="data/processed/filtered_positive.csv",
    output_file="data/processed/Appendix_B_combined_filtered_positive.csv",
):
    """One-off script that merges LLM predictions with filtered price-change data on id."""
    df1 = pd.read_csv(predictions_file)
    df2 = pd.read_csv(filtered_file, encoding="ISO-8859-1", low_memory=False)

    df1['id'] = df1['id'].astype(str)
    df2['id'] = df2['id'].astype(str)

    merged_df = df1.merge(df2[['id', 'price_change']], on='id', how='left')
    merged_df.to_csv(output_file, index=False)

    print("CSV files merged successfully!")
    return output_file


if __name__ == "__main__":
    merge_positive_predictions()
