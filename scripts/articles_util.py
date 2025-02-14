ARTICLES_BASE_PATH = "data_1Min/articles/"
STATEMENTS_BASE_PATH = "data_1Min/statements/"
# STATEMENTS_ANALYSIS_TIME_PERIOD = 15 * 60 # considering time period of 15 mins for stock price open to close percentage change

def read_file_content(file, encoding):
    with open(file, encoding = encoding, mode = 'r') as f:
        content = f.readlines()
    return " ".join(content)







import pandas as pd
def csv():
    df1 = pd.read_csv("/Users/atishaykasliwal/Downloads/fedtalk-openai-analysis-main/data_5Min/analysis/predictions_insights_combined11.csv")

# Load the second CSV (Source of additional columns)
    df2 = pd.read_csv("/Users/atishaykasliwal/Downloads/fedtalk-openai-analysis-main/data_1Min/combined_filtered_negative.csv")

# Merge on a common column (e.g., 'id')
    merged_df = df1.merge(df2[['id', 'price_change', 'volatility', "statement_price_movement"]], on='id', how='left')
    merged_df = merged_df.drop('Combined_Actual', axis=1)

# Save the merged CSV
    ans  = merged_df.to_csv("Appendix_B_combined_filtered_positive.csv", index=False)

    print("CSV files merged successfully!")
    return ans

csv()