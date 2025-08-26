ARTICLES_BASE_PATH = "data_1Min/articles/"
STATEMENTS_BASE_PATH = "data_1Min/statements/"
# STATEMENTS_ANALYSIS_TIME_PERIOD = 15 * 60 # considering time period of 15 mins for stock price open to close percentage change

def read_file_content(file, encoding):
    with open(file, encoding = encoding, mode = 'r') as f:
        content = f.readlines()
    return " ".join(content)







import pandas as pd
def csv():
    df1 = pd.read_csv("Positive_Prediction.csv")

# Load the second CSV (Source of additional columns)
    df2 = pd.read_csv("filtered_positive.csv", encoding="ISO-8859-1", low_memory=False)


    df1['id'] = df1['id'].astype(str)  # Convert to string
    df2['id'] = df2['id'].astype(str)  # Convert to string
    merged_df = df1.merge(df2, on='id', how='inner')

# Merge on a common column (e.g., 'id')
    merged_df = df1.merge(df2[['id', 'price_change']], on='id', how='left')

# Save the merged CSV
    ans  = merged_df.to_csv("Appendix_B_combined_filtered_positive.csv", index=False)

    print("CSV files merged successfully!")
    return ans

csv()
