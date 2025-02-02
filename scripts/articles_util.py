ARTICLES_BASE_PATH = "data_10Min/articles/"
STATEMENTS_BASE_PATH = "data_10Min/statements/"
# STATEMENTS_ANALYSIS_TIME_PERIOD = 15 * 60 # considering time period of 15 mins for stock price open to close percentage change

def read_file_content(file, encoding):
    with open(file, encoding = encoding, mode = 'r') as f:
        content = f.readlines()
    return " ".join(content)