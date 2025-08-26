import requests
import datetime
import time
import pandas as pd
import os

def get_general_news(query, api_key, start_time, end_time, page=1):
    # url = f'https://financialmodelingprep.com/api/v4/general_news?page={page}&from={start_date}&to={end_date}&apikey={api_key}&query=Fed'
    # url = f'https://financialmodelingprep.com/api/v4/general_news?page={page}&from={start_date}&to={end_date}&apikey={api_key}'
    url = f'https://gnews.io/api/v4/search?q={query}&apikey={api_key}&in=title,description&from={start_time}&to={end_time}&lang=en&expand=content&page={page}&max=25'
    response = requests.get(url)
    try:
        news_data = response.json()
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return []
    return news_data

def process_and_save_news(news_data, file_name):
    articles_found = list()
    with open(file_name, 'a+', encoding='utf-8') as file:
        for article in news_data["articles"]:
            if isinstance(article, dict):
                # if datetime.datetime.strptime(article.get('publishedDate'), "%Y-%m-%dT%H:%M:%S.%fZ") >= datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - datetime.timedelta(hours=0, minutes=30):
                #     continue
                # if "fed" not in article.get('title', 'N/A').lower() and "fed" not in article.get('text', 'N/A').lower():
                #     continue
                # file.write(f"{article.get('title', 'N/A')}\n")
                # file.write(f"{article.get('publishedAt', 'N/A')}\n")
                file.write(f"{article.get('description', 'N/A')}\n")
                file.write(f"{article.get('content', 'N/A')}\n")
                # file.write(f"URL: {article.get('url', 'N/A')}\n")
                # file.write(article.get('text'))
                file.write("\n\n")
                articles_found.append([article.get('title', 'N/A'), article.get('publishedAt', 'N/A'), article.get('description', 'N/A'), article.get('content', 'N/A'), article.get('url', 'N/A')])
            else:
                print(f"Unexpected article format: {article}")
    file.close()
    return articles_found

if __name__ == "__main__":
    # start_ts = ["2024-03-20 18:30:00", "2024-05-01 18:30:00", "2024-06-12 18:30:00", "2022-01-26 19:30:00", "2022-03-16 18:30:00", "2022-05-04 18:30:00", "2022-06-15 18:30:00",
    #             "2022-07-27 18:30:00", "2022-09-21 18:30:00", "2022-11-02 18:30:00", "2022-12-14 19:30:00", "2021-01-27 19:30:00", "2021-03-17 18:30:00",
    #             "2021-04-28 18:30:00", "2021-06-16 18:30:00", "2021-07-28 18:30:00", "2021-09-22 18:30:00", "2021-11-03 18:30:00", "2021-12-15 19:30:00"]
    # start_ts = ["2024-01-31 19:30:00", "2023-02-01 19:30:00", "2023-03-22 18:30:00"
    #             , "2023-05-03 18:30:00", "2023-06-14 18:30:00", "2023-07-26 18:30:00", "2023-09-20 18:30:00", "2023-11-01 18:30:00", "2023-12-13 19:30:00"]
    # api_key = '43bcd56f4dd4ea381339cfbc6fcd86c3'  # Replace with your actual API key
    # query = "(Federal Reserve) OR (Fed Reserve)"
    # # start_date = '2023-09-20'  # Replace with the desired start date (YYYY-MM-DD)
    # # end_date = '2023-09-20'  # Replace with the desired end date (YYYY-MM-DD)
    # # start_ts = ["2024-03-20 18:30:00", "20S24-01-31 19:30:00"]
    # for ts in start_ts:
    #     page = 1
    #     date = ts[:10]
    #     articles = list()
    #     article_titles = set()
    #     while True:
    #         start_time = date + "T00:00:00Z"
    #         end_time = datetime.datetime.strftime(datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - datetime.timedelta(hours=0, minutes=61), "%Y-%m-%dT%H:%M:%SZ")
    #         time.sleep(1)
    #         news_data = get_general_news(query, api_key, start_time, end_time, page)
    #         if "articles" not in news_data or len(news_data["articles"]) == 0:
    #             break
    #         total_articles_count = news_data["totalArticles"]
    #         articles.extend(process_and_save_news(news_data, "../data_5Min/articles/" + date + ".txt"))
    #         # print(f"Page {page} news saved.")
    #         page += 1
    #     articles_data = pd.DataFrame(articles, columns = ['title', 'publishedAt', 'description', 'content', 'url'])
    #     articles_data.to_csv('../data_5Min/articles/' + date + '.csv')
    #     print("News data has been saved to " +  "../data_5Min/articles/" + date + ".txt, number of articles found: " + str(total_articles_count))


    
    for filename in os.listdir("data_5Min/articles"):
        if not filename.endswith(".csv"):
            continue
        news_data = pd.read_csv("data_5Min/articles/" + filename)
        date = filename.split(".")[0]
        news_data = news_data[news_data['publishedAt'] >= date + "T08:00:00"]
        with open(date + ".txt", 'a+', encoding='utf-8') as file:
            for _, row in news_data.iterrows():
                file.write(f"{row['description']}\n")
                file.write(f"{row['content']}\n")
                # file.write(f"URL: {article.get('url', 'N/A')}\n")
                # file.write(article.get('text'))
                file.write("\n\n")
        file.close()