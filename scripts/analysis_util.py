from openai import OpenAI
from api_keys import openai_api_key
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json

client = OpenAI(api_key=openai_api_key)


def rate_news(news):
    system_prompt = """
    Suppose you are an analyst, and you need to classify the content of a speech in a certain society and determine the emotional bias of the speech.
    Categories:

    - positive: The speaker is generally optimistic about the content of the speech and the mood is relatively positive.
    - negative: The speaker is generally not optimistic about the content of the speech, and the mood is relatively negative.
    - neutral: The speaker has no obvious emotional bias towards the content of the speech, and prefers simple explanation.

    """
    user_prompt = f"""
    Read the following news and classify its emotional bias and certainty level:
    News: {news}.
    Output format: [bias] [certainty level]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # 解析响应并分割为情绪倾向和确定性等级
        parts = response.choices[0].message.content.strip().lower().split()
        # 验证输出格式并处理不符合预期的情况
        if len(parts) == 2 and parts[0] in ['positive', 'negative', 'neutral'] and parts[1] in ['high', 'moderate',
                                                                                                'low']:
            return parts[0], parts[1]
        else:
            print(f"Unexpected format or content: {response.choices[0].message.content}")
            return "unknown", "unknown"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "error", "error"
    

def get_topic_and_keywords(news):
    system_prompt = """
    Imagine you are an analyst tasked with listening to a series of speeches. 
    Your job is to identify the main topic of each speech and extract the key terms mentioned throughout. 
    Pay close attention to the recurring themes, specific terminology used by the speaker, 
    and any particular phrases that encapsulate the essence of the speech. 
    Your analysis will help in categorizing the speeches into thematic areas and 
    understanding the focal points of the speaker's message. Use the following structure to 
    present your findings for each speech:

    #Topic:

    ##Labor Market

    ##Price Stability/Inflation

    ##Economic Activity

    ##Monetary Policy

    ##Trade

    ##Challenges and Risks
    """

    user_prompt = f"""
    Read the following speeches. Then do two things. 
    1: Report only the topics that you determined for the speeches. Do not reply N/A.
    2: Extract the key words in the speech
    speeches: {news}.

    Output a format like

    Topic:"Labor Market"
    Key words:"Unemployment Rate","Wage Growth","Job Creation","Skills Gap"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        # 增加容错处理
        topic_parts = content.split('Topic:')
        topic = topic_parts[1].split('\n')[0].replace('"', '').strip() if len(topic_parts) > 1 else "Unknown"
        keywords_parts = content.split('Key words:')
        keywords = keywords_parts[1].replace('"', '').strip() if len(keywords_parts) > 1 else "Unknown"
        return topic, keywords
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def get_similarities_and_differences(transcript, text):
    system_prompt = """
    As a financial analyst tasked with comparing two texts, your objective is 
    to identify similarities and differences in their themes, topics, and specific 
    terminology used in the two provided texts. Identify similarities and differences between the first text and 
    the second text. Pay close attention to recurring themes and phrases that encapsulate the 
    essence of the texts.
    """

    user_prompt = f"""
    Read the following texts. Then do the following:

    1. Identify and list key terms and phrases that are unique to each text.
    2. Identify the similarities between the first text and the second text, 
        focusing on themes, topics, and specific terminology used.
    3. Identify the differences between the first text and the second text, paying attention to:
        a. Themes or topics that are similar but differ in emphasis or tone.
        b. Variations in the detail or scope of information presented.
        c. Specific terminology used in one text but not the other.

    First Text: 
    "{transcript}"
     
    Second Text:
    "{text}"

    Output format (only in plain text, with no special characters):

    Unique Key Terms/Phrases:

    1. First Text: term/phrase 1, term/phrase 2
    2. Second Text: term/phrase 1, term/phrase 2

    Similarities:

    1. Similarity 1
    2. Similarity 2

    Differences:

    Themes/Topics with Different Emphasis or Tone:
    1. Theme/Topic 1
    2. Theme/Topic 2

    Variations in Detail/Scope:
    1. Detail/Scope difference 1
    2. Detail/Scope difference 2

    Specific Terminology Differences:
    1. Terminology difference 1
    2. Terminology difference 2
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content        
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def get_similar_and_different_terms(transcript, text):
    system_prompt = """
    As a financial analyst tasked with comparing two texts, your objective is 
    to identify similarities and differences in specific 
    terminology used in the two provided texts. Identify 3 similar and 3 different key terms between the first text and 
    the second text. Pay close attention to recurring themes and phrases that encapsulate the 
    essence of the texts, and use that to identify 3 key similar terms, and 3 key terms that are signficantly different and unique to each text.
    """

    user_prompt = f"""
    Read the following texts. Then do the following:

    1. Identify and list 3 key terms or phrases that are similar between the first text and the second text.
    2. Identify and list 3 key terms or phrases that are significantly different between the first text and the second text and unique to each text.

    First Text: 
    "{transcript}"
     
    Second Text:
    "{text}"

    Output format: Return the output in the specified format.
    Do NOT include any additional text. Return the output in string format, NOT in JSON format.

    {{
        "Similar": [term/phrase 1, term/phrase 2],
        "Different": [(term/phrase 1 from first text, term/phrase 2 from second text), (term/phrase 3 from first text, term/phrase 4 from second text)]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content        
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def compare_sentiment_and_certainity_level(transcript, text):
    system_prompt = """
    You are an analyst tasked with comparing the sentiment of two professional texts, which typically maintain a neutral tone. 
    Your goal is to identify and compare subtle differences in emotional bias and certainty levels conveyed through 
    specific details in word choice, sentence structure, and overall tone.
    Compare how the sentiment of the first text relates to the second text in terms of emotional bias and certainty level.
    """
    user_prompt = f"""
   Steps:

    1. Compare Emotional Bias:
    Determine the subtle emotional bias (slightly more positive, slightly more negative, or similarly neutral) of the first text with respect to the 
    second text. Discuss whether the first text is slightly more positive, slightly more negative, or similarly 
    neutral compared to the second text. Provide specific phrases or sections from both texts that illustrate 
    the emotional biases.

    2. Compare Certainty Levels:
    Determine the certainty level (high, moderate, or low) of the first text with respect to the second text. 
    Discuss whether the first text expresses higher, lower, or similar levels of certainty compared to the second text. 
    Provide specific phrases or sections from both texts that illustrate the levels of certainty.

    Categories for Emotional Bias:
    1. Slightly Positive: The speaker shows a subtle optimism or positive outlook.
    2. Slightly Negative: The speaker shows a subtle pessimism or negative outlook.
    3. Neutral: The speaker has no obvious emotional bias towards the content and maintains a balanced, objective tone.

    Certainty Levels:
    1. High: The speaker expresses strong confidence and assurance in their statements.
    2. Moderate: The speaker shows some level of confidence but also acknowledges uncertainties.
    3. Low: The speaker exhibits significant uncertainty or doubt in their statements

    First Text:
    "{transcript}"

    Second Text:
    "{text}"

    Output format (only in plain text, with no special characters):

    Emotional Bias Comparison:

    First text vs Second text: [slightly more positive/slightly more negative/similarly neutral]
    Explanation: [Provide a detailed explanation of the comparison, including specific phrases or sections]

    Certainty Level Comparison:

    First text vs Second text: [higher certainty/lower certainty/similar certainty]
    Explanation: [Provide a detailed explanation of the comparison, including specific phrases or sections]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content        
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    
def get_price_change_predictions_using_statements(train_prompt: list, test_prompt: list):
    # system_prompt = """
    # Given the below prompt and the predictions from GPT based on the prompt, and the actual values, improve the prompt in order to
    # improve the accuracy of the predictions:
    # [{'Id': 8, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 14, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 22, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, 
    # {'Id': 29, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 30, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 32, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 34, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, {'Id': 37, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 46, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 47, 'Predictd Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}]
    # """
    # user_prompt = f"""
    system_prompt = """
    You are an analysis assistant specialized in monetary policy and economic indicators. Your task is to predict future price movements based on 
    detailed analyses of Federal Open Market Committee (FOMC) meeting statements and press conferences. The baseline text is the official summary of 
    FOMC decisions. Analyze the key differences in economic outlook, inflation expectations, labor market conditions, and intentions for monetary 
    policy adjustments, focusing on the nuances in language and emphasis.
    """
    user_prompt = f"""
    Steps:

    1. Understanding the Data:
    - Each record in the data list contains:
        - `Id`: The ID of the record.
        - `First Text`: The FOMC press conference transcript to analyze.
        - `Second Text`: The baseline text containing the official FOMC meeting statement summary.
        - `Price Movement`: The price movement [Positive/Negative]

    2. Detailed Analysis:
    - Compare the `First Text` with the `Second Text`, focusing on:
        - Changes in emphasis on economic growth, inflation expectations, labor market conditions, and financial stability.
        - Adjustments in language related to future monetary policy direction, such as potential rate hikes, pauses, or cuts.
        - Variations in detailed economic outlook and risk assessments.
        - Overall sentiment and terminology used to describe economic conditions and policy stances.

    3. Prediction Based on Analysis:
    - Determine whether these textual differences and overall economic outlook would likely lead to a Positive or Negative price movement.

    4. Training Data Insights:
    - Learn from the provided training data to understand patterns of how particular sentiments, terminologies, and economic indicators 
    correlate with price movements.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Predicting Test Data:
    For each record in the test data list, predict the price movement (Positive/Negative).

    Output Format: Return the output as a list of predictions and a summary of the insights in the specified format. Do NOT include any additional text. Return the output in 
    string format, NOT in JSON format.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }},
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }}
    ],
    "insights": [text summary of the learnings of how patterns of particular sentiments, terminologies, and economic indicators correlate with price movements]
    }}
    """
    try:
        # print(user_prompt)
        # return []
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        # print(content)
        return json.loads(content)
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def get_price_change_predictions_using_news(train_prompt: list, test_prompt: list):
    # system_prompt = """
    # Given the below prompt and the predictions from GPT based on the prompt, and the actual values, improve the prompt in order to
    # improve the accuracy of the predictions:
    # [{'Id': 8, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 14, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 22, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, 
    # {'Id': 29, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 30, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 32, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 34, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, {'Id': 37, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 46, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 47, 'Predictd Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}]
    # """
    # user_prompt = f"""
    system_prompt = """
    You are an analysis assistant specialized in monetary policy and economic indicators. Your task is to predict future price movements based on 
    detailed analyses of news articles related to the Federal Open Market Committee (FOMC) meetings and press conferences. The baseline text is a collection of
    related news articles published on the same day as and before the FOMC meeting press conference. Analyze the key differences in economic outlook, 
    inflation expectations, labor market conditions, and intentions for monetary policy adjustments, focusing on the nuances in language and emphasis.
    """
    user_prompt = f"""
    Steps:

    1. Understanding the Data:
    - Each record in the data list contains:
        - `Id`: The ID of the record.
        - `First Text`: The FOMC press conference transcript to analyze.
        - `Second Text`: The baseline text containing related news articles.
        - `Price Movement`: The price movement [Positive/Negative]

    2. Detailed Analysis:
    - Compare the `First Text` with the `Second Text`, focusing on:
        - Changes in emphasis on economic growth, inflation expectations, labor market conditions, and financial stability.
        - Adjustments in language related to future monetary policy direction, such as potential rate hikes, pauses, or cuts.
        - Variations in detailed economic outlook and risk assessments.
        - Overall sentiment and terminology used to describe economic conditions and policy stances.

    3. Prediction Based on Analysis:
    - Determine whether these textual differences and overall economic outlook would likely lead to a Positive or Negative price movement.

    4. Training Data Insights:
    - Learn from the provided training data to understand patterns of how particular sentiments, terminologies, and economic indicators 
    correlate with price movements.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Predicting Test Data:
    For each record in the test data list, predict the price movement (Positive/Negative).

    Output Format: Return the output as a list of predictions and a summary of the insights in the specified format. Do NOT include any additional text. Return the output in 
    string format, NOT in JSON format.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }},
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }}
    ],
    "insights": [text summary of the learnings of how patterns of particular sentiments, terminologies, and economic indicators correlate with price movements]
    }}
    """
    try:
        # print(user_prompt)
        # return []
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        # print(content)
        return json.loads(content)
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def get_price_change_predictions_using_scores(train_prompt: list, test_prompt: list):
    # system_prompt = """
    # Given the below prompt and the predictions from GPT based on the prompt, and the actual values, improve the prompt in order to
    # improve the accuracy of the predictions:
    # [{'Id': 8, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 14, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 22, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, 
    # {'Id': 29, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 30, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 32, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 34, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, {'Id': 37, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 46, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 47, 'Predictd Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}]
    # """
    # user_prompt = f"""
    system_prompt = """
    You are an advanced analysis assistant with expertise in economic indicators and monetary policy. Your primary objective is to predict future price movements based on 
    a detailed quantitative analysis of Federal Open Market Committee (FOMC) meeting statements, news articles, and press conference transcripts. The key metric provided to you 
    is the average similarity score, which represents the quantitative similarity between the press conference transcript and a baseline text (either the FOMC meeting statement 
    or related news articles). Your task is to interpret this score and use it to accurately predict price movements.
    """

    user_prompt = f"""
    Steps:

    1. Understand the Data:
    - Each data record consists of:
        - `Id`: A unique identifier for the record.
        - `Average Similarity Score`: The computed average similarity score between the press conference transcript and the baseline text (either the FOMC meeting statement 
        or news articles).
        - `Price Movement`: The recorded price movement [Positive/Negative].

    2. Analytical Focus:
    - Analyze the relationship between the `Average Similarity Score` and the `Price Movement`.
    - Identify patterns or thresholds in the similarity score that are indicative of either a Positive or Negative price movement.

    3. Predictive Modeling:
    - Based on your analysis, predict the price movement (Positive/Negative) for each record in the test data set, using the similarity score as the primary input.

    4. Insights from Training Data:
    - Draw conclusions from the training data, highlighting any discovered trends or correlations between similarity scores and price movements.
    - Use these insights to inform your predictions on the test data.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Output Predictions:
    For each record in the test data set, predict the price movement (Positive/Negative).

    Output Format: Provide your output as a structured list of predictions followed by a summary of insights. Avoid any extraneous text. The output should be in a simple string format, not JSON.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }},
        {{
            "Id": [Id of the test data record in integer format],
            "Price Movement": [Prediction of price movement as Positive/Negative]
        }}
    ],
    "insights": [Concise summary of the relationship between the average similarity score and price movement]
    }}
    """
    try:
        # print(user_prompt)
        # return []
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        json_content = json.loads(content)
        # print(content)
        return json_content["predictions"], json_content["insights"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    

def get_price_volatility_predictions_using_scores(train_prompt: list, test_prompt: list):
    # system_prompt = """
    # Given the below prompt and the predictions from GPT based on the prompt, and the actual values, improve the prompt in order to
    # improve the accuracy of the predictions:
    # [{'Id': 8, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 14, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 22, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, 
    # {'Id': 29, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 30, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, {'Id': 32, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 34, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Positive'}, {'Id': 37, 'Predicted Price Movement': 'Negative', 'Actual Price Movement': 'Positive'}, {'Id': 46, 'Predicted Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}, 
    # {'Id': 47, 'Predictd Price Movement': 'Positive', 'Actual Price Movement': 'Negative'}]
    # """
    # user_prompt = f"""
    system_prompt = """
    You are an advanced analysis assistant with expertise in economic indicators and monetary policy. Your primary objective is to predict future price volatility based on 
    a detailed quantitative analysis of Federal Open Market Committee (FOMC) meeting statements, news articles, and press conference transcripts. The key metric provided to you 
    is the average similarity score, which represents the quantitative similarity between the press conference transcript and a baseline text (either the FOMC meeting statement 
    or related news articles). Your task is to interpret this score and use it to accurately predict price volatility.
    """

    user_prompt = f"""
    Steps:

    1. Understand the Data:
    - Each data record consists of:
        - `Id`: A unique identifier for the record.
        - `Average Similarity Score`: The computed average similarity score between the press conference transcript and the baseline text (either the FOMC meeting statement 
        or news articles).
        - `Price Volatility`: The recorded price volatility [High/Low].

    2. Analytical Focus:
    - Analyze the relationship between the `Average Similarity Score` and the `Price Volatility`.
    - Identify patterns or thresholds in the similarity score that are indicative of either a High or Low price volatility.

    3. Predictive Modeling:
    - Based on your analysis, predict the price volatility (High/Low) for each record in the test data set, using the similarity score as the primary input.

    4. Insights from Training Data:
    - Draw conclusions from the training data, highlighting any discovered trends or correlations between similarity scores and price volatility.
    - Use these insights to inform your predictions on the test data.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Output Predictions:
    For each record in the test data set, predict the price volatility (High/Low).

    Output Format: Provide your output as a structured list of predictions followed by a summary of insights. Avoid any extraneous text. The output should be in a simple string format, not JSON.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Price Volatility": [Prediction of price volatility as High/Low]
        }},
        {{
            "Id": [Id of the test data record in integer format],
            "Price Volatility": [Prediction of price volatility as High/Low]
        }}
    ],
    "insights": [Concise summary of the relationship between the average similarity score and price volatility]
    }}
    """
    try:
        # print(user_prompt)
        # return []
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        json_content = json.loads(content)
        # print(content)
        return json_content["predictions"], json_content["insights"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
    
    
def get_market_reaction_predictions(train_prompt: list, test_prompt: list):
    system_prompt = """
    You are an expert financial analyst specializing in understanding market reactions to economic events, particularly FOMC press conferences. Your task is to predict how the market will respond at different time moments during the press conference based on a comparison of the FOMC press conference statement, earlier news releases from the same day, and the FOMC statement released half an hour before the press conference. Your primary focus is to analyze shifts in tone, focus, and any other subtle changes between the different sources.
    """

    user_prompt = f"""
    Steps:

    1. Understand the Data:
    - Each data record consists of:
        - `Id`: A unique identifier for the record.
        - `Press Conference Statement`: The transcript of the FOMC press conference statement.
        - `Earlier News Releases`: The news releases published earlier on the same day.
        - `FOMC Statement`: The FOMC statement released half an hour before the press conference.
        - `Market Reaction`: The recorded market reaction (Positive, Negative, No Movement).

    2. Analytical Focus:
    - Analyze the differences between the `Press Conference Statement`, `Earlier News Releases`, and the `FOMC Statement`.
    - Identify shifts in tone or focus, and how they may signal a positive or negative market reaction at different time moments during the press conference.

    3. Predictive Modeling:
    - Based on your analysis, predict the market reaction (Positive, Negative) for different time moments during the press conference.

    4. Insights from Training Data:
    - Draw conclusions from the training data, focusing on patterns or inconsistencies between the sources.
    - Use these insights to inform your predictions on the test data.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Output Predictions:
    For each record in the test dataset, predict the market reaction at different time moments during the press conference.

    Output Format: Provide your output as a structured list of predictions, including explanations for each reaction at different time moments. Avoid any extraneous text. The output should be in a simple string format, not JSON.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Market Reaction": 
            [
                {{
                    "Time": [Time moment during the press conference],
                    "Reaction": [Positive/Negative],
                    "Explanation": [Key reasons why this market reaction occurred at this time based on changes in sentiment, tone, or focus]
                }},
                {{
                    "Time": [Another time moment],
                    "Reaction": [Positive/Negative],
                    "Explanation": [Reasons for the reaction at this time moment]
                }}
            ]
        }}
    ],
    "insights": [Overall summary of the patterns or inconsistencies observed across different time moments and how they affect market reactions]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        json_content = json.loads(content)
        
        # Remove "No Movement" reactions entirely
        filtered_predictions = [
            {
                "Id": prediction["Id"],
                "Market Reaction": [
                    reaction for reaction in prediction["Market Reaction"] if reaction["Reaction"] in ["Positive", "Negative"]
                ]
            }
            for prediction in json_content["predictions"]
        ]
        
        return filtered_predictions, json_content["insights"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"





import json
import openai  # DeepSeek uses OpenAI-compatible API

def get_market_reaction_predictions(train_prompt: list, test_prompt: list):
    system_prompt = """
    You are an expert financial analyst specializing in understanding market reactions to economic events, particularly FOMC press conferences. Your task is to predict how the market will respond at different time moments during the press conference based on a comparison of the FOMC press conference statement, earlier news releases from the same day, and the FOMC statement released half an hour before the press conference. Your primary focus is to analyze shifts in tone, focus, and any other subtle changes between the different sources.
    """

    user_prompt = f"""
    Steps:

    1. Understand the Data:
    - Each data record consists of:
        - `Id`: A unique identifier for the record.
        - `Press Conference Statement`: The transcript of the FOMC press conference statement.
        - `Earlier News Releases`: The news releases published earlier on the same day.
        - `FOMC Statement`: The FOMC statement released half an hour before the press conference.
        - `Market Reaction`: The recorded market reaction (Positive, Negative, No Movement).

    2. Analytical Focus:
    - Analyze the differences between the `Press Conference Statement`, `Earlier News Releases`, and the `FOMC Statement`.
    - Identify shifts in tone or focus, and how they may signal a positive or negative market reaction at different time moments during the press conference.

    3. Predictive Modeling:
    - Based on your analysis, predict the market reaction (Positive, Negative) for different time moments during the press conference.

    4. Insights from Training Data:
    - Draw conclusions from the training data, focusing on patterns or inconsistencies between the sources.
    - Use these insights to inform your predictions on the test data.

    Training Data:
    {train_prompt}

    Test Data:
    {test_prompt}

    Output Predictions:
    For each record in the test dataset, predict the market reaction at different time moments during the press conference.

    Output Format: Provide your output as a structured list of predictions, including explanations for each reaction at different time moments. Avoid any extraneous text. The output should be in a simple string format, not JSON.

    {{
    "predictions": 
    [
        {{
            "Id": [Id of the test data record in integer format],
            "Market Reaction": 
            [
                {{
                    "Time": [Time moment during the press conference],
                    "Reaction": [Positive/Negative],
                    "Explanation": [Key reasons why this market reaction occurred at this time based on changes in sentiment, tone, or focus]
                }},
                {{
                    "Time": [Another time moment],
                    "Reaction": [Positive/Negative],
                    "Explanation": [Reasons for the reaction at this time moment]
                }}
            ]
        }}
    ],
    "insights": [Overall summary of the patterns or inconsistencies observed across different time moments and how they affect market reactions]
    }}
    """

    try:
        # Initialize DeepSeek API client
        client = openai.OpenAI(api_key="sk-904cf05c6c68487b89ba443439912f65", base_url="https://api.deepseek.com/v1")

        response = client.chat.completions.create(
            model="deepseek-coder-r1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.choices[0].message.content
        json_content = json.loads(content)

        # Remove "No Movement" reactions entirely
        filtered_predictions = [
            {
                "Id": prediction["Id"],
                "Market Reaction": [
                    reaction for reaction in prediction["Market Reaction"] if reaction["Reaction"] in ["Positive", "Negative"]
                ]
            }
            for prediction in json_content["predictions"]
        ]

        return filtered_predictions, json_content["insights"]

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error", "Error"
