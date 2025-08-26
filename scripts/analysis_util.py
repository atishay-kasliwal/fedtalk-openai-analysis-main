from openai import OpenAI
from api_keys import openai_api_key
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json

client = OpenAI(api_key=openai_api_key)


def get_market_reaction_predictions(train_prompt: list, test_prompt: list):
    system_prompt = """
    You are an expert financial analyst specializing in understanding market reactions to economic events, particularly FOMC press conferences. Your task is to predict how the market will respond at different time moments during the FOMC press conference based on a comparison of the FOMC press conference speech transcript, earlier news releases from the same day, and the FOMC statement released half an hour before the press conference. Your primary focus is to analyze similarities, shifts in tone, focuses, any other subtle changes, and patterns among the different sources. You will use this analysis to generate similarity scores among the three different sources of information, based on the level of impact of these similarities and differences on the market price movement. 
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
    
    - Analyze the similarities and differences between the `Press Conference Statement`, `Earlier News Releases`, and the `FOMC Statement`.
    - Analyze similarities, shifts in tone, focus, any other subtle changes, and patterns among the different sources, and how they may signal a positive or negative market reaction at different time moments during the press conference.
        
    3. Predictive Modeling:
    - Based on your analysis, predict the market reaction (Positive, Negative) for different time moments during the press conference.
    - Include the magnitude of the reaction as a percentage change.
    - Based on your analysis, predict the market reaction (Positive, Negative, percentage change) for different time moments during the press conference.
    - Generate three similarity scores: similarity between `Press Conference Statement' and `Earlier News Releases`; similarity between `Press Conference Statement' and `FOMC Statement`; similarity between `Earlier News Releases` and `FOMC Statement` in percentage format only. 
    
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
                    "Percent Change": [percentage change],
                    "Explanation": [Key reasons why you predicted the direction and magnitude of market reaction occurred at this time],
                    "Similarity 1": [Similarity Score between `Press Conference Statement` and `Earlier News Releases in integer format only, without the '%' sign. `],
                    "Similarity 2": [Similarity Score between `Press Conference Statement` and `FOMC Statement in integer format only, without the '%' sign. `],
                    "Similarity 3": [Similarity Score between `Earlier News Releases` and `FOMC Statement in integer format only, without the '%' sign. `]
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



