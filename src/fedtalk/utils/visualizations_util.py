from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import pandas as pd

# display word cloud graphic of input phrases - frequencies all initialized to 1- why?

def generate_and_display_word_cloud(phrases, title):
    # Creating a dictionary where each phrase is a key and each value is the same
    # This approach maintains the integrity of each phrase without using underscores
    phrase_frequencies = {}
    for phrase in phrases:
        if phrase not in phrase_frequencies:
            phrase_frequencies[phrase] = 0
        phrase_frequencies[phrase] += 1

    # Generating the word cloud with phrase frequencies
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(phrase_frequencies)

    # Displaying the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=15)
    plt.show()


def plot_frequency_bar_chart(keywords, title):
    # Plot frequency bar chart given list of keywords and title of bar chart
    keywords_frequencies = {}
    for keyword in keywords:
        if keyword not in keywords_frequencies:
            keywords_frequencies[keyword] = 0
        keywords_frequencies[keyword] += 1


    _, ax = plt.subplots()

    ax.bar(keywords_frequencies.keys(), keywords_frequencies.values())
    ax.set_xlabel("Keywords")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    plt.show()

def plot_topics_similarity(topics_speech, topics_other, title):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    topics_speech_embeddings = model.encode(topics_speech)
    topics_other_embeddings = model.encode(topics_other)

    similarity_matrix = util.cos_sim(topics_speech_embeddings, topics_other_embeddings)

    df_similarity = pd.DataFrame(similarity_matrix.numpy(), index = topics_speech,
                                 columns = topics_other)
    plt.figure(figsize = (15, 10))
    sns.heatmap(df_similarity, annot = False)
    plt.title(title)
    plt.xlabel("Speech topics")
    plt.ylabel("News topics")
    plt.show()

