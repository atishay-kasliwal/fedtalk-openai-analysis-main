from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd
import articles_util
from api_keys import pinecone_api_key
import itertools

pc = Pinecone(api_key=pinecone_api_key)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
model_dimension = 768
pinecone_spec = {"serverless": {"region": "us-east-1","cloud": "aws"}}

def init_db(index_name: str):
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension = model_dimension, spec = pinecone_spec)

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def upsert_data(index_name: str, namespace: str, data_file: str):
    print("Upserting " + index_name + " " + data_file)

    data = pd.read_csv(data_file)

    index = pc.Index(index_name)

    vectors = list()

    sentence_vectors = set()

    for _, row in data.iterrows():
        sent_vector = model.encode(row['sentence'])
        if tuple(sent_vector) not in sentence_vectors:
            sentence_vectors.add(tuple(sent_vector))
            vectors.append({"id": str(row['id']), "values": sent_vector})
    
    # for i in range(20, len(vectors), 20):
    #     print(f"Upserting {i-20} to {i} of vectors")
    #     index.upsert(vectors = vectors[:i], namespace = namespace)

    # Upsert data with 200 vectors per upsert request
    for ids_vectors_chunk in chunks(vectors, batch_size=200):
        print(f"Upserting batch")
        index.upsert(vectors=ids_vectors_chunk, namespace = namespace) 

def query(index_name: str, date: str, query_text: str, num_matches: int):
    index = pc.Index(index_name)

    embeddings = model.encode(query_text)
    # print(embeddings)

    response = index.query(namespace = str(date), vector = embeddings, top_k = num_matches, include_metadata = True)

    data_ids = set()

    total_similarity_score = 0

    if "matches" in response:
        for match in response["matches"]:
            data_ids.add(match["id"])
            total_similarity_score += match["score"]
    
    if data_ids:
        if index_name == "statement":
            return get_statement_sentences(date, data_ids), round(total_similarity_score/len(response["matches"]), 4)
        else:
            return get_news_sentences(date, data_ids), round(total_similarity_score/len(response["matches"]), 4)

    return ""

def query_avg_similarity_score(index_name: str, date: str, query_text: str, num_matches: int):
    index = pc.Index(index_name)

    embeddings = model.encode(query_text)
    # print(embeddings)

    response = index.query(namespace = str(date), vector = embeddings, top_k = num_matches, include_metadata = True)

    total_similarity_score = 0

    if "matches" in response:
        for match in response["matches"]:
            total_similarity_score += match["score"]
    
        return round(total_similarity_score/len(response["matches"]), 4)

    return 0

def get_statement_sentences(date: str, data_ids: set):
    statements_data_file_name = articles_util.STATEMENTS_BASE_PATH + str(date) + '.csv'
    statements_data = pd.read_csv(statements_data_file_name)
    statements_data = statements_data[statements_data.id.map(str).isin(data_ids)]
    return ".".join(statements_data['sentence'].tolist())

def get_news_sentences(date: str, data_ids: set):
    news_data_file_name = articles_util.ARTICLES_BASE_PATH + "news/" + str(date) + '.csv'
    news_data = pd.read_csv(news_data_file_name)
    news_data = news_data[news_data.id.map(str).isin(data_ids)]
    return ".".join(news_data['sentence'].tolist())