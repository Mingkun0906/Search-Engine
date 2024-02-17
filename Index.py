from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import os
import json
import re
from collections import defaultdict
import math

STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
])
LOCAL_DIR = "/Users/mingkunliu/Downloads/WEBPAGES_RAW/"
LOCAL_INDEX = "/Users/mingkunliu/Downloads/121/Project3/inverted_index"


def create_inverted_index():
    inverted_index = defaultdict(dict)
    doc_freq = defaultdict(int)  # Document frequency for each token

    with open(os.path.join(LOCAL_DIR, "bookkeeping.json"), "r") as the_file:
        bookkeeping = json.load(the_file)

    total_docs = len(bookkeeping.keys()) # total number of documents
    
    for doc_id in bookkeeping.keys():
        file_path = os.path.join(LOCAL_DIR, doc_id)
        tokens = tokenizer(file_path)
        token_freq = calculate_tf(tokens)
        seen_tokens = set()

        for token in tokens:
            if token not in seen_tokens:
                doc_freq[token] += 1
                seen_tokens.add(token)
            inverted_index[token][doc_id] = token_freq[token]  # Store TF initially

    for token, docs in inverted_index.items():
        idf = calculate_idf(doc_freq[token], total_docs)
        for doc_id in docs:
            inverted_index[token][doc_id] *= idf  # Update TF to TF-IDF
    
    return inverted_index
    

def calculate_idf(doc_freq, total_docs):
    return math.log(total_docs / (1 + doc_freq))


def calculate_tf(tokens):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total_tokens = len(tokens)
    tf_normalized = {}
    for token, count in tf.items():
        tf_normalized[token] = count / total_tokens
    return tf_normalized


def write_inverted_index(inverted_index):
    sorted_inverted_index = {k: inverted_index[k] for k in sorted(inverted_index)}
    with open("/Users/mingkunliu/Downloads/121/Project3/inverted_index", 'w') as file:
        json.dump(sorted_inverted_index, file, indent=4)
    

def load_inverted_index(file_path):
    with open(file_path, 'r') as file:
        inverted_index = json.load(file)
    return inverted_index


def tokenizer(file_path):
    with open(file_path, "r", encoding = "utf-8") as the_file:
        html_content = the_file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    tokens = tokenize_helper(text)
    return tokens


def tokenize_helper(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if re.match("^[a-zA-Z]+$", token)]
    tokens = [token for token in tokens if token not in STOP_WORDS]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def search_and_rank(query_tokens, inverted_index):
    doc_scores = defaultdict(float)
    for token in query_tokens:
        if token in inverted_index:
            for doc_id, tfidf in inverted_index[token].items():
                doc_scores[doc_id] += tfidf

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


def prompt():
    query = input("What are you looking for? ")
    query_tokens = tokenize_helper(query)
    inverted_index = load_inverted_index(LOCAL_INDEX)
    ranked_docs = search_and_rank(query_tokens, inverted_index)
    with open(os.path.join(LOCAL_DIR, "bookkeeping.json"), "r") as the_file:
        bookkeeping = json.load(the_file)
    for doc_id, _ in ranked_docs[0:20]:
        print(f"{doc_id}: {bookkeeping[doc_id]}")
    
    print(f"Unique docId in index: {count_unique_doc_ids(inverted_index)}")
    print(f"Unique words in index: {count_unique_tokens(inverted_index)}")
    print(f"Total size in KB: {count_index_size(LOCAL_INDEX)}")
    print(len(ranked_docs))
    

def count_unique_doc_ids(inverted_index):
    unique_doc_ids = set()
    for docs in inverted_index.values():
        unique_doc_ids.update(docs.keys())
    return len(unique_doc_ids)


def count_unique_tokens(inverted_index):
    unique_tokens_count = len(inverted_index.keys())
    return unique_tokens_count


def count_index_size(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / 1024


def run():
    if os.path.exists(LOCAL_INDEX):
        prompt()

    else:
        try:
            inverted_index = create_inverted_index()
            write_inverted_index(inverted_index)
            prompt()
        except:
            pass
    
if __name__ == "__main__":
    run()

