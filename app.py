from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords



# Use to handle certificate error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('stopwords')

app = Flask(__name__)




# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents)

n_components = 100
svd_model = TruncatedSVD(n_components=n_components)
X_reduced = svd_model.fit_transform(X)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    query_vec = vectorizer.transform([query])

    query_reduced = svd_model.transform(query_vec)

    cos_sim = cosine_similarity(query_reduced, X_reduced)

    top_indices = np.argsort(cos_sim[0])[::-1][:5]
    top_similarities = cos_sim[0][top_indices]
    top_documents = [documents[i] for i in top_indices]

    return top_documents, top_similarities.tolist(), top_indices.tolist()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
