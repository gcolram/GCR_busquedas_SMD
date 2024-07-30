from flask import Flask, request, jsonify
import os
import logging
import json
from dotenv import load_dotenv
from airtable import Airtable
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import openai
import spacy
from nltk.corpus import wordnet
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar las variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# Inicializar el cliente de Airtable
airtable_client = Airtable(base_id=AIRTABLE_BASE_ID, api_key=AIRTABLE_TOKEN, table_name=AIRTABLE_TABLE_NAME)

# Inicializar el modelo y el tokenizador de Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Inicializar cliente de OpenAI
openai.api_key = OPENAI_API_KEY

# Inicializar Spacy
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    logging.error("El modelo de Spacy 'es_core_news_md' no está descargado. Ejecute 'python -m spacy download es_core_news_md' para descargarlo.")
    exit(1)

# Definición de la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "Bienvenido a la API de mi aplicación."

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400

        records = fetch_records()
        if not records:
            return jsonify({"error": "No records found"}), 500

        results = semantic_search(query, records)
        if not results:
            return jsonify({"error": "No relevant records found"}), 404

        response = generate_natural_language_response(query, results, records)
        detailed_report = generate_detailed_report(results, records)

        return jsonify({
            "response": response,
            "detailed_report": detailed_report
        })

    except Exception as e:
        logging.error(f"Error processing search request: {e}")
        return jsonify({"error": str(e)}), 500

def fetch_records():
    try:
        records = airtable_client.get_all()
        logging.info(f"Se han obtenido {len(records)} registros de Airtable.")
        return records
    except Exception as e:
        logging.error(f"Error obteniendo datos de Airtable: {e}")
        return None

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def encode_texts(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)

def save_embeddings(embeddings, filename, timestamp):
    np.savez(filename, embeddings=embeddings.numpy(), timestamp=timestamp)

def load_embeddings(filename):
    data = np.load(filename)
    return torch.tensor(data['embeddings']), data['timestamp']

def tfidf_filter(query, texts, top_n=100):
    vectorizer = TfidfVectorizer()
    preprocessed_texts = [preprocess_text(text) for text in texts]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    query_tfidf = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return related_docs_indices, cosine_similarities

def normalize_scores(scores):
    if np.max(scores) - np.min(scores) == 0:
        return np.zeros_like(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

def expand_query(query):
    expanded_query = set()
    for word in query.split():
        expanded_query.add(word)
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_query.add(lemma.name())
    return ' '.join(expanded_query)

def semantic_search(query, records):
    search_results = []

    combined_texts = []
    record_ids = []
    for record in records:
        fields = record['fields']
        combined_text = " ".join([preprocess_text(str(value)) for key, value in fields.items() if value])
        if combined_text:
            combined_texts.append(combined_text)
            record_ids.append(record['id'])

    expanded_query = expand_query(query)

    # TF-IDF Filtering
    filtered_indices, tfidf_scores = tfidf_filter(expanded_query, combined_texts)
    filtered_texts = [combined_texts[i] for i in filtered_indices]
    filtered_record_ids = [record_ids[i] for i in filtered_indices]

    try:
        embeddings_file = "embeddings.npz"
        current_timestamp = max(record['createdTime'] for record in records)
        current_timestamp = datetime.fromisoformat(current_timestamp[:-1]).timestamp()
        if os.path.exists(embeddings_file):
            embeddings, saved_timestamp = load_embeddings(embeddings_file)
            if current_timestamp <= saved_timestamp:
                logging.info("Usando embeddings guardados.")
                query_embedding = encode_texts([query])
            else:
                raise ValueError("Los datos han cambiado, se deben recalcular los embeddings.")
        else:
            raise ValueError("El archivo de embeddings no existe.")
    except Exception as e:
        logging.info(f"{e}. Calculando y guardando nuevos embeddings.")
        embeddings = encode_texts(filtered_texts)
        save_embeddings(embeddings, embeddings_file, current_timestamp)
        query_embedding = encode_texts([query])

    cosine_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Normalizar los scores
    normalized_tfidf_scores = normalize_scores(tfidf_scores[filtered_indices])
    normalized_cosine_scores = normalize_scores(cosine_scores)

    # Combinación ponderada de los scores con ajuste dinámico
    alpha = 0.3  # Peso de TF-IDF
    beta = 0.7   # Peso de BERT
    if max(normalized_tfidf_scores) > 0.5:  # Ajuste dinámico
        alpha = 0.5
        beta = 0.5

    for tfidf_score, cosine_score, record_id, combined_text in zip(normalized_tfidf_scores, normalized_cosine_scores, filtered_record_ids, filtered_texts):
        total_score = alpha * tfidf_score + beta * cosine_score
        search_results.append((total_score, record_id, combined_text))

    search_results = sorted(search_results, key=lambda x: x[0], reverse=True)[:10]

    logging.info(f"Se han encontrado {len(search_results)} registros relevantes.")
    return search_results

def generate_natural_language_response(query, results, records):
    if not results:
        return f"No se encontraron escritos relevantes para la consulta '{query}'."

    relevant_texts = ""
    for score, record_id, combined_text in results:
        record = next((r for r in records if r['id'] == record_id), None)
        if record:
            fields = record['fields']
            relevant_texts += f"ID: {record_id}\n"
            for key, value in fields.items():
                relevant_texts += f"{key}: {value}\n"
            relevant_texts += f"Texto combinado utilizado para la búsqueda: {combined_text[:100]}...\n\n"  # Mostrar primeros 100 caracteres del texto combinado

    prompt = (f"Basado en los siguientes escritos relevantes, responde a la consulta '{query}':\n\n"
              f"{relevant_texts}\n\n"
              "Respuesta:")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,  # Aumentar significativamente el número de tokens
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

def generate_detailed_report(results, records):
    report = ""
    for score, record_id, combined_text in results:
        record = next((r for r in records if r['id'] == record_id), None)
        if record:
            fields = record['fields']
            report += f"ID: {record_id}, Score: {score}\n"
            for key, value in fields.items():
                report += f"{key}: {value}\n"
            report += f"Texto combinado utilizado para la búsqueda: {combined_text[:200]}...\n\n"  # Mostrar primeros 200 caracteres del texto combinado
            report += "="*50 + "\n"
    return report

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
