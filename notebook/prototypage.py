# Faire les importations nécessaires
from mistralai import Mistral
import psycopg
from psycopg import Cursor
import os
import time

# Déclarer les variables nécessaires
# Chemin absolu depuis le dossier notebook
conversation_folder = os.path.join(os.path.dirname(__file__), "..", "data", "DISTRIBUTION_ACCUEIL_UBS", "TRANS_TXT")

# Initialiser le client Mistral
mistral_client = Mistral(api_key="aYaXhZOe6ZZxwB7cB3Jkd0PtyHmGZZ4k")

db_connection_str = "dbname=ragdb user=postgres password=postgresql host=localhost port=5432"

# Modèle d'embedding Mistral
EMBEDDING_MODEL = "mistral-embed"


def create_conversation_list(folder_path: str) -> list[str]:
    text_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="latin-1") as file:
                text = file.read()
                lines = text.split("\n")
                filtered_list = [chaine.removeprefix("     ") for chaine in lines if not chaine.startswith("<")]
                text_list.extend(filtered_list)
    return text_list


def calculate_embeddings_batch(corpus_list: list[str], client: Mistral) -> list[list[float]]:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            embeddings = client.embeddings.create(inputs=corpus_list, model=EMBEDDING_MODEL)
            return [emb.embedding for emb in embeddings.data]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise


def calculate_embeddings(corpus: str, client: Mistral) -> list[float]:
    return calculate_embeddings_batch([corpus], client)[0]


def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    cursor.execute('''INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)''', (corpus, embedding))


def similar_corpus(input_corpus: str, client: Mistral, db_connection_str: str) -> list[tuple[int, str, float]]:
    input_embedding = calculate_embeddings(corpus=input_corpus, client=client)
    
    with psycopg.connect(db_connection_str) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, corpus, embedding <-> %s::vector AS distance
                FROM embeddings
                ORDER BY distance
                LIMIT 5
            """, (input_embedding,))
            results = cur.fetchall()
    
    return results


with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""DROP TABLE IF EXISTS embeddings""")
        
        # Créer l'extension pgvector
        cur.execute("""CREATE EXTENSION IF NOT EXISTS vector""")
        
        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                    ID SERIAL PRIMARY KEY, 
                    corpus TEXT,
                    embedding VECTOR(1024));  
                    """)
        
        corpus_list = create_conversation_list(folder_path=conversation_folder)
        corpus_list = [c for c in corpus_list if c.strip()]
        
        # Traiter par batch de 50
        batch_size = 50
        for i in range(0, len(corpus_list), batch_size):
            batch = corpus_list[i:i+batch_size]
            embeddings = calculate_embeddings_batch(batch, mistral_client)
            
            for corpus, embedding in zip(batch, embeddings):
                save_embedding(corpus=corpus, embedding=embedding, cursor=cur)
            
            time.sleep(2)
        
        conn.commit()


# Introduire une requête pour interroger
query = "cours d'anglais"
results = similar_corpus(input_corpus=query, client=mistral_client, db_connection_str=db_connection_str)

print("Résultats de la recherche:")
for id, corpus, distance in results:
    print(f"ID: {id}, Distance: {distance:.4f}")
    print(f"Texte: {corpus}\n")