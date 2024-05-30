import json
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Carregar o arquivo JSON
try:
    with open('reviews_formatted.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("Arquivo JSON carregado com sucesso.")
except FileNotFoundError:
    print("Arquivo JSON não encontrado. Verifique o nome e o caminho do arquivo.")
    exit()

# Converter para DataFrame
try:
    df = pd.json_normalize(data)
    print("Arquivo JSON convertido para DataFrame com sucesso.")
    print(df.head())  # Exibir as primeiras linhas do DataFrame para verificação
except Exception as e:
    print(f"Erro ao converter JSON para DataFrame: {e}")
    exit()

# Inicializar stopwords e stemmer
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def processar_texto(texto):
    # Análise Léxica
    tokens = word_tokenize(texto)
    # Remoção de Stopwords
    tokens_sem_stopwords = [word for word in tokens if word.lower() not in stop_words]
    # Stemming
    tokens_stemmed = [stemmer.stem(word) for word in tokens_sem_stopwords]
    return tokens_stemmed

# Aplicar o processamento na coluna 'description'
try:
    df['description_processada'] = df['description'].apply(processar_texto)
    print("Texto processado com sucesso.")
    print(df[['title', 'description', 'description_processada']].head())  # Exibir o resultado processado
except KeyError:
    print("Coluna 'description' não encontrada no DataFrame.")
except Exception as e:
    print(f"Erro ao processar texto: {e}")
