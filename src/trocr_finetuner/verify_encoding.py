# src/trocr_finetuner/verify_encoding.py

import os

import pandas as pd
from datasets import load_dataset

print("--- INICIANDO SCRIPT DE DIAGNÓSTICO DE CODIFICAÇÃO ---")

# --- TESTE 1: Leitura do seu arquivo CSV ---
print("\n--- TESTE 1: Lendo 'labels.csv' com Pandas ---")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))
    csv_path = os.path.join(data_dir, "labels.csv")

    print("\nAttempt 1: Lendo como UTF-8...")
    df_utf8 = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    print("Primeiras 5 labels (lidas como UTF-8):")
    print(df_utf8["text"].head().to_list())

    print("\nAttempt 2: Lendo como Latin-1...")
    df_latin1 = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    print("Primeiras 5 labels (lidas como Latin-1):")
    print(df_latin1["text"].head().to_list())

except FileNotFoundError:
    print(f"ERRO: Arquivo CSV não encontrado em '{csv_path}'. Verifique o caminho.")
except Exception as e:
    print(f"ERRO ao ler o CSV: {e}")


# --- TESTE 2: Leitura da Wikipedia com a biblioteca 'datasets' ---
print("\n--- TESTE 2: Lendo 'Wikipedia' com a biblioteca Datasets ---")
try:
    # Filtra por um artigo que garantidamente tem acentos
    wiki = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train[:100]")
    portugal_texts = [
        text for text in wiki["text"]
        if "História de Portugal" in text or "Revolução dos Cravos" in text
    ]
    if portugal_texts:
        print("Amostra de texto da Wikipedia (deve ter acentos):")
        print(portugal_texts[0][:200] + "...") # Mostra os primeiros 200 caracteres
    else:
        print("Não foi encontrada amostra específica da Wikipedia, mostrando a primeira linha:")
        print(wiki["text"][0][:200] + "...")

except Exception as e:
    print(f"ERRO ao carregar o dataset da Wikipedia: {e}")


# --- TESTE 3: Verificação do sistema de arquivos ---
print("\n--- TESTE 3: Testando escrita e leitura de arquivo local ---")
test_string = "Atenção: a votação começará em breve."
test_file_path = "encoding_test.txt"
try:
    print(f"Escrevendo a frase '{test_string}' em '{test_file_path}' com UTF-8...")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_string)

    print("Lendo a frase de volta do arquivo com UTF-8...")
    with open(test_file_path, encoding="utf-8") as f:
        read_string = f.read()

    print(f"Frase lida: '{read_string}'")

    if test_string == read_string:
        print("✅ SUCESSO: A escrita e leitura de arquivos em UTF-8 está funcionando corretamente.")
    else:
        print("❌ FALHA: A frase lida é diferente da original. Há um problema fundamental de I/O no seu ambiente.")

    os.remove(test_file_path) # Limpa o arquivo de teste

except Exception as e:
    print(f"ERRO durante o teste de escrita/leitura: {e}")

print("\n--- DIAGNÓSTICO CONCLUÍDO ---")
