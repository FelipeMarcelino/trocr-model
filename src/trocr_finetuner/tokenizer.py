import os

import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, RobertaTokenizerFast

# ==============================================================================
# 0. CONFIGURAÇÃO DE CAMINHOS
# ==============================================================================
print("--- 0. Configurando Caminhos ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.abspath(os.path.join(script_dir, "../../tokenizer"))
data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))
csv_path = os.path.join(data_dir, "labels.csv")

os.makedirs(output_dir, exist_ok=True)
print(f"Diretório de saída: {output_dir}\n")

# ==============================================================================
# 1. COLETA E VERIFICAÇÃO DO CORPUS
# ==============================================================================
print("--- 1. Coletando Corpus em Português ---")

# --- Leitura Defensiva do seu CSV ---
try:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    print("CSV lido com sucesso como UTF-8.")
except UnicodeDecodeError:
    print("Falha ao ler CSV como UTF-8. Tentando como Latin-1...")
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    print("CSV lido com sucesso como Latin-1.")
labels_dataset = df["text"].astype(str).tolist()
print(f"Carregadas {len(labels_dataset)} labels do seu dataset.")

# --- Corpus Públicos ---
wiki = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train[:1%]")
wiki_texts = [str(x["text"]) for x in wiki]
print(f"Carregados {len(wiki_texts)} documentos da Wikipedia.")

all_texts = labels_dataset + wiki_texts
print(f"\nCorpus total combinado com {len(all_texts)} documentos em memória.\n")

# ==============================================================================
# 2. TREINAMENTO DO TOKENIZER (MÉTODO À PROVA DE FALHAS)
# ==============================================================================
print("--- 2. Treinando o Tokenizer BPE diretamente da memória ---")

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=1,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
)

# Esta função entrega os dados em lotes, diretamente da lista 'all_texts'
def batch_iterator(batch_size=1000):
    for i in range(0, len(all_texts), batch_size):
        yield all_texts[i : i + batch_size]

# Treina a partir do iterador, evitando completamente a escrita/leitura de arquivos
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(all_texts))

# Salva os arquivos brutos (vocab.json e merges.txt) na pasta de saída
tokenizer.model.save(output_dir)
print("✅ Tokenizer BPE treinado e salvo!\n")

# ==============================================================================
# 3. CONVERSÃO PARA O FORMATO HUGGINGFACE TRANSFORMERS
# ==============================================================================
print("--- 3. Convertendo para o formato Transformers ---")

vocab_file = os.path.join(output_dir, "vocab.json")
merges_file = os.path.join(output_dir, "merges.txt")

hf_tokenizer = RobertaTokenizerFast(
    vocab_file=vocab_file,
    merges_file=merges_file,
    bos_token="<s>",
    eos_token="</s>",
    sep_token="</s>",
    cls_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

hf_tokenizer.save_pretrained(output_dir)
print(f"✅ Tokenizer convertido e salvo no formato HuggingFace em: {output_dir}\n")

# ==============================================================================
# 4. VALIDAÇÃO FINAL
# ==============================================================================
print("--- 4. Validação Final do Tokenizer ---")

loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)

text = "Atenção: a votação começará em breve."
tokens = loaded_tokenizer.tokenize(text)

print(f"Texto de teste: '{text}'")
print(f"Tokens gerados: {tokens}")

corrupted_chars = ["Ã", "¡", "©", "§", "¨", "£"]
is_corrupted = any(char in token for token in tokens for char in corrupted_chars)

if not is_corrupted:
    print("\n✅ SUCESSO! Nenhum símbolo de codificação estranho foi encontrado.")
else:
    print("\n❌ FALHA! Símbolos de codificação estranhos ainda estão presentes.")
