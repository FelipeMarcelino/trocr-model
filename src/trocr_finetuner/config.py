import torch

# --- Caminhos ---
DATA_DIR = "data"
CSV_FILENAME = "labels.csv"
IMAGE_DIR_NAME = "images"

# --- Modelos ---
#BASE_MODEL_ID = "microsoft/trocr-base-handwritten"
BASE_MODEL_ID = "microsoft/trocr-base-stage1"

# Onde os checkpoints do LoRA serão salvos durante o treinamento
LORA_CHECKPOINT_DIR = "./trocr-lora-finetuned"
# Onde o modelo final, pronto para produção, será salvo
FINAL_MODEL_DIR = "./meu-trocr-final"

# --- Configurações de Treinamento ---
# Com 8GB de VRAM, um batch size pequeno é essencial.
# Se tiver erro de "Out of Memory", diminua para 2.
BATCH_SIZE = 4
NUM_EPOCHS = 50 # Para um dataset pequeno (~500 amostras), mais épocas são necessárias.
LEARNING_RATE = 5e-4

# --- Configurações LoRA ---
# Rank da adaptação. Valores comuns são 8, 16, 32.
LORA_R = 16
# Alpha é um parâmetro de escala. Comum usar 2*r.
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Módulos do modelo a serem adaptados. Para TrOCR, 'q_proj' e 'v_proj' são os principais.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# --- Ambiente ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
