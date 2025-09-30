# src/trocr_finetuner/dataset_loader.py
import logging
import os

from datasets import Image as dsImage
from datasets import load_dataset

from . import config

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(processor):
    """Carrega o dataset a partir do CSV e prepara para o treinamento."""
    csv_path = os.path.join(config.DATA_DIR, config.CSV_FILENAME)
    logger.info(f"Carregando dataset de: {csv_path}")

    # Carrega o CSV
    dataset = load_dataset("csv", data_files=csv_path, delimiter=";", split="train")

    # Mapeia os nomes de arquivo para os caminhos completos e carrega as imagens
    def resolve_image_path(example):
        image_path = os.path.join(config.DATA_DIR, config.IMAGE_DIR_NAME, example["image"])
        example["image"] = image_path
        return example

    dataset = dataset.map(resolve_image_path)
    dataset = dataset.cast_column("image", dsImage())

    logger.info("Dividindo dataset em treino e teste (90/10)...")
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Função de pré-processamento
    def preprocess_function(examples):
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values
        labels = processor.tokenizer(text=examples["text"], padding="max_length", max_length=64, truncation=True).input_ids
        # Importante: troca o pad_token_id por -100 para que o loss não seja calculado sobre ele
        labels = [[l if l != processor.tokenizer.pad_token_id else -100 for l in label] for label in labels]

        return {"pixel_values": pixel_values, "labels": labels}

    logger.info("Aplicando pré-processamento em todo o dataset com .map()...")

    processed_train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["image", "text"], # Remove as colunas originais que causam o erro
    )
    processed_eval_dataset = dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=["image", "text"], # Remove as colunas originais
    )

    logger.info("Dataset preparado e pré-processado com sucesso.")
    return processed_train_dataset, processed_eval_dataset
