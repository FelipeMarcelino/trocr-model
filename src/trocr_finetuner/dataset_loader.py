# src/trocr_finetuner/dataset_loader.py
import logging
import os

import torch
from datasets import Image as dsImage
from datasets import load_dataset
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(processor):
    """Carrega o dataset a partir do CSV e prepara para o treinamento."""
    csv_path = os.path.join(config.DATA_DIR, config.CSV_FILENAME)
    logger.info(f"Carregando dataset de: {csv_path}")

    dataset = load_dataset("csv", data_files=csv_path, delimiter=";", split="train")

    def resolve_image_path(example):
        image_path = os.path.join(config.DATA_DIR, config.IMAGE_DIR_NAME, example["image"])
        example["image"] = image_path
        return example

    dataset = dataset.map(resolve_image_path)
    dataset = dataset.cast_column("image", dsImage())

    logger.info("Dividindo dataset em treino e teste (90/10)...")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # --- FUNÇÃO DE PRÉ-PROCESSAMENTO CORRIGIDA E ROBUSTA ---
    def preprocess_function(examples):
        # 1. Garante que TODAS as imagens tenham o mesmo tamanho
        target_size = (384, 384)
        resized_images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in examples["image"]]

        # 2. Processa as imagens já redimensionadas
        pixel_values = processor(images=resized_images, return_tensors="pt").pixel_values

        # 3. Processa os textos
        labels = processor.tokenizer(text=examples["text"], padding="max_length", max_length=64, truncation=True).input_ids

        # 4. Substitui o pad_token_id por -100
        labels_as_list = [[l if l != processor.tokenizer.pad_token_id else -100 for l in label] for label in labels]

        # 5. CONVERTE A LISTA DE LABELS DE VOLTA PARA UM TENSOR
        labels_as_tensor = torch.tensor(labels_as_list)

        return {"pixel_values": pixel_values, "labels": labels_as_tensor}

    logger.info("Aplicando pré-processamento em todo o dataset com .map()...")

    processed_train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    ).with_format("torch")

    processed_eval_dataset = dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
    ).with_format("torch")


    logger.info("Dataset preparado e pré-processado com sucesso.")
    return processed_train_dataset, processed_eval_dataset
