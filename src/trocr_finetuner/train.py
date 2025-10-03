# src/trocr_finetuner/train.py
import logging

from peft import PeftModel
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          default_data_collator)

from . import config, dataset_loader, metrics, model_loader

# --- Configuração de Log ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Iniciando o fluxo de fine-tuning do TrOCR com LoRA ---")

    # 1. Carregar modelo e processador
    lora_model, processor, base_model = model_loader.get_lora_model_and_processor()

    lora_model.to(config.DEVICE)

    # 2. Carregar e preparar o dataset
    train_dataset, eval_dataset = dataset_loader.load_and_prepare_dataset(processor)

    # 3. Definir argumentos de treinamento
    training_args = Seq2SeqTrainingArguments( # <-- Voltamos a usar esta output_dir=config.LORA_CHECKPOINT_DIR,
        # Argumentos Essenciais para Seq2Seq
        predict_with_generate=True,

        # Nova Estratégia: Por Época
        eval_strategy="epoch",
        save_strategy="epoch",


        dataloader_pin_memory=False,
        warmup_ratio=0.1,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        generation_max_length=64,
        generation_num_beams=1,
        generation_confifrom tokenizers import Tokenizerg=lora_model.generation_config,

        # Hiperparâmetros
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        fp16=False,
        max_grad_norm=1.0,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,


        # Logging e Melhor Modelo
        logging_strategy="epoch", # Loga no final de cada época
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="tensorboard",
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # 4. Instanciar o Trainer
    trainer = Seq2SeqTrainer(
        model=lora_model,
        tokenizer=processor.tokenizer,
        args=training_args,
        compute_metrics=lambda p: metrics.compute_metrics(p, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 5. Iniciar o treinamento
    logger.info("Iniciando o treinamento...")
    trainer.train()
    logger.info("Treinamento concluído.")

    logger.info("Mesclando o modelo LoRA com o modelo base e salvando...")

    # Pega o caminho do melhor checkpoint que o Trainer salvou
    best_checkpoint_path = trainer.state.best_model_checkpoint
    if best_checkpoint_path is None:
        logger.warning("Não foi encontrado o melhor checkpoint. Salvando o último modelo.")
        # Se não houver "melhor", apenas use o modelo atual que já foi carregado
        final_model = lora_model
    else:
        logger.info(f"Carregando o melhor modelo do checkpoint: {best_checkpoint_path}")
        # Carrega os adaptadores LoRA do melhor checkpoint
        final_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)

    # Mescla os pesos
    final_model = final_model.merge_and_unload()

    logger.info(f"Salvando o modelo final e o processador em: {config.FINAL_MODEL_DIR}")
    final_model.save_pretrained(config.FINAL_MODEL_DIR)
    processor.save_pretrained(config.FINAL_MODEL_DIR)

    logger.info("--- Fluxo de fine-tuning concluído com sucesso! ---")

if __name__ == "__main__":
    main()
