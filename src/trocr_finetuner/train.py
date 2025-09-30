# src/trocr_finetuner/train.py
import logging

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

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
    lora_model, processor = model_loader.get_lora_model_and_processor()
    lora_model.to(config.DEVICE)

    # 2. Carregar e preparar o dataset
    train_dataset, eval_dataset = dataset_loader.load_and_prepare_dataset(processor)

    # 3. Definir argumentos de treinamento
    training_args = Seq2SeqTrainingArguments( # <-- Voltamos a usar esta output_dir=config.LORA_CHECKPOINT_DIR,
        # Argumentos Essenciais para Seq2Seq
        predict_with_generate=True,

        # Nova Estratégia: Por Época
        eval_strategy="steps",
        save_strategy="steps",


        generation_max_length=64,
        generation_num_beams=1,
        generation_config=lora_model.generation_config,

        # Hiperparâmetros
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        fp16=True,
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

    # 6. Salvar o modelo final para produção
    logger.info("Mesclando o modelo LoRA com o modelo base e salvando...")
    # Carrega o melhor checkpoint salvo durante o treinamento
    final_model = model_loader.PeftModel.from_pretrained(lora_model, training_args.output_dir + "/checkpoint-best/")
    # Mescla os pesos
    final_model = final_model.merge_and_unload()

    logger.info(f"Salvando o modelo final e o processador em: {config.FINAL_MODEL_DIR}")
    final_model.save_pretrained(config.FINAL_MODEL_DIR)
    processor.save_pretrained(config.FINAL_MODEL_DIR)

    logger.info("--- Fluxo de fine-tuning concluído com sucesso! ---")

if __name__ == "__main__":
    main()
