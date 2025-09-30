import logging

from peft import LoraConfig, get_peft_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import config

logger = logging.getLogger(__name__)

def get_lora_model_and_processor():
    """Carrega o modelo TrOCR base e aplica a configuração LoRA."""
    logger.info(f"Carregando processador e modelo base: {config.BASE_MODEL_ID}")
    processor = TrOCRProcessor.from_pretrained(config.BASE_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(config.BASE_MODEL_ID)

    # Configura o modelo para o fine-tuning
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Define a configuração LoRA
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    logger.info("Aplicando adaptação LoRA ao modelo...")
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    return lora_model, processor
