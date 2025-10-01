# model_loader.py
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

    # --- CORREÇÃO DEFINITIVA ---
    # A função de salvamento da PEFT precisa saber o ID do modelo original
    # para recarregar a config base. Essa informação fica "escondida"
    # nas configs do encoder/decoder. Nós a colocamos no nível superior.
    model.config._name_or_path = config.BASE_MODEL_ID

    # Mantenha estas configurações que são importantes para a geração
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # A linha abaixo não é mais necessária aqui, pois a correção acima é mais fundamental
    # model.config.vocab_size = model.config.decoder.vocab_size

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

    return lora_model, processor, model
