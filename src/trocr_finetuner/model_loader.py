# src/trocr_finetuner/model_loader.py
import logging

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from . import config

logger = logging.getLogger(__name__)

def get_lora_model_and_processor():
    """Carrega o modelo base TrOCR, substitui seu tokenizador por um em português,
    ajusta o modelo e aplica a configuração LoRA.
    """
    # É recomendado usar um modelo "stage1" como base para fine-tuning em novos idiomas
    # Considere mudar o BASE_MODEL_ID no seu config.py para "microsoft/trocr-base-stage1"
    BASE_MODEL_ID = config.BASE_MODEL_ID
    TOKENIZER_ID = "neuralmind/bert-base-portuguese-cased"

    logger.info(f"Carregando processador de imagem de: {BASE_MODEL_ID}")
    # 1. Carrega o processador original, que usaremos para o processamento de imagem.
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)

    logger.info(f"Carregando novo tokenizador em português de: {TOKENIZER_ID}")
    # 2. Carrega o novo tokenizador que você deseja usar.
    new_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # 3. SUBSTITUI o tokenizador antigo dentro do objeto processor.
    processor.tokenizer = new_tokenizer

    logger.info(f"Carregando modelo base de: {BASE_MODEL_ID}")
    # 4. Carrega o modelo base.
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_ID)

    # --- ETAPA CRÍTICA: AJUSTAR O MODELO AO NOVO TOKENIZADOR ---
    logger.info("Redimensionando a camada de embedding do modelo para o novo vocabulário.")
    # 5. Redimensiona a camada de embedding do decodificador para o tamanho do novo vocabulário.
    # Isso descarta os pesos de embedding antigos (em inglês) e cria novos, prontos para serem treinados.
    model.decoder.resize_token_embeddings(len(new_tokenizer))

    logger.info("Ajustando a configuração do modelo para os novos tokens especiais.")
    # 6. Ajusta a configuração do modelo para usar os tokens especiais do novo tokenizador (estilo BERT).
    model.config._name_or_path = config.BASE_MODEL_ID
    model.config.decoder_start_token_id = new_tokenizer.cls_token_id
    model.config.pad_token_id = new_tokenizer.pad_token_id
    model.config.eos_token_id = new_tokenizer.sep_token_id
    model.config.vocab_size = len(new_tokenizer)
    model.config.decoder.vocab_size = len(new_tokenizer)



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

    return lora_model, processor, model
