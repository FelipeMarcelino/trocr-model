# src/trocr_finetuner/debug_data.py
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config, dataset_loader, model_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def verify_generation(model, dataset, device, batch_size,generation_config, dataset_name):
    logger.info(f"--- Iniciando verificação de GERAÇÃO no Dataset de {dataset_name} ---")
    data_loader = DataLoader(dataset, batch_size=batch_size)

    for i, batch in enumerate(tqdm(data_loader, desc=f"Verificando Geração em {dataset_name}")):
        try:
            pixel_values = batch["pixel_values"].to(device)
            # Não precisamos dos labels para a geração

            # --- MUDANÇA CRÍTICA AQUI ---
            # Simulamos o que o Trainer faz na avaliação com predict_with_generate=True
            # Este processo é mais sensível a instabilidades numéricas.
            with torch.no_grad():
                _ = model.generate(
                    pixel_values=pixel_values,
                    max_length=64, # Usamos os mesmos parâmetros do train.py
                    num_beams=1,
                )

        except Exception as e:
            logger.error(f"!!! ERRO DE GERAÇÃO ou INSTABILIDADE NUMÉRICA no lote nº {i} de {dataset_name}: {e} !!!")
            logger.error("Esta é a causa provável do seu OverflowError durante o treinamento.")
            logger.error(f"Verifique as imagens correspondentes a este lote no seu CSV original (aprox. linhas {i*batch_size} a {(i+1)*batch_size}).")
            return False

    logger.info(f"✅ Verificação de GERAÇÃO do Dataset de {dataset_name} concluída. Nenhum problema encontrado.")
    return True

def main():
    try:
        logger.info("Carregando dependências para verificação de dados...")
        lora_model, processor, model = model_loader.get_lora_model_and_processor()
        train_dataset, eval_dataset = dataset_loader.load_and_prepare_dataset(processor)

        device = torch.device(config.DEVICE)
        model.to(device)
        model.eval()

        # Usamos as configurações do config.py para garantir que o teste seja idêntico
        generation_config = getattr(config, "generation_config", {})
        if "max_length" not in generation_config:
            generation_config["max_length"] = 64
        if "num_beams" not in generation_config:
            generation_config["num_beams"] = 1

        train_ok = verify_generation(lora_model, train_dataset, device, config.BATCH_SIZE, generation_config, "Treinamento")

        if train_ok:
            verify_generation(lora_model, eval_dataset, device, config.BATCH_SIZE, generation_config, "Validação")

    except Exception as e:
        logger.error(f"Falha ao inicializar o processo de depuração: {e}", exc_info=True)

if __name__ == "__main__":
    main()
