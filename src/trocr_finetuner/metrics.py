# metrics.py
import logging

import evaluate
import numpy as np

logger = logging.getLogger(__name__)

# Inicializa métricas CER e WER
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(eval_pred, processor):
    """Recebe EvalPrediction do Trainer e o processor do TrOCR.
    Retorna dict com CER e WER.
    """
    predictions, labels = eval_pred

    # Se predictions vier como logits
    if isinstance(predictions, tuple) or predictions.ndim > 2:
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = np.argmax(predictions, axis=-1)

    # Corrige labels (-100 -> pad_token_id)
    labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

    # Corrige predictions (se por algum bug vier -100 → troca por pad também)
    predictions = np.where(predictions == -100, processor.tokenizer.pad_token_id, predictions)

    # Decodifica
    pred_texts = processor.batch_decode(predictions, skip_special_tokens=True)
    label_texts = processor.batch_decode(labels, skip_special_tokens=True)

    # Remove None/vazios
    pred_texts = [t if t is not None else "" for t in pred_texts]
    label_texts = [t if t is not None else "" for t in label_texts]

    # Calcula métricas
    cer = cer_metric.compute(predictions=pred_texts, references=label_texts)
    wer = wer_metric.compute(predictions=pred_texts, references=label_texts)

    logger.info(f"CER: {cer:.4f}, WER: {wer:.4f}")
    return {"cer": cer, "wer": wer}
