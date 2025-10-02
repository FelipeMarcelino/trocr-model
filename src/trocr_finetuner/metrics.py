# metrics.py
import logging

import evaluate
import torch

logger = logging.getLogger(__name__)

# Inicializa métricas CER e WER
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(eval_pred, processor):
    """Recebe EvalPrediction do Trainer e o processor do TrOCR.
    Retorna dict com CER e WER.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # Se estiverem logits, pega argmax
    if isinstance(predictions, tuple) or predictions.ndim > 2:
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = torch.argmax(torch.tensor(predictions), dim=-1).numpy()

    # Substitui tokens -100 pelo pad_token_id antes de decodificar
    labels = torch.where(torch.tensor(labels) == -100,
                         processor.tokenizer.pad_token_id,
                         torch.tensor(labels)).numpy()

    # Decodifica predições e labels
    pred_texts = processor.batch_decode(predictions, skip_special_tokens=True)
    label_texts = processor.batch_decode(labels, skip_special_tokens=True)

    # Remove possíveis None ou strings vazias
    pred_texts = [t if t is not None else "" for t in pred_texts]
    label_texts = [t if t is not None else "" for t in label_texts]

    # Calcula CER e WER
    cer = cer_metric.compute(predictions=pred_texts, references=label_texts)
    wer = wer_metric.compute(predictions=pred_texts, references=label_texts)

    logger.info(f"CER: {cer:.4f}, WER: {wer:.4f}")
    return {
        "cer": cer,
        "wer": wer,
    }
