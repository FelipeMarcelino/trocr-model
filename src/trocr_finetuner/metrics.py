import evaluate
import numpy as np

cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):
    """Calcula a Taxa de Erro de Caractere (CER)."""
    # Extrai preds e labels
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Alguns trainers retornam logits em vez de IDs
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # Se os preds vierem em logits, pega o argmax
    if pred_ids.ndim == 3:  # (batch, seq_len, vocab_size)
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Decodificação
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Ajusta labels (substitui -100 pelo pad_token_id antes de decodificar)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
