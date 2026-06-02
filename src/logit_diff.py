def logit_diff(logits, io_tokens, s_tokens, mean=True):
    """
    Logit difference between IO and S at the final token position.
    Higher = model more confidently predicts IO over S.
    """
    final_logits = logits[:, -1, :]  # [batch, vocab]
    io_logits = final_logits[range(len(io_tokens)), io_tokens]
    s_logits = final_logits[range(len(s_tokens)), s_tokens]
    diff = io_logits - s_logits
    return diff.mean() if mean else diff
