import torch

def beam_search_decode(log_probs: torch.Tensor, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: torch.Tensor of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    vocab_inds = torch.arange(log_probs.shape[1])
    timesteps = log_probs.shape[0]

    init_beams = torch.argsort(log_probs[0], descending = True)[:beam_width]
    init_log_probs = log_probs[0][init_beams]

    beams = init_beams.unsqueeze(1)
    curr_log_probs = init_log_probs

    for t in range(1, timesteps):
        expanded_log_probs = torch.cartesian_prod(curr_log_probs, log_probs[t])
        expanded_log_probs = torch.sum(expanded_log_probs, dim = 1)

        expanded_beams = torch.cat([
            beams.repeat_interleave(len(vocab_inds), dim = 0),
            vocab_inds.repeat(len(beams)).unsqueeze(1)
        ], dim=1)

        best = torch.argsort(expanded_log_probs, descending = True)[:beam_width]
        curr_log_probs = expanded_log_probs[best]
        beams = expanded_beams[best]

    return [
        (beam.tolist(), round(logprob.item(), 4))
        for beam, logprob in zip(beams, curr_log_probs)
    ][:top_k]




