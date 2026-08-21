import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids=None,
    boost_value_ids=None,
    boost_ctx_id=None,
    boost_weight: float = 1.0,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight
        # position-aware recognition upweight: boost emotion value tokens that
        # occur in a [user ...] context (a ctx token within the previous W positions).
        if boost_value_ids and boost_ctx_id is not None and boost_weight != 1.0:
            isval = torch.zeros_like(target, dtype=torch.bool)
            for vid in boost_value_ids:
                isval |= (target == vid)
            isctx = (target == boost_ctx_id)
            near = torch.zeros_like(isctx)
            W = 4
            for k in range(1, W + 1):
                sh = torch.zeros_like(isctx)
                sh[..., k:] = isctx[..., :-k]
                near |= sh
            boost_mask = isval & near
            weights = torch.where(boost_mask, weights * boost_weight, weights)

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights)

    return mb_loss
