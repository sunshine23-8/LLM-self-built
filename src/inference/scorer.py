from decimal import Decimal, InvalidOperation
import operator
import re
import torch
import torch.nn as nn

from src.eval.gsm8k_eval import extract_gsm8k_prediction
from src.model.transformer import DecoderOnlyLM

_EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def safe_check_arithmetic_steps(think_text: str) -> tuple[int, int]:
    matches = _EQ_RE.findall(think_text)
    valid = 0
    total = 0
    for lhs_s, op_s, rhs_s, res_s in matches:
        try:
            a, b, expected = Decimal(lhs_s), Decimal(rhs_s), Decimal(res_s)
            if op_s == "/" and b == 0:
                total += 1
                continue
            computed = _OPS[op_s](a, b)
            total += 1
            if abs(computed - expected) <= Decimal("1e-3"):
                valid += 1
        except (InvalidOperation, ZeroDivisionError):
            total += 1
    return valid, total


class HeuristicCoTScorer:
    """Self-written P2 scorer combining format compliance, arithmetic step consistency, and length-norm logprob."""

    def __init__(self, format_weight: float = 1.5, arith_weight: float = 1.0, logprob_weight: float = 0.5) -> None:
        self.format_weight = format_weight
        self.arith_weight = arith_weight
        self.logprob_weight = logprob_weight

    def score(self, full_text: str, mean_logprob: float) -> float:
        pred, strict_ok, src = extract_gsm8k_prediction(full_text)
        fmt_score = 1.0 if strict_ok else (0.4 if pred is not None and src != "none" else 0.0)
        think_body = full_text.split("<think>", 1)[1].split("</think>", 1)[0] if "<think>" in full_text else full_text
        valid_eqs, total_eqs = safe_check_arithmetic_steps(think_body)
        arith_score = (valid_eqs / total_eqs) if total_eqs > 0 else 0.5
        return (
            self.format_weight * fmt_score
            + self.arith_weight * arith_score
            + self.logprob_weight * float(mean_logprob)
        )


def get_final_answer_token_indices(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    non_pad = input_ids != pad_token_id
    last_non_pad = non_pad.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
    is_eos = input_ids[batch_idx, last_non_pad] == eos_token_id
    return torch.where(is_eos & (last_non_pad > 0), last_non_pad - 1, last_non_pad)


class OutcomeVerifierModel(nn.Module):
    """P3 Learned Outcome Reward Model pooling the final answer token hidden state."""

    def __init__(self, backbone: DecoderOnlyLM) -> None:
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(backbone.config.d_model, 1)

    def forward(self, input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor:
        _, hidden = self.backbone.forward_with_hidden(input_ids)
        pool_idx = get_final_answer_token_indices(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = hidden[batch_idx, pool_idx, :]
        return self.score_head(final_hidden).squeeze(-1)

    @torch.inference_mode()
    def score_sequences(self, input_ids: torch.Tensor, pad_token_id: int = 0, eos_token_id: int = 2) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(input_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id))
