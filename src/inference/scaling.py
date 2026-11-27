from collections import defaultdict
from dataclasses import dataclass
import time
import torch

from src.data.cot_dataset import format_cot_prompt
from src.eval.gsm8k_eval import extract_gsm8k_prediction, is_numeric_equal, normalize_numeric_str
from src.inference.generator import AutoregressiveGenerator
from src.inference.scorer import HeuristicCoTScorer, OutcomeVerifierModel
from src.tokenizer.bpe import ByteLevelBPETokenizer


@dataclass
class CandidateChain:
    text: str
    think_text: str
    extracted_answer: str | None
    strict_format_valid: bool
    extraction_source: str
    think_tokens: int
    generated_tokens: int
    mean_logprob: float
    heuristic_score: float
    orm_score: float | None = None


@dataclass
class PoolEvalRecord:
    question: str
    gold_answer: str
    greedy_answer: str | None
    greedy_correct: bool
    greedy_tokens: int
    greedy_forward_passes: int
    pool: list[CandidateChain]
    maj_at_4: bool
    maj_at_8: bool
    maj_at_16: bool
    bon_at_8_heuristic: bool
    bon_at_16_heuristic: bool
    bon_at_8_orm: bool | None
    bon_at_16_orm: bool | None
    weighted_maj_at_16: bool | None
    oracle_pass_at_16: bool
    pool_generated_tokens: int
    pool_forward_passes: int
    pool_wall_ms: float


def _canonical_vote_key(ans: str | None) -> str | None:
    if ans is None:
        return None
    dec = normalize_numeric_str(ans)
    return str(dec.normalize()) if dec is not None else ans


def majority_vote(candidates: list[CandidateChain]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    best_lp: dict[str, float] = defaultdict(lambda: float("-inf"))
    rep_str: dict[str, str] = {}
    for c in candidates:
        key = _canonical_vote_key(c.extracted_answer)
        if key is None:
            continue
        counts[key] += 1
        if c.mean_logprob > best_lp[key]:
            best_lp[key] = c.mean_logprob
            rep_str[key] = c.extracted_answer  # type: ignore[assignment]
    if not counts:
        return None
    winner = max(counts.keys(), key=lambda k: (counts[k], best_lp[k]))
    return rep_str[winner]


def weighted_majority_vote(candidates: list[CandidateChain], score_attr: str = "orm_score") -> str | None:
    weights: dict[str, float] = defaultdict(float)
    rep_str: dict[str, str] = {}
    for c in candidates:
        key = _canonical_vote_key(c.extracted_answer)
        if key is None:
            continue
        score = getattr(c, score_attr)
        weights[key] += float(score) if score is not None else 0.0
        if key not in rep_str:
            rep_str[key] = c.extracted_answer  # type: ignore[assignment]
    if not weights:
        return None
    winner = max(weights.keys(), key=lambda k: weights[k])
    return rep_str[winner]


class TestTimeScaler:
    def __init__(
        self,
        generator: AutoregressiveGenerator,
        tokenizer: ByteLevelBPETokenizer,
        heuristic_scorer: HeuristicCoTScorer | None = None,
        orm_model: OutcomeVerifierModel | None = None,
        max_batch_k: int = 8,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.heuristic = heuristic_scorer or HeuristicCoTScorer()
        self.orm = orm_model
        self.max_batch_k = max_batch_k
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        self.eos_id = tokenizer.token_to_id("<|eos|>")

    def _to_candidate(self, full_ids: torch.Tensor, prompt_len: int, logprobs: torch.Tensor) -> CandidateChain:
        gen_ids = [i for i in full_ids[prompt_len:].tolist() if i != self.pad_id]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        pred, strict_ok, src = extract_gsm8k_prediction(text)

        think_text = ""
        think_tokens = 0
        if "<think>" in text:
            after_open = text.split("<think>", 1)[1]
            think_text = after_open.split("</think>", 1)[0] if "</think>" in after_open else after_open
            think_tokens = len(self.tokenizer.encode(think_text))

        active_lp = logprobs[: len(gen_ids)]
        mean_lp = float(active_lp.mean().item()) if active_lp.numel() > 0 else -10.0
        h_score = self.heuristic.score(text, mean_lp)

        orm_score: float | None = None
        if self.orm is not None:
            seq_tensor = full_ids.unsqueeze(0)
            orm_score = float(self.orm.score_sequences(seq_tensor, pad_token_id=self.pad_id, eos_token_id=self.eos_id).item())

        return CandidateChain(
            text=text,
            think_text=think_text,
            extracted_answer=pred,
            strict_format_valid=strict_ok,
            extraction_source=src,
            think_tokens=think_tokens,
            generated_tokens=len(gen_ids),
            mean_logprob=mean_lp,
            heuristic_score=h_score,
            orm_score=orm_score,
        )

    def evaluate_question(
        self,
        question: str,
        gold_answer: str,
        k_pool: int = 16,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> PoolEvalRecord:
        prompt_str = format_cot_prompt(question)
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt_str)], dtype=torch.long)

        # E1: Deterministic Greedy baseline (1x compute)
        greedy_res = self.generator.generate(
            prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0, use_cache=True, stop_token_ids=[self.eos_id]
        )
        greedy_cand = self._to_candidate(greedy_res.sequences[0], greedy_res.prompt_len, greedy_res.token_logprobs[0])
        greedy_ok = is_numeric_equal(greedy_cand.extracted_answer, gold_answer)

        # Shared pool of K rollouts via chunked Copy KV Cache
        pool: list[CandidateChain] = []
        pool_tokens = 0
        pool_forwards = 0
        t0 = time.perf_counter()
        remaining = k_pool
        while remaining > 0:
            cur_k = min(remaining, self.max_batch_k)
            res = self.generator.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                stop_token_ids=[self.eos_id],
                num_return_sequences=cur_k,
            )
            pool_tokens += res.generated_tokens
            pool_forwards += res.forward_passes
            for row in range(cur_k):
                pool.append(self._to_candidate(res.sequences[row], res.prompt_len, res.token_logprobs[row]))
            remaining -= cur_k
        wall_ms = (time.perf_counter() - t0) * 1000.0

        maj4 = is_numeric_equal(majority_vote(pool[:4]), gold_answer)
        maj8 = is_numeric_equal(majority_vote(pool[:8]), gold_answer)
        maj16 = is_numeric_equal(majority_vote(pool[:16]), gold_answer)

        bon8_h = is_numeric_equal(max(pool[:8], key=lambda c: c.heuristic_score).extracted_answer, gold_answer)
        bon16_h = is_numeric_equal(max(pool[:16], key=lambda c: c.heuristic_score).extracted_answer, gold_answer)

        if self.orm is not None:
            bon8_orm = is_numeric_equal(max(pool[:8], key=lambda c: c.orm_score or 0.0).extracted_answer, gold_answer)
            bon16_orm = is_numeric_equal(max(pool[:16], key=lambda c: c.orm_score or 0.0).extracted_answer, gold_answer)
            wmaj16 = is_numeric_equal(weighted_majority_vote(pool[:16], score_attr="orm_score"), gold_answer)
        else:
            bon8_orm, bon16_orm, wmaj16 = None, None, None

        oracle16 = any(is_numeric_equal(c.extracted_answer, gold_answer) for c in pool[:16])

        return PoolEvalRecord(
            question=question,
            gold_answer=gold_answer,
            greedy_answer=greedy_cand.extracted_answer,
            greedy_correct=greedy_ok,
            greedy_tokens=greedy_cand.generated_tokens,
            greedy_forward_passes=greedy_res.forward_passes,
            pool=pool,
            maj_at_4=maj4,
            maj_at_8=maj8,
            maj_at_16=maj16,
            bon_at_8_heuristic=bon8_h,
            bon_at_16_heuristic=bon16_h,
            bon_at_8_orm=bon8_orm,
            bon_at_16_orm=bon16_orm,
            weighted_maj_at_16=wmaj16,
            oracle_pass_at_16=oracle16,
            pool_generated_tokens=pool_tokens,
            pool_forward_passes=pool_forwards,
            pool_wall_ms=wall_ms,
        )
