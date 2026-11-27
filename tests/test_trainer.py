from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import ModelConfig
from src.model.transformer import DecoderOnlyLM
from src.training.trainer import TrainConfig, Trainer, cosine_lr_with_warmup


def test_cosine_lr_schedule():
    assert cosine_lr_with_warmup(0, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 0.0
    assert cosine_lr_with_warmup(10, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 1e-3
    assert cosine_lr_with_warmup(100, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4) == 1e-4


def test_trainer_decreases_loss_and_roundtrips_checkpoint(tmp_path: Path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, block_size=32, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)
    model = DecoderOnlyLM(cfg)

    x = torch.randint(0, 64, (8, 16))
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = -100
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    tcfg = TrainConfig(
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=2,
        max_steps=8,
        grad_accum_steps=1,
        amp_dtype="float32",
        eval_interval=4,
        save_interval=4,
        out_dir=tmp_path / "ckpt",
    )
    trainer = Trainer(model, tcfg, loader, val_loader=loader, causal_shift=False)
    history = trainer.train()
    assert history["train_loss"][-1] < history["train_loss"][0]
    ckpt_file = tmp_path / "ckpt" / "step_000008.pt"
    assert ckpt_file.exists()
    log_file = tmp_path / "ckpt" / "training_log.jsonl"
    assert log_file.exists()
    assert len(log_file.read_text(encoding="utf-8").strip().splitlines()) == 8

    model2 = DecoderOnlyLM(cfg)
    trainer2 = Trainer(model2, tcfg, loader, causal_shift=False)
    loaded_step = trainer2.load_checkpoint(ckpt_file)
    assert loaded_step == 8
