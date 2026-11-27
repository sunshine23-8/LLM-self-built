from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import signal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.transformer import DecoderOnlyLM


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    amp_dtype: str = "float16"
    gradient_checkpointing: bool = False
    eval_interval: int = 100
    save_interval: int = 200
    log_interval: int = 20
    enable_tensorboard: bool = True
    out_dir: Path = Path("artifacts/checkpoints")


def cosine_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor, causal_shift: bool) -> torch.Tensor:
    if causal_shift:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


class Trainer:
    def __init__(
        self,
        model: DecoderOnlyLM,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        causal_shift: bool = False,
    ) -> None:
        self.model = model
        self.cfg = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.causal_shift = causal_shift
        self.device = next(model.parameters()).device
        self.model.gradient_checkpointing = train_config.gradient_checkpointing

        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_config.lr,
            betas=train_config.betas,
        )

        self.amp_torch_dtype = getattr(torch, train_config.amp_dtype) if hasattr(torch, train_config.amp_dtype) else torch.float32
        self.use_amp = self.device.type == "cuda" and self.amp_torch_dtype in {torch.float16, torch.bfloat16}
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_torch_dtype == torch.float16))

    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": asdict(self.model.config),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt["step"])

    @torch.inference_mode()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        losses: list[float] = []
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = _compute_loss(logits, y, causal_shift=self.causal_shift)
            losses.append(float(loss.item()))
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def train(self, start_step: int = 0) -> dict[str, list[float]]:
        self.model.train()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        data_iter = iter(self.train_loader)

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.cfg.out_dir / "training_log.jsonl"
        log_mode = "a" if start_step > 0 and log_path.exists() else "w"

        tb_writer = None
        if self.cfg.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(log_dir=str(self.cfg.out_dir / "tb_logs"))
            except ImportError:
                tb_writer = None

        last_completed_step = start_step
        last_saved_step = start_step

        def _handle_sigterm(signum: int, frame: object) -> None:
            raise KeyboardInterrupt

        prev_sigterm = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
        if prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, _handle_sigterm)
            except ValueError:
                prev_sigterm = None

        try:
            with log_path.open(log_mode, encoding="utf-8") as log_file:
                for step in range(start_step + 1, self.cfg.max_steps + 1):
                    lr = cosine_lr_with_warmup(step, self.cfg.warmup_steps, self.cfg.max_steps, self.cfg.lr, self.cfg.min_lr)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                    self.optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    for _ in range(self.cfg.grad_accum_steps):
                        try:
                            x, y = next(data_iter)
                        except StopIteration:
                            data_iter = iter(self.train_loader)
                            x, y = next(data_iter)
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.use_amp):
                            logits = self.model(x)
                            loss = _compute_loss(logits, y, causal_shift=self.causal_shift) / self.cfg.grad_accum_steps
                        self.scaler.scale(loss).backward()
                        accum_loss += float(loss.item())

                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    last_completed_step = step
                    history["train_loss"].append(accum_loss)

                    record: dict[str, float | int] = {"step": step, "train_loss": accum_loss, "lr": lr}
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", accum_loss, step)
                        tb_writer.add_scalar("train/lr", lr, step)

                    if step % self.cfg.log_interval == 0 or step == 1:
                        print(f"Step {step}/{self.cfg.max_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e}")

                    if self.val_loader is not None and step % self.cfg.eval_interval == 0:
                        val_loss = self.evaluate()
                        history["val_loss"].append(val_loss)
                        record["val_loss"] = val_loss
                        if tb_writer is not None:
                            tb_writer.add_scalar("val/loss", val_loss, step)
                        print(f"--> Step {step} Validation Loss: {val_loss:.4f}")

                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()

                    if step % self.cfg.save_interval == 0 or step == self.cfg.max_steps:
                        save_path = self.cfg.out_dir / f"step_{step:06d}.pt"
                        self.save_checkpoint(save_path, step=step)
                        last_saved_step = step
                        print(f"Saved checkpoint to {save_path}")
        except KeyboardInterrupt:
            if last_completed_step > last_saved_step:
                interrupt_path = self.cfg.out_dir / f"step_{last_completed_step:06d}.pt"
                self.save_checkpoint(interrupt_path, step=last_completed_step)
                print(f"\nTraining interrupted! Saved checkpoint at step {last_completed_step} to {interrupt_path}")
            else:
                print(f"\nTraining interrupted at step {last_completed_step} (already saved).")
        finally:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if tb_writer is not None:
                tb_writer.close()

        return history
