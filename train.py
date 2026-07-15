"""AdamW trainer with warm-up, cosine decay, and gradient clipping."""

import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod


MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_learning_rate(
    step,
    total_steps,
    peak_lr,
    warmup_steps,
    min_lr_ratio,
):
    min_lr = peak_lr * min_lr_ratio

    if step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)

    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / (
        total_steps - warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (peak_lr - min_lr)


def build_optimizer(model, lr, weight_decay):
    decay = []
    no_decay = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    parameter_groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="ckpt.pt")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    assert args.steps <= MAX_STEPS
    assert 0 <= args.warmup_steps < args.steps
    assert 0 < args.min_lr_ratio <= 1
    assert args.grad_clip > 0

    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tokenizer = tokenizer_mod.load()
    ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    print(
        f"corpus: {len(text.encode('utf-8')):,} bytes -> "
        f"{len(ids):,} tokens "
        f"(vocab {tokenizer.vocab_size})"
    )

    config = Config()
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)
    parameter_count = model.n_params()

    print(f"model: {parameter_count:,} params")
    assert parameter_count <= MAX_PARAMS

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    started = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        learning_rate = get_learning_rate(
            step=step,
            total_steps=args.steps,
            peak_lr=args.lr,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        x, y = get_batch(
            ids,
            config.block_size,
            args.batch,
            device,
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % args.log_every == 0:
            recent = losses[-args.log_every:]
            average_loss = sum(recent) / len(recent)
            elapsed_ms = (
                (time.time() - started) / step * 1000
            )

            print(
                f"step {step:5d}  "
                f"loss {average_loss:.4f}  "
                f"lr {learning_rate:.2e}  "
                f"({elapsed_ms:.0f} ms/step)"
            )

    checkpoint = {
        "model": model.state_dict(),
        "config": {
            key: getattr(config, key)
            for key in dir(config)
            if not key.startswith("_")
            and not callable(getattr(config, key))
        },
        "steps": args.steps,
        "train_loss_curve": losses,
    }

    torch.save(checkpoint, args.out)

    print(
        f"saved {args.out} "
        f"({time.time() - started:.0f}s total)"
    )


if __name__ == "__main__":
    main()
