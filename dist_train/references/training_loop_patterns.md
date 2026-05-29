# Training Loop

## Pretrain Loop with Gradient Accumulation

```python
class Trainer:
    def __init__(self, config):
        self.step = 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.is_main_process = dist.get_rank() == 0
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        set_seed(config.seed + dist.get_rank())

        # AC -> FSDP -> move frozen parts
        apply_ac(model)
        hsdp_mesh = build_hsdp_mesh() if dist.get_world_size() > local_world_size else None
        model = fsdp_wrap_v2(model, hsdp_mesh=hsdp_mesh)

        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        self.dataloader = DataLoader(dataset, batch_size=config.bs, sampler=sampler,
            num_workers=config.num_workers, pin_memory=True)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay, fused=True, foreach=False)
        self.lr_scheduler = LambdaLR(self.optimizer,
            lr_lambda=lambda s: min(s / max(1, config.warmup_steps), 1.0))
        self.grad_accum = config.gradient_accumulation_steps

    def train_one_epoch(self):
        self.optimizer.zero_grad()
        acc_losses = []
        for i, batch in enumerate(self.dataloader):
            should_sync = (i+1) % self.grad_accum == 0 or (i+1) == len(self.dataloader)
            self.model.set_requires_gradient_sync(should_sync)

            batch = {k: v.to(self.device, dtype=self.dtype) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss = self.model.compute_loss(batch) / self.grad_accum
            loss.backward()
            acc_losses.append(loss.detach())
            del batch

            if should_sync:
                grad_norm = clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1

                mean_loss = dist_mean(torch.stack(acc_losses).sum()).cpu().item()
                acc_losses = []

                if self.step % config.gc_interval == 0:
                    gc.collect(); torch.cuda.empty_cache()
                if config.save_ckpt and self.step % config.save_interval == 0:
                    self.save()
                if self.is_main_process and self.step % config.log_interval == 0:
                    log.info(f"[Step {self.step}] loss={mean_loss:.4f} grad={grad_norm:.4f} lr={self.lr_scheduler.get_last_lr()[0]:.2e}")

    def train(self):
        while self.step < config.num_train_steps:
            self.train_one_epoch()
            dist.barrier()
```

## Gradient Accumulation

`set_requires_gradient_sync(False)` skips all-reduce during backward for non-sync steps (FSDP v2 equivalent of DDP `no_sync()`).

1. Accumulate for N micro-batches with sync disabled
2. Enable sync on last micro-batch
3. Clip -> step -> schedule -> zero_grad

## Evaluation with Distributed Gather

```python
def evaluate(self):
    self.model.eval()
    # ... inference ...
    if dist.get_world_size() > 1:
        gathered = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, pred)
        pred = torch.cat(gathered)
    if self.is_main_process:
        save(pred)
    dist.barrier()
    self.model.train()
```

## LR Schedule

```python
def warmup_constant(step, warmup_steps=1000):
    return min(step / max(1, warmup_steps), 1.0)
```
