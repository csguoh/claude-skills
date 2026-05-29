# Checkpoint Management

## FSDP v1 Save

```python
def fsdp_state_dict(model):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()
```

## FSDP v2 Save (Recommended)

```python
def save(self):
    # Collective op - all ranks must call
    state_dict = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    # Optional: gather optimizer state for resume
    optim_state = get_optimizer_state_dict(self.model, self.optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True)) if self.use_resume else None

    if self.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        save_file({k: v.to(torch.bfloat16) for k, v in state_dict.items()},
                  os.path.join(ckpt_dir, "model.safetensors"))
        if optim_state is not None:
            torch.save({'optimizer_state_dict': optim_state, 'step': self.step,
                        'lr_scheduler_state': self.lr_scheduler.state_dict()},
                       os.path.join(ckpt_dir, "optimizer.pt"))
    dist.barrier()
```

Key: `get_model_state_dict` / `get_optimizer_state_dict` are collective ops. Cast to bf16 before saving. Always barrier even on failure.

## FSDP v2 Resume

```python
def resume(self, ckpt_path):
    state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"))
    param_shapes = {k: v.shape for k, v in state_dict.items()}

    # Collective op
    set_model_state_dict(self.model, state_dict,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    optim_file = os.path.join(ckpt_path, "optimizer.pt")
    if os.path.exists(optim_file):
        ckpt = torch.load(optim_file, map_location="cpu", weights_only=False)
        saved_state = ckpt['optimizer_state_dict'].get('state', {})
        # Fill default state for params missing from checkpoint (never had gradients)
        for name, shape in param_shapes.items():
            if name not in saved_state:
                saved_state[name] = {'step': torch.tensor(0.0),
                    'exp_avg': torch.zeros(shape), 'exp_avg_sq': torch.zeros(shape)}
        # Collective op
        set_optimizer_state_dict(self.model, self.optimizer, ckpt['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        self.step = ckpt['step']
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state'])
```

Handle missing optimizer keys by filling zero default state for params that never received gradients.

## Checkpoint Layout

```
save_path/
└── {timestamp}/
    ├── checkpoint_step_N/
    │   ├── model.safetensors
    │   └── optimizer.pt        # only when use_resume=True
    └── checkpoint_step_M/
```
