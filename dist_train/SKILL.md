---
name: dist-train
description: This skill should be used when the user asks to "set up distributed training", "configure FSDP", "add HSDP sharding", "enable activation checkpointing", "save/resume checkpoints in FSDP", "configure mixed precision training", "write a training loop with gradient accumulation", "launch a distributed job" via torchrun, "scaffold a new training project", or "set up dataset pipeline".
version: 0.1.0
---

# Distributed Training

PyTorch distributed training patterns: FSDP/HSDP, activation checkpointing, mixed precision, checkpoint save/resume, training loop.

## Setup Order

1. `launch_distributed_job()` - init process group from torchrun env vars
2. Build model
3. `apply_ac(model)` - activation checkpointing (**before** FSDP wrap)
4. `fsdp_wrap(model)` or `fsdp_wrap_v2(model, hsdp_mesh=...)` - FSDP/HSDP
5. Build optimizer on FSDP-wrapped parameters
6. Optionally `resume()` from checkpoint
7. Training loop

## FSDP Wrapping

- **v1 `fsdp_wrap`**: `FullyShardedDataParallel` wrapper with auto_wrap_policy (size or transformer). Strategies: `full`, `hybrid_full`, `hybrid_zero2`, `no_shard`.
- **v2 `fsdp_wrap_v2`**: `fully_shard` per-submodule API with `MixedPrecisionPolicy`. Pass `hsdp_mesh` for HSDP.

HSDP: build 2D DeviceMesh (replicate across nodes, shard within node). Auto-detect: use HSDP when `world_size > LOCAL_WORLD_SIZE`.

## Mixed Precision

bf16 params + fp32 reductions. Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`.

## Checkpoint

- **v1**: `FSDP.state_dict_type()` + `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`
- **v2**: `get_model_state_dict()` / `set_model_state_dict()` from `torch.distributed.checkpoint.state_dict`

All state dict ops are collective - all ranks must participate. Only rank 0 writes to disk.

## Launch

```bash
torchrun --nproc_per_node=8 train.py            # single node
torchrun --nnodes=N --nproc_per_node=8 train.py  # multi node
```

## References

- **`references/distributed_primitives.md`** - FSDP v1/v2, HSDP mesh, activation checkpointing, EMA, process group init
- **`references/checkpoint_management.md`** - Save/resume with optimizer state
- **`references/training_loop_patterns.md`** - Pretrain loop with gradient accumulation
- **`references/dataset_patterns.md`** - Dataset class hierarchy, composition, processor pipeline
- **`references/modeling_patterns.md`** - modeling/processing file templates, compute_loss pattern
- **`references/project_structure.md`** - Project directory layout conventions
