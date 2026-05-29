# Distributed Primitives

## Process Group Init

```python
def launch_distributed_job(backend="nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    init_method = f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=180))
    torch.cuda.set_device(local_rank)
```

## Reduction Utilities

```python
def dist_mean(t):
    if dist.is_initialized(): dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t

def dist_max(t):
    if dist.is_initialized(): dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t
```

## FSDP v1

```python
def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False,
              wrap_strategy="size", min_num_params=int(5e7),
              transformer_module=None, cpu_offload=False, ignored_modules=None):
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32, cast_forward_inputs=False) if mixed_precision else None
    if wrap_strategy == "transformer":
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_module)
    else:
        policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    strategy = {"full": ShardingStrategy.FULL_SHARD, "hybrid_full": ShardingStrategy.HYBRID_SHARD,
                "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2, "no_shard": ShardingStrategy.NO_SHARD}[sharding_strategy]
    return FSDP(module, auto_wrap_policy=policy, sharding_strategy=strategy, mixed_precision=mp,
                device_id=torch.cuda.current_device(), limit_all_gathers=True, use_orig_params=True,
                cpu_offload=CPUOffload(offload_params=cpu_offload), sync_module_states=False, ignored_modules=ignored_modules)
```

Key flags: `use_orig_params=True` (required for mixed precision), `limit_all_gathers=True` (prevents OOM).

## FSDP v2 (per-submodule `fully_shard`)

```python
def fsdp_wrap_v2(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32, hsdp_mesh=None):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)
    cfg = {"mp_policy": mp_policy, "reshard_after_forward": True}
    if hsdp_mesh is not None:
        cfg["mesh"] = hsdp_mesh

    for block in model.blocks:
        for sub in [block.attn, block.ffn]:  # adapt to your model's sub-module names
            fully_shard(sub, **cfg)
        fully_shard(block, **cfg)
    fully_shard(model, **cfg)
    return model
```

Shard sub-modules (attn, ffn) before parent block for finer communication control. `reshard_after_forward=True` frees params after forward to save memory.

## HSDP DeviceMesh

```python
def build_hsdp_mesh(num_gpus_per_node=8):
    return init_device_mesh("cuda",
        (dist.get_world_size() // num_gpus_per_node, num_gpus_per_node),
        mesh_dim_names=("replicate", "shard"))
```

Auto-detect: `hsdp_mesh = build_hsdp_mesh(n) if world_size > n else None`

## Activation Checkpointing

```python
def apply_ac(model):
    for i, block in enumerate(model.blocks):
        model.blocks[i] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
```

**Must call before FSDP wrapping.** `preserve_rng_state=False` for speed.

## EMA Under FSDP

```python
class EMA_FSDP:
    def __init__(self, fsdp_module, decay=0.999):
        self.decay = decay
        self.shadow = {}
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(self.decay).add_(p.float().cpu(), alpha=1-self.decay)

    def copy_to(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
```

`summon_full_params` is collective - all ranks must call.

## Seed

```python
if config.seed == 0:
    s = torch.randint(0, 10000000, (1,), device=device)
    dist.broadcast(s, src=0)
    config.seed = s.item()
set_seed(config.seed + global_rank)
```
