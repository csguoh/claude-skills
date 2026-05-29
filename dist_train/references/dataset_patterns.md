# Dataset Patterns

## Architecture

```
proj_name_datasets/
├── base.py              # BaseDataset abstract class
├── source_a.py          # concrete dataset per data source
├── source_b.py
├── composed.py          # ComposedDataset: weighted multi-dataset mixing
└── __init__.py          # build_dataset(), collate_fn, registry imports

├── processor.py         # Processor: raw __getitem__ output -> model-ready tensors
```

## Base Dataset

```python
class BaseDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(self, data_root, split=None, processor=None, **kwargs):
        self.data_root = data_root
        self.processor = processor

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __getitem__(self, idx) -> dict: ...

    def _preprocess(self, **raw_fields):
        if self.processor is None:
            return raw_fields
        return self.processor(**raw_fields, return_tensors="pt")
```

`__getitem__` loads raw data, then calls `_preprocess` which delegates to `processor` for tokenization/normalization/tensor conversion.

## Concrete Dataset

```python
class MyDataset(BaseDataset):
    def __init__(self, data_root, split=None, processor=None, **kwargs):
        super().__init__(data_root, split=split, processor=processor, **kwargs)
        self.storage = MyStorage(data_root, split=split)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, idx):
        raw = self.storage[idx]
        return self._preprocess(**raw)
```

Separate I/O (storage) from data transforms (processor). Storage uses context manager for file handle lifecycle:

```python
class MyStorage:
    @contextmanager
    def open(self, episode_index):
        handle = h5py.File(self.paths[episode_index], 'r')
        try:
            yield handle
        finally:
            handle.close()
```

`__getitem__` opens and closes handles per call. Never hold file handles as instance state — worker processes fork and inherited handles leak or deadlock.

## Processor

Processor handles data normalization, tokenization, image resize等, and returns `BatchFeature`. Called by `BaseDataset._preprocess` at the end of `__getitem__`.

```python
class MyProcessor(ProcessorMixin):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, text, images, actions, states, **kwargs):
        text_inputs = self.tokenizer(text, padding="max_length", truncation=True)
        images = self._resize_to_tensor(images)
        actions = self._normalize(actions)
        states = self._normalize(states)
        return BatchFeature(data={
            "images": images, "actions": actions,
            "states": states, **text_inputs,
        }, tensor_type=kwargs.get("return_tensors"))

    def post_process(self, model_output):
        """Inference: denormalize model output back to real space."""
        return self._denormalize(model_output)
```

## Composed Dataset

```python
class ComposedDataset(Dataset):
    def __init__(self, datasets: List[BaseDataset], sample_weights: Optional[List[float]] = None):
        self._datasets = datasets
        if sample_weights:
            total = sum(sample_weights)
            normed = [w / total for w in sample_weights]
            max_virtual = max(len(d) / normed[i] for i, d in enumerate(datasets))
            self._virtual_sizes = [int(max_virtual * w) for w in normed]
        else:
            self._virtual_sizes = [len(d) for d in datasets]

    def __len__(self):
        return sum(self._virtual_sizes)

    def __getitem__(self, idx):
        dataset_id, local_idx = self._resolve(idx)
        return self._datasets[dataset_id][local_idx % len(self._datasets[dataset_id])]
```

Weighted sampling via virtual sizes: smaller datasets get oversampled (modulo wrap), larger datasets get undersampled. Compatible with `DistributedSampler(shuffle=True)`.

## Registry & Builder (`__init__.py` template)

```python
from .base import BaseDataset
from .composed import ComposedDataset
from .source_a import SourceADataset
from .source_b import SourceBDataset

class Registry:
    def __init__(self):
        self._objects = {}

    def register(self, name=None):
        def decorator(cls):
            self._objects[cls.__name__ if name is None else name] = cls
            return cls
        return decorator

    def __getitem__(self, name):
        return self._objects[name]

DATASET_REGISTRY = Registry()

def collate_fn(instances):
    batch = {}
    for key in {k for inst in instances for k in inst}:
        vals = [inst[key] for inst in instances]
        batch[key] = torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
    return batch

def build_dataset(data_types, data_paths, processor=None, sample_weights=None, **kwargs):
    datasets = [DATASET_REGISTRY[dt](data_root=dp, processor=processor, **kwargs)
                for dt, dp in zip(data_types, data_paths)]
    weights_list = None
    if sample_weights is not None:
        weights_list = [float(sample_weights[dt]) for dt in data_types]
    return ComposedDataset(datasets, sample_weights=weights_list)
```

Each dataset class uses `@DATASET_REGISTRY.register()` decorator. Builder instantiates by name from config, `sample_weights` is a dict keyed by dataset name.

## CPU Memory Leak Mitigation

Large-scale training (millions of samples, many workers) is prone to CPU memory leaks.

**Avoid Python native dict/list as Dataset attributes:**
DataLoader with `num_workers > 0` forks worker processes with copy-on-write (CoW). Python objects (dict, list) have inline refcounts — any read increments the refcount, writing to the memory page and triggering CoW duplication across all workers. Use numpy arrays for large index mappings instead:
```python
# Bad: Python list/dict — triggers CoW in every worker
self.indices = list(range(N))                     # N refcount writes
self.item_to_dataset = {i: ds_id for i in range(N)}

# Good: numpy array — data lives outside Python heap, no refcount on access
self.indices = np.arange(N)
self.cum_sizes = np.array([...])  # use np.searchsorted instead of dict lookup
```

**File handle discipline:**
- `HDF5_USE_FILE_LOCKING=FALSE` env var: prevents h5py deadlocks when multiple workers open same file.
- Context manager `open()`/`close()` per `__getitem__` call. Never cache h5py/video handles across calls — worker processes hold them indefinitely.
- Video decoders (`av.open`) must implement `__del__` calling `close()` as safety net.

**Training loop cleanup:**
```python
del batch
if step % gc_interval == 0:
    gc.collect(); torch.cuda.empty_cache()
del state_dict, optim_state  # after checkpoint save/resume
```

