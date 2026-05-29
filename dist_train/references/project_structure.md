# Project Structure

```
proj_name/
├── proj_name_model/
│   └── model_name/
│       ├── modeling_model_name.py   # trainable backbone definition
│       └── processing_model_name.py # full pipeline: backbone + frozen encoders (text enc, etc.)
├── proj_name_data/                  # data processing, transforms, collators
├── proj_name_datasets/              # dataset classes, one per training data source
├── proj_name_trainer/
│   └── trainer_model_name.py        # trainer class per model
├── proj_name_infer/                 # inference scripts and utilities
├── benchmark/                       # evaluation code
├── config/                          # yaml configs
├── scripts/                         # launch bash scripts and python entry points
└── submit_xxx.sh                    # root-level cluster job submission
```

## Conventions

- `modeling_*.py`: only trainable parameters. Frozen components (pretrained text encoders, vision encoders) live in `processing_*.py` which assembles the full forward pipeline.
- Dataset classes under `proj_name_datasets/`, one file per data source. Keep data loading/processing separate from training logic.
- Trainer classes under `proj_name_trainer/`, one file per model variant. Trainer owns the training loop, optimizer, checkpoint logic.
- Entry points in `scripts/`, cluster submission at repo root.
