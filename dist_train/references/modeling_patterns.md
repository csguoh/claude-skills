# Modeling Patterns

## File Structure

```
proj_name_model/model_name/
├── modeling_model_name.py      # trainable backbone (transformer, diffusion model, etc.)
└── processing_model_name.py    # full pipeline: frozen encoders + backbone + compute_loss
```

## modeling_model_name.py — Trainable Backbone

Only contains the trainable model definition. Pure `nn.Module`, no frozen components.

```python
class MyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define trainable layers

    def forward(self, input_dict) -> tuple:
        # core forward pass, returns predictions
        return predictions
```

This is the module that gets FSDP-wrapped and whose parameters are passed to the optimizer.

## processing_model_name.py — Full Pipeline

Assembles the complete model: loads frozen encoders (text encoder, VAE, etc.) + trainable backbone. Provides `compute_loss` for training and `generate`/`inference` for inference.

```python
class MyModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        # frozen components — loaded to device, requires_grad_(False)
        self.text_encoder = load_text_encoder(config.text_encoder_path, device=device)
        self.text_encoder.requires_grad_(False)

        self.vae = load_vae(config.vae_path, device=device)
        self.vae.requires_grad_(False)

        # trainable backbone — this gets FSDP-wrapped externally
        self.generator = load_backbone(config.backbone_path, device=device)
        self.generator.requires_grad_(True)

    def _prepare_train_input_dict(self, batch_data):
        """Encode raw batch through frozen components into model inputs."""
        input_dict = {}
        with torch.no_grad():
            input_dict['text_emb'] = self.text_encoder(batch_data['input_ids']).detach()
            input_dict['latents'] = self.vae.encode(batch_data['images']).detach()
        # add noise, compute targets, masks, etc.
        return input_dict

    def compute_loss(self, batch_data):
        input_dict = self._prepare_train_input_dict(batch_data)
        predictions = self.generator(input_dict)

        # loss with masking (e.g. padding, conditional frames)
        target = input_dict['target']
        loss = F.mse_loss(predictions.float(), target.float().detach(), reduction='none')
        mask = input_dict['valid_mask']
        loss = (loss.flatten(2).mean(dim=2) * mask).sum() / mask.sum().clamp(min=1)
        return loss
```

## Key Conventions

- **Frozen vs trainable separation**: frozen encoders in `processing_*`, trainable backbone in `modeling_*`. Only `self.generator` (or equivalent) gets FSDP-wrapped.
- **`_prepare_train_input_dict`**: runs frozen encoders under `torch.no_grad()`, calls `.detach()` on outputs to cut the graph. Prepares noise, targets, masks for training.
- **`compute_loss`**: calls `_prepare_train_input_dict` → backbone forward → loss computation. Returns scalar loss(es) for backward. Use `reduction='none'` + manual masking for variable-length/padded data.
- **FSDP wrapping target**: trainer wraps `model.generator` (not the entire model), so frozen components stay outside FSDP and don't consume sharding communication.
