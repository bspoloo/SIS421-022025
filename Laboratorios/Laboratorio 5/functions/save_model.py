import torch

def save_model_with_metadata(model, trainer, config, save_path="model_with_metadata.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'hyperparameters': config,
        'epoch': trainer.current_epoch,
        'val_acc': trainer.callback_metrics.get('val_acc', 0.0),
        'val_loss': trainer.callback_metrics.get('val_loss', 0.0)
    }
    
    torch.save(checkpoint, save_path)
    print(f"Modelo con metadata guardado como: {save_path}")
    return save_path
