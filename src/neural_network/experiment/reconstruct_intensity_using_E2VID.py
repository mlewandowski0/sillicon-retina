import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import wandb
from datetime import datetime
import torch.optim as optim

from torch.utils.data import DataLoader

from src.neural_network.networks.E2VIDModel import E2VIDModel
from src.neural_network.networks.PerceptualLoss import PerceptualLoss
from src.neural_network.reconstruction.ECOCODatasetWrapper import ECOCODatasetWrapper
from src.neural_network.reconstruction.ECOCO_dataset_file_cached import ECOCO_dataset_file_cached
from src.neural_network.reconstruction.reconstruction_utils import train_epoch, validate, save_checkpoint

from torch.optim.lr_scheduler import StepLR, _LRScheduler


class StepLRWithMin(StepLR):
    def __init__(self, optimizer, step_size, gamma = 0.1, last_epoch = -1, min_lr = 1e-6):
        super().__init__(optimizer, step_size, gamma, last_epoch)
        self.min_lr = min_lr
    
    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to json config file")
    args = ap.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = json.load(f)

    print("E2VID Training on ECOCO Dataset")
    print("================================")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    if config.get('use_wandb', False):
        wandb.init(
            project="e2vid-ecoco",
            config=config,
            name=f"e2vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    model = E2VIDModel(num_bins=config["num_bins"],
                       num_prev_images=config["num_prev_images"],
                       dropout_rate=config["dropout_rate"]).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    train_base = ECOCO_dataset_file_cached(
        root=config['data_root'],
        split='train',
        cache_folder=config['cache_folder'] + "_train",
        limit=config.get('train_limit', None),

    )

    train_dataset = ECOCODatasetWrapper(
        train_base,
        sequence_length=config['sequence_length'],
        skip_frames=config.get('skip_frames', 1),
        augment=config["augment"],
    )

    val_base = ECOCO_dataset_file_cached(
        root=config['data_root'],
        split='validation',
        cache_folder=config['cache_folder'] + "_validation",
        limit=config.get('val_limit', None)
    )

    val_dataset = ECOCODatasetWrapper(
        val_base,
        sequence_length=config['sequence_length'],
        skip_frames=config.get('skip_frames', 1),
        augment = config["augment"]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config['prefetch_factor'],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        prefetch_factor=config['prefetch_factor'],
    )

    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")\

    # Loss and optimizer
    criterion = PerceptualLoss(use_lpips=config.get('use_lpips', False))

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = StepLRWithMin(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma'],
        min_lr=config.get('min_lr', 1e-6)
        
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch, config)

        print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log to wandb
        if config.get('use_wandb', False):
            wandb.log({
                'train_loss_epoch': train_loss,
                'val_loss_epoch': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, config, is_best
            )

        # Update learning rate
        scheduler.step()

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")