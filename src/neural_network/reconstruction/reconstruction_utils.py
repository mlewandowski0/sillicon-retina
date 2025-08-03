import os.path

import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import reconstruction
from tqdm import tqdm
from pathlib import Path
import wandb
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, config):
    model.train()
    total_loss = 0.0
    total_samples = 0
    initial_frame_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):

        # (B, L, 1, H, W)
        frames = batch['frames'].to(device)

        # (B, L, 5, H, W)
        grids = batch['grids'].to(device)

        # (B, L)
        valid_masks = batch['valid_mask'].to(device)

        # (B, )
        is_initial = batch["is_initial"]

        batch_size, seq_length = frames.shape[:2]
        initial_frame_count += is_initial.sum().item()


        # initialize previous images buffer
        # start with zeros to simulate no history
        prev_images = torch.zeros(batch_size, 3, frames.shape[3], frames.shape[4]).to(device)

        # unroll the network for L steps
        total_step_loss = 0.0
        valid_steps = 0
        reconstructions = []


        for t in range(seq_length):
            # Skip if this frame is padding (only in initial sequences
            if not valid_masks[:, t].any():
                continue

            # Get current event voxel grids
            current_grid = grids[:, t] # (B, 5, H, W)

            # Forward pass
            reconstruction = model(current_grid, prev_images)
            reconstructions.append(reconstruction)

            # compute loss only for valid frames
            target = frames[:, t] # (B ,1 ,H ,W)

            # Mask the loss computation
            if valid_masks[:, t].all():
                # all samples in batch are valid
                loss = criterion(reconstruction, target)
            else:
                valid_samples = valid_masks[:, t]
                if valid_samples.any():
                    loss = criterion(
                        reconstruction[valid_samples],
                        target[valid_samples]
                    )
                else:
                    continue

            total_step_loss += loss
            valid_steps += 1

            # update previous images buffer
            prev_images = torch.cat([prev_images[:, 1:, :, :],
                                     reconstruction.detach()], dim=1)

        if config["debug_save"] and batch_idx % config["debug_iters"] == 0:
            # Save sample predictions
            if not os.path.exists(config['debug_folder']):
                os.makedirs(config['debug_folder'])

            with torch.no_grad():
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for i in range(min(4, batch_size)):
                    axes[0, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title('Ground Truth')
                    axes[1, i].imshow(reconstruction[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title('Prediction')
                plt.savefig(os.path.join(config['debug_folder'],f'epoch_{epoch}_batch_{batch_idx}.png'))
                plt.close()

        # Average loss over valid steps
        if valid_steps > 0:
            avg_loss = total_step_loss / valid_steps

            # backprop through time
            optimizer.zero_grad()
            avg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # update metrics
            total_loss += avg_loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({
                'loss': avg_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'initial_frames': f'{initial_frame_count}/{batch_idx + 1}'
            })

            # Log to wandb if enabled
            if config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss_step': avg_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + batch_idx,
                    'initial_frame_ratio': initial_frame_count / ((batch_idx + 1) * batch_size)
                })

    print(f"Epoch {epoch + 1}: {initial_frame_count} initial frame sequences out of {len(dataloader) * batch_size} total")
    return total_loss / total_samples

def validate(model, dataloader, criterion, device, epoch, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation after {epoch + 1}/{config['num_epochs']}")

        for batch in pbar:
            frames = batch['frames'].to(device)
            grids = batch['grids'].to(device)

            batch_size, seq_len = frames.shape[:2]
            prev_images = torch.zeros(batch_size, 3, frames.shape[3], frames.shape[4]).to(device)

            total_step_loss = 0.0

            for t in range(seq_len):
                current_grid = grids[:, t]
                reconstruction = model(current_grid, prev_images)

                target = frames[:, t]
                loss = criterion(reconstruction, target)
                total_step_loss += loss

                prev_images = torch.cat([
                    prev_images[:, 1:, :, :],
                    reconstruction
                ], dim=1)


            avg_loss = total_step_loss / batch_size
            total_loss += avg_loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({"val_loss" : avg_loss.item()})

    return total_loss / total_samples


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }

    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model with val_loss: {val_loss:.4f}")

    # Keep only last N checkpoints
    if config.get('keep_last_n_checkpoints', 5) > 0:
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > config['keep_last_n_checkpoints']:
            for old_checkpoint in checkpoints[:-config['keep_last_n_checkpoints']]:
                old_checkpoint.unlink()
