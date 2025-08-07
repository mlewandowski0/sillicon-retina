import argparse
import json
import wandb
from datetime import datetime
import torch.optim as optim
import torch
import os
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity, schedule


from torch.utils.data import DataLoader
from tqdm import tqdm

from src.neural_network.networks.E2VID_convLSTM import E2VID_convLSTM
from src.neural_network.networks.PerceptualLoss import PerceptualLoss
from src.neural_network.networks.TemporalConsistencyLoss import TemporalConsistencyLoss
from src.neural_network.dataloaders.ECOCODatasetWrapper import ECOCODatasetWrapper
from src.neural_network.dataloaders.ECOCO_dataset_file_cached import ECOCO_dataset_file_cached
from src.neural_network.dataloaders.reconstruction_utils import save_checkpoint  # <-- add this

from torch.optim.lr_scheduler import StepLR


torch.backends.cudnn.benchmark = True

class StepLRWithMin(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=1e-6):
        super().__init__(optimizer, step_size, gamma, last_epoch)
        self.min_lr = min_lr

    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]


def train_epoch(model, loader, optimizer, device, epoch, config,
                lpips_loss: PerceptualLoss, tc_loss: TemporalConsistencyLoss, profiler):

    model.train()
    total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):
        # shapes:
        # frames: (B, L, 1, H, W)
        # grids:  (B, L, 5, H, W)
        # flows:  (B, L, 2, H, W)  (optional)

        if profiler is not None:
            profiler.step()

        frames = batch['frames'].to(device)
        grids = batch['grids'].to(device)
        flows = batch.get('optical_flows', None)
        if flows is not None:
            flows = flows.to(device)
        valid = batch['valid_mask'].to(device)

        B, L, _, H, W = frames.shape

        # reset recurrent state with the current spatial size
        model.reset(input_shape=(H, W), batch_size=B)

        loss = 0.0
        valid_steps = 0
        Ihat_prev = None
        Igt_prev = None
        perceptual_term_sum = 0.0
        temporal_consistency_term_sum = 0.0

        Igt_t = None
        Ihat_t = None

        for t in range(L):
            # skip time step if no sample in batch has it valid
            if not valid[:, t].any():
                continue

            x_t = grids[:, t]       # (B, 5, H, W)
            Igt_t = frames[:, t]    # (B, 1, H, W)
            Ihat_t = model(x_t)     # (B, 1, H, W)

            # dataloaders loss (make sure it reduces to scalar)
            perceptual_loss = lpips_loss(Ihat_t, Igt_t).mean()

            # temporal consistency (device-safe zero if not applicable)
            temporal_consistency_term = torch.zeros((), device=Igt_t.device)
            if (tc_loss is not None) and (Igt_prev is not None) and (Ihat_prev is not None) and (flows is not None):
                # optical flow at step t: (B, 2, H, W)
                of = flows[:, t]
                temporal_consistency_term = tc_loss(of, Ihat_t, Ihat_prev, Igt_t, Igt_prev)

            loss = loss + perceptual_loss + temporal_consistency_term

            perceptual_term_sum += float(perceptual_loss.detach().cpu().item())
            temporal_consistency_term_sum += float(temporal_consistency_term.detach().cpu().item())

            Ihat_prev = Ihat_t.detach()
            Igt_prev = Igt_t
            valid_steps += 1

        # Optional debug save
        if config.get("debug_save", False) and (batch_idx % config.get("debug_iters", 100) == 0):
            if (Igt_t is not None) and (Ihat_t is not None):
                os.makedirs(config["debug_folder"], exist_ok=True)
                with torch.no_grad():
                    fig, axes = plt.subplots(2, min(4, B), figsize=(16, 8))
                    if axes.ndim == 1:
                        axes = axes[None, :]  # ensure 2D for consistent indexing
                    for i in range(min(4, B)):
                        axes[0, i].imshow(Igt_t[i, 0].detach().cpu().numpy(), cmap='gray')
                        axes[0, i].set_title('GT')
                        axes[0, i].axis('off')
                        axes[1, i].imshow(Ihat_t[i, 0].detach().cpu().numpy(), cmap='gray')
                        axes[1, i].set_title('Pred')
                        axes[1, i].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config['debug_folder'], f'epoch_{epoch}_batch_{batch_idx}.png'))
                    plt.close()

        # normalize by number of valid time steps (prevents dependence on L)
        if valid_steps == 0:
            continue
        loss = loss / valid_steps

        # optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += float(loss.detach().cpu().item())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total / (batch_idx + 1):.4f}",
            'lr': optimizer.param_groups[0]['lr'],
            'percep_sum': f"{perceptual_term_sum:.3f}",
            'tc_sum': f"{temporal_consistency_term_sum:.3f}",
        })

        # Log to wandb if enabled
        if config.get('use_wandb', False) and batch_idx % 10 == 0:
            wandb.log({
                'train_loss_step': float(loss.detach().cpu().item()),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'perceptual_loss_sum': perceptual_term_sum,
                'temporal_consistency_term_sum': temporal_consistency_term_sum,
                'epoch': epoch,
                'step': epoch * len(loader) + batch_idx
            })

    if profiler is not None:
        profiler.__exit__(None, None, None)

    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, epoch, config,
             lpips_loss: PerceptualLoss, tc_loss: TemporalConsistencyLoss):
    model.eval()
    total = 0.0

    pbar = tqdm(loader, desc=f"Validation {epoch + 1}/{config['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)      # (B, L, 1, H, W)
        grids = batch['grids'].to(device)        # (B, L, 5, H, W)
        flows = batch.get('optical_flows', None)
        if flows is not None:
            flows = flows.to(device)
        valid = batch['valid_mask'].to(device)   # (B, L)

        B, L, _, H, W = frames.shape
        model.reset(input_shape=(H, W), batch_size=B)

        loss = 0.0
        valid_steps = 0
        Ihat_prev = None
        Igt_prev = None

        for t in range(L):
            if not valid[:, t].any():
                continue

            x_t = grids[:, t]
            Igt_t = frames[:, t]
            Ihat_t = model(x_t)

            Lr = lpips_loss(Ihat_t, Igt_t).mean()

            Ltc = torch.zeros((), device=Igt_t.device)
            if (tc_loss is not None) and (Igt_prev is not None) and (Ihat_prev is not None) and (flows is not None):
                of = flows[:, t]
                Ltc = tc_loss(of, Ihat_t, Ihat_prev, Igt_t, Igt_prev)

            loss = loss + Lr + Ltc

            Ihat_prev = Ihat_t
            Igt_prev = Igt_t
            valid_steps += 1

        if valid_steps == 0:
            continue
        loss = loss / valid_steps
        total += float(loss.detach().cpu().item())

        pbar.set_postfix({'val_loss': f"{total / (batch_idx + 1):.4f}"})

    return total / max(1, len(loader))


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

    model = E2VID_convLSTM(input_channels=5,
                           skip_type = config['skip_type']).to(device)
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
        augment=config["augment"]
    )

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

    # profiling
    profile_on = config.get("profile_on", False)
    if profile_on:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=5, warmup=5, active=20, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                config.get("profile_dir", "./tb_prof")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        prof.__enter__()
    else:
        prof = None

    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")

    # Losses and optimizer
    criterion = PerceptualLoss(use_lpips=config.get('use_lpips', False))
    temporal_criterion = TemporalConsistencyLoss()  # ensure it runs on same device internally

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            lpips_loss=criterion, tc_loss=temporal_criterion, profiler=prof
        )

        val_loss = validate(
            model, val_loader, device, epoch, config,
            lpips_loss=criterion, tc_loss=temporal_criterion, profiler=prof
        )

        print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if config.get('use_wandb', False):
            wandb.log({
                'train_loss_epoch': train_loss,
                'val_loss_epoch': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, config, is_best
            )

        scheduler.step()

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
