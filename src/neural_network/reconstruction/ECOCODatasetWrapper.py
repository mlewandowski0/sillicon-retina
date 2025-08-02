from torch.utils.data import Dataset
import numpy as np 
import torch 


class ECOCODatasetWrapper(Dataset):
    """ Wrapper for ECOCO dataset to prepare data for E2VID training.
    """
    def __init__(self, ecoco_dataset, sequence_length= 8, skip_frames=1,
                 include_intial_frames_prob = 0.3, augment=True):
        self.dataset = ecoco_dataset
        self.sequence_length = sequence_length
        self.skip_frames = skip_frames
        self.include_initial_frames_prob = include_intial_frames_prob
        self.augment = augment
        
         # Build valid indices for sequences
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            n_frames = item['frames'].shape[0]
            n_grids = item['grids'].shape[0]
            
            # Check if we have enough frames for a sequence
            if n_frames >= sequence_length and n_grids >= sequence_length - 1:
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} valid sequences out of {len(self.dataset)} items")

    def _augment_sequence(self, frames, grids):
        """Apply data augmentation to frames and grids"""

        # frames here are of shape (N, H, W)
        # while events are of shape (N, 5, H, W)

        if not self.augment:
            return frames, grids

        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames = frames[ :, :, ::-1].copy()
            grids = grids[:, :,  :, ::-1].copy()
            # Flip polarity for horizontal flip
            grids = grids * -1

        # Random vertical flip
        if np.random.rand() > 0.5:
            frames = frames[ :, ::-1, :].copy()
            grids = grids[ :, :, ::-1, :].copy()
            # Flip polarity for vertical flip
            grids = grids * -1

        # Random brightness adjustment for frames
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 1)

        # Random noise for grids
        if np.random.rand() > 0.5:
            noise_level = 0.1
            noise = np.random.randn(*grids.shape) * noise_level
            grids = grids + noise

        return frames.astype(np.float32), grids.astype(np.float32)

    def __len__(self):
        return int(len(self.valid_indices) * 1.5)

    def __getitem__(self, idx):
        """
        Returns a sequence ready for E2VID training.
        """
        # Map idx to valid_indices with wraparound
        dataset_idx = self.valid_indices[idx % len(self.valid_indices)]
        item = self.dataset[dataset_idx]

        frames = item['frames']  # (N, H, W)
        grids = item['grids']  # (N-1, 5, H, W)

        # Normalize frames to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        grids = grids.astype(np.float32)

        # Decide whether to include initial frames (no history case)
        include_initial = np.random.random() < self.include_initial_frames_prob

        if include_initial:
            # Create a sequence that starts from the beginning
            # This teaches the network to reconstruct from zero history
            start_idx = 0

            # For initial frames, we need to handle the case where we don't have
            # enough frames/grids for a full sequence
            actual_length = min(self.sequence_length, frames.shape[0], grids.shape[0] + 1)

            # Extract what we have
            frame_sequence = frames[:actual_length]
            grid_sequence = grids[:actual_length - 1] if actual_length > 1 else np.zeros(
                (0, 5, grids.shape[2], grids.shape[3]), dtype=np.float32)

            # Pad with zeros if necessary to reach sequence_length
            if actual_length < self.sequence_length:
                # Pad frames
                frame_padding = np.zeros((self.sequence_length - actual_length, frames.shape[1], frames.shape[2]),
                                         dtype=np.float32)
                frame_sequence = np.concatenate([frame_sequence, frame_padding], axis=0)

                # Pad grids
                grid_padding = np.zeros((self.sequence_length - actual_length, 5, grids.shape[2], grids.shape[3]),
                                        dtype=np.float32)
                if len(grid_sequence) > 0:
                    grid_sequence = np.concatenate([grid_sequence, grid_padding], axis=0)
                else:
                    grid_sequence = grid_padding

            # Add initial zero grid (representing no events before first frame)
            grid_sequence = np.concatenate([
                np.zeros((1, 5, grids.shape[2], grids.shape[3]), dtype=np.float32),
                grid_sequence[:self.sequence_length - 1]
            ], axis=0)

        else:
            # Normal case: random starting point in the middle of sequence
            max_start = min(frames.shape[0] - self.sequence_length, grids.shape[0] - self.sequence_length + 1)
            start_idx = np.random.randint(0, max_start + 1)

            # Extract sequence
            frame_sequence = frames[start_idx:start_idx + self.sequence_length]
            grid_sequence = grids[start_idx:start_idx + self.sequence_length - 1]

            # Add extra grid at the beginning to match frame count
            # This represents events between frame[start_idx-1] and frame[start_idx]
            if start_idx > 0:
                # Use actual previous events if available
                prev_grid = grids[start_idx - 1]
            else:
                # Use zeros if at the beginning
                prev_grid = np.zeros((5, grids.shape[2], grids.shape[3]), dtype=np.float32)

            grid_sequence = np.concatenate([
                prev_grid[np.newaxis, ...],
                grid_sequence
            ], axis=0)


        # apply data augmentation
        frame_sequence,grid_sequence = self._augment_sequence(frame_sequence, grid_sequence)

        # Convert to tensors
        frame_sequence = torch.from_numpy(frame_sequence).unsqueeze(1)  # (L, 1, H, W)
        grid_sequence = torch.from_numpy(grid_sequence)  # (L, 5, H, W)

        # Create mask indicating which frames are valid (not padded)
        # This can be used for loss computation if needed
        valid_mask = torch.ones(self.sequence_length, dtype=torch.bool)
        if include_initial and actual_length < self.sequence_length:
            valid_mask[actual_length:] = False

        return {
            'frames': frame_sequence,
            'grids': grid_sequence,
            'valid_mask': valid_mask,
            'is_initial': include_initial
        }


if __name__ == "__main__":
    from .ECOCO_dataset_file_cached import ECOCO_dataset_file_cached 
    dataset = ECOCO_dataset_file_cached(root="data/ecoco_depthmaps_test", split="train", cache_folder="data/ecoco_cache", limit=100)
    wrapped_dataset = ECOCODatasetWrapper(dataset, sequence_length=8, skip_frames=1, include_intial_frames_prob=1.0)
    print(f"Wrapped dataset length: {len(wrapped_dataset)}")
    sample = wrapped_dataset[0]
    frames = sample['frames']
    grids = sample['grids']

    print("Sample frames shape: ", frames.shape)
    print("Sample grids shape: ", grids.shape)
    print("Sample valid mask: ", sample['valid_mask'])