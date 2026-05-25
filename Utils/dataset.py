import os
import cv2
import torch
from torch.utils.data import Dataset

class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None, file_names=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        image_map = {self._sample_id(name): name for name in image_files}
        mask_map = {self._sample_id(name): name for name in mask_files}

        common_ids = sorted(set(image_map) & set(mask_map))
        missing_images = sorted(set(mask_map) - set(image_map))
        missing_masks = sorted(set(image_map) - set(mask_map))

        if missing_images:
            print(f"WARNING: Missing images for {len(missing_images)} mask files")
        if missing_masks:
            print(f"WARNING: Missing masks for {len(missing_masks)} image files")

        if file_names is None:
            selected_ids = common_ids
        else:
            requested_ids = [self._sample_id(name) for name in file_names]
            missing_requested = [
                sample_id for sample_id in requested_ids
                if sample_id not in image_map or sample_id not in mask_map
            ]
            if missing_requested:
                print(f"WARNING: Skipping {len(missing_requested)} requested samples with missing pairs")
            selected_ids = [
                sample_id for sample_id in requested_ids
                if sample_id in image_map and sample_id in mask_map
            ]

        self.sample_ids = selected_ids
        self.samples = [(image_map[sample_id], mask_map[sample_id]) for sample_id in selected_ids]

    @staticmethod
    def _sample_id(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    def __len__(self):
        return len(self.samples)

    def _safe_read(self, img_path, mask_path):
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Failed to read image: {img_path}")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[WARNING] Failed to read mask: {mask_path}")
            return None, None

        return image, mask

    def __getitem__(self, idx):

        while True:

            img_name, mask_name = self.samples[idx]

            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image, mask = self._safe_read(img_path, mask_path)

            if image is None or mask is None:
                idx = (idx + 1) % len(self.samples)
                continue

            break

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = image.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask