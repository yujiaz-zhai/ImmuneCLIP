import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def denormalize_clip(images):
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def structural_similarity(x, y, window_size=11):
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean()


class BackdoorInversion(nn.Module):
    def __init__(self, model, target_word, weights, learning_rate=1e-2):
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.target_word = target_word
        self.weights = weights
        device = next(model.parameters()).device
        self.mask = nn.Parameter(torch.rand(1, 3, 224, 224, device=device))

        mean = self.mask.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = self.mask.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        low = (torch.zeros_like(mean) - mean) / std
        high = (torch.ones_like(mean) - mean) / std
        self.register_buffer("trigger_low", low)
        self.register_buffer("trigger_high", high)
        self.trigger = nn.Parameter(low + torch.rand_like(self.mask) * (high - low))
        self.optimizer = torch.optim.Adam([self.mask, self.trigger], lr=learning_rate)

    def apply_trigger(self, images):
        mask = self.mask.clamp(0, 1)
        trigger = self.trigger.clamp(self.trigger_low, self.trigger_high)
        return (1 - mask) * images + mask * trigger

    def target_features(self, target_images):
        with torch.no_grad():
            return self.model.encode_image(target_images).mean(dim=0, keepdim=True)

    def optimize(self, clean_images, target_features):
        self.optimizer.zero_grad(set_to_none=True)
        poisoned_images = self.apply_trigger(clean_images)
        poisoned_features = self.model.encode_image(poisoned_images)

        poisoned_normalized = poisoned_features / poisoned_features.norm(
            dim=-1, keepdim=True
        )
        target_normalized = target_features / target_features.norm(
            dim=-1, keepdim=True
        )
        info_nce = 1 - (poisoned_normalized * target_normalized).sum(dim=-1).mean()
        embedding = (poisoned_features - target_features).norm(dim=-1).mean()
        ssim = 1 - structural_similarity(
            denormalize_clip(poisoned_images), denormalize_clip(clean_images)
        )
        sparsity = self.mask.abs().sum()

        total = (
            self.weights["infonce_weight"] * info_nce
            + self.weights["emd_weight"] * embedding
            + self.weights["ssim_weight"] * ssim
            + self.weights["mask_weight"] * sparsity
        )
        total.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.mask.clamp_(0, 1)
            self.trigger.clamp_(self.trigger_low, self.trigger_high)

        return {
            "total": float(total.detach()),
            "infonce": float(info_nce.detach()),
            "embedding": float(embedding.detach()),
            "ssim": float(ssim.detach()),
            "mask_l1": float(sparsity.detach()),
        }

    def export_state(self):
        return {
            "mask": self.mask.detach().cpu(),
            "trigger": self.trigger.detach().cpu(),
            "target_word": self.target_word,
        }


def load_inverted_trigger(path, device):
    checkpoint = torch.load(path, map_location=device)
    if "mask" not in checkpoint or "trigger" not in checkpoint:
        raise KeyError(f"Invalid inverted trigger checkpoint: {path}")
    return checkpoint["mask"].to(device), checkpoint["trigger"].to(device)


def apply_inverted_trigger(images, mask, trigger):
    mask = mask.to(images).clamp(0, 1)
    trigger = trigger.to(images)
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    trigger = trigger.clamp((0 - mean) / std, (1 - mean) / std)
    return (1 - mask) * images + mask * trigger
