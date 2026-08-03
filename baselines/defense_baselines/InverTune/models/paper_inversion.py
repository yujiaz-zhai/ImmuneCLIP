import torch
import torch.nn as nn
import torch.nn.functional as F

from models.invertune_badclip import CLIP_MEAN, CLIP_STD, denormalize_clip


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


class PaperBackdoorInversion(nn.Module):
    def __init__(
        self,
        model,
        text_classifier,
        target_label,
        weights,
        learning_rate=1e-2,
        temperature=0.07,
    ):
        super().__init__()
        self.model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.text_classifier = text_classifier.detach()
        self.target_label = int(target_label)
        self.weights = weights
        self.temperature = float(temperature)
        device = next(model.parameters()).device

        self.mask = nn.Parameter(torch.rand(1, 3, 224, 224, device=device))
        mean = self.mask.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = self.mask.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        low = (torch.zeros_like(mean) - mean) / std
        high = (torch.ones_like(mean) - mean) / std
        self.register_buffer("trigger_low", low)
        self.register_buffer("trigger_high", high)
        self.trigger = nn.Parameter(low + torch.rand_like(self.mask) * (high - low))
        self.optimizer = torch.optim.Adam(
            [self.mask, self.trigger], lr=float(learning_rate)
        )

    def apply_trigger(self, images):
        mask = self.mask.clamp(0, 1)
        trigger = self.trigger.clamp(self.trigger_low, self.trigger_high)
        return (1 - mask) * images + mask * trigger

    @torch.no_grad()
    def encode_target(self, target_images):
        features = self.model.encode_image(target_images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[:1]

    def optimize(self, clean_images, target_feature):
        self.optimizer.zero_grad(set_to_none=True)
        poisoned_images = self.apply_trigger(clean_images)
        image_features = self.model.encode_image(poisoned_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.text_classifier
        logits = logits / self.temperature
        labels = torch.full(
            (clean_images.shape[0],),
            self.target_label,
            dtype=torch.long,
            device=clean_images.device,
        )
        alignment = F.cross_entropy(logits, labels)
        embedding = torch.norm(image_features - target_feature, p=2)
        similarity = 1 - structural_similarity(
            denormalize_clip(poisoned_images), denormalize_clip(clean_images)
        )
        sparsity = self.mask.abs().sum()
        total = (
            float(self.weights["infonce_weight"]) * alignment
            + float(self.weights["emd_weight"]) * embedding
            + float(self.weights["ssim_weight"]) * similarity
            + float(self.weights["mask_weight"]) * sparsity
        )
        total.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.mask.clamp_(0, 1)
            self.trigger.clamp_(self.trigger_low, self.trigger_high)

        return {
            "total": float(total.detach()),
            "alignment": float(alignment.detach()),
            "embedding": float(embedding.detach()),
            "ssim": float(similarity.detach()),
            "mask_l1": float(sparsity.detach()),
        }

    @torch.no_grad()
    def attack_success_rate(self, images):
        features = self.model.encode_image(self.apply_trigger(images))
        features = features / features.norm(dim=-1, keepdim=True)
        predictions = (features @ self.text_classifier).argmax(dim=-1)
        return float(predictions.eq(self.target_label).float().mean() * 100)

    def export_state(self, diagnostics=None):
        return {
            "mask": self.mask.detach().cpu(),
            "trigger": self.trigger.detach().cpu(),
            "target_label": self.target_label,
            "diagnostics": diagnostics or {},
        }
