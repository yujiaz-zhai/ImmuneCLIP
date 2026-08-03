import torch
import torch.distributed as dist


def prepare_train_batch(batch, options):
    if options.inmodal:
        input_ids = torch.cat(
            [
                batch["input_ids"][0].to(options.device, non_blocking=True),
                batch["input_ids"][1].to(options.device, non_blocking=True),
            ]
        )
        attention_mask = torch.cat(
            [
                batch["attention_mask"][0].to(options.device, non_blocking=True),
                batch["attention_mask"][1].to(options.device, non_blocking=True),
            ]
        )
        pixel_values = torch.cat(
            [
                batch["pixel_values"][0].to(options.device, non_blocking=True),
                batch["pixel_values"][1].to(options.device, non_blocking=True),
            ]
        )
    else:
        input_ids = batch["input_ids"].to(options.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(options.device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(options.device, non_blocking=True)

    return input_ids, attention_mask, pixel_values


def gather_backdoor_indices(batch, options):
    if not options.unlearn:
        return None

    if options.distributed:
        backdoor_indices = batch["is_backdoor"].to(options.device)
        gathered = [torch.zeros_like(backdoor_indices) for _ in range(options.num_devices)]
        dist.all_gather(tensor_list=gathered, tensor=backdoor_indices)
        return torch.cat(gathered).to(options.device, non_blocking=True)

    return batch["is_backdoor"].to(options.device, non_blocking=True)


def get_loss(umodel, outputs, criterion, options, gathered_backdoor_indices):
    if options.inmodal:
        image_embeds = outputs.image_embeds[: len(outputs.image_embeds) // 2]
        augmented_image_embeds = outputs.image_embeds[len(outputs.image_embeds) // 2 :]
        text_embeds = outputs.text_embeds[: len(outputs.text_embeds) // 2]
        augmented_text_embeds = outputs.text_embeds[len(outputs.text_embeds) // 2 :]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    if options.distributed:
        if options.inmodal:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            gathered_augmented_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            gathered_augmented_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(gathered_augmented_image_embeds, augmented_image_embeds)
            dist.all_gather(gathered_augmented_text_embeds, augmented_text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[: options.rank]
                + [image_embeds]
                + gathered_image_embeds[options.rank + 1 :]
            )
            text_embeds = torch.cat(
                gathered_text_embeds[: options.rank]
                + [text_embeds]
                + gathered_text_embeds[options.rank + 1 :]
            )
            augmented_image_embeds = torch.cat(
                gathered_augmented_image_embeds[: options.rank]
                + [augmented_image_embeds]
                + gathered_augmented_image_embeds[options.rank + 1 :]
            )
            augmented_text_embeds = torch.cat(
                gathered_augmented_text_embeds[: options.rank]
                + [augmented_text_embeds]
                + gathered_augmented_text_embeds[options.rank + 1 :]
            )
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            image_embeds = torch.cat(
                gathered_image_embeds[: options.rank]
                + [image_embeds]
                + gathered_image_embeds[options.rank + 1 :]
            )
            text_embeds = torch.cat(
                gathered_text_embeds[: options.rank]
                + [text_embeds]
                + gathered_text_embeds[options.rank + 1 :]
            )

    constraint = torch.tensor(0.0, device=options.device)
    if options.unlearn:
        normal_indices = (~gathered_backdoor_indices).nonzero().squeeze()
        backdoor_indices = gathered_backdoor_indices.nonzero()
        backdoor_indices = backdoor_indices[:, 0] if len(backdoor_indices.shape) == 2 else backdoor_indices
        if len(backdoor_indices):
            backdoor_image_embeds = image_embeds[backdoor_indices]
            backdoor_text_embeds = text_embeds[backdoor_indices]
            similarity_backdoor_embeds = torch.diagonal(backdoor_image_embeds @ backdoor_text_embeds.t())
            constraint = (similarity_backdoor_embeds + options.unlearn_target).square().mean().to(options.device, non_blocking=True)
        image_embeds = image_embeds[normal_indices]
        text_embeds = text_embeds[normal_indices]

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if options.inmodal:
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)
    target = torch.arange(batch_size).long().to(options.device, non_blocking=True)

    if options.inmodal:
        crossmodal_contrastive_loss = (
            criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)
        ) / 2
        inmodal_contrastive_loss = (
            criterion(logits_image_per_augmented_image, target)
            + criterion(logits_text_per_augmented_text, target)
        ) / 2
        contrastive_loss = (
            options.clip_weight * crossmodal_contrastive_loss
            + options.inmodal_weight * inmodal_contrastive_loss
        )
    else:
        contrastive_loss = (
            criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)
        ) / 2

    if options.unlearn:
        contrastive_loss = contrastive_loss + (options.constraint_weight * constraint)

    return contrastive_loss, contrastive_loss, constraint


def compute_batch_loss(model, batch, criterion, options):
    input_ids, attention_mask, pixel_values = prepare_train_batch(batch, options)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    umodel = model.module if options.distributed else model
    backdoor_indices = gather_backdoor_indices(batch, options)
    loss, contrastive_loss, constraint = get_loss(
        umodel, outputs, criterion, options, backdoor_indices
    )
    return loss, contrastive_loss, constraint, len(input_ids)
