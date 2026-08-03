import logging

import torch
import torch.nn as nn

from src.clip_loss import compute_batch_loss


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def snapshot_trainable_params(model):
    params = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if parameter.requires_grad:
            params[name] = parameter.detach().clone()
    return params


class EWCRegularizer:
    def __init__(self, checkpoint, fisher, lambd):
        self.checkpoint = checkpoint
        self.fisher = fisher
        self.lambd = lambd

    def penalty(self, model):
        if self.checkpoint is None or self.fisher is None:
            return torch.tensor(0.0, device=next(unwrap_model(model).parameters()).device)

        penalty = torch.tensor(0.0, device=next(unwrap_model(model).parameters()).device)
        for name, parameter in unwrap_model(model).named_parameters():
            if parameter.requires_grad and name in self.checkpoint:
                penalty = penalty + (self.fisher[name] * (parameter - self.checkpoint[name]) ** 2).sum()
        return self.lambd * penalty


@torch.no_grad()
def _zero_fisher_like(model):
    zeros = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if parameter.requires_grad:
            zeros[name] = torch.zeros_like(parameter)
    return zeros


def estimate_fisher(model, dataloader, options, max_batches=None):
    criterion = nn.CrossEntropyLoss().to(options.device)
    fisher = _zero_fisher_like(model)
    used_batches = 0

    was_training = model.training
    model.eval()

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        model.zero_grad()
        loss, _, _, _ = compute_batch_loss(model, batch, criterion, options)
        loss.backward()

        for name, parameter in unwrap_model(model).named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2

        used_batches += 1

    if used_batches == 0:
        raise RuntimeError("Fisher estimation received zero batches.")

    for name in fisher:
        fisher[name] /= used_batches

    checkpoint = snapshot_trainable_params(model)

    if was_training:
        model.train()

    logging.info(f"EWC Fisher estimated from {used_batches} batches")
    return EWCRegularizer(checkpoint=checkpoint, fisher=fisher, lambd=options.ewc_lambda)
