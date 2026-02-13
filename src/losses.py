"""
Loss functions for nano-train.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with ignore_index support.

    Args:
        ignore_index: Target value to ignore in the loss.
        reduction: "mean", "sum", or "none".
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Cross-entropy measures the difference between two probability distributions.
        For classification, we compare:
        - Predicted distribution: softmax(logits) - what the model thinks
        - Target distribution: one-hot at targets - what's actually correct

        Formula: L = -log(softmax(logits)[target_class])

        The lower the loss, the better the predictions:
        - Loss = 0  → perfect prediction (p=1.0 for correct class)
        - Loss = 2.3 → random (for 10 classes, p=0.1 each)
        - Loss = 10 → terrible (p≈0.00005 for correct class)

        VISUALIZED with concrete example:

        Input: "cat sat" → tokens: [2, 5, 3]
        Vocab size: 10 (tokens 0-9)

        Step 1: Model produces logits (raw scores, not probabilities yet):
        -------
        logits[0, 0, :] = [1.93, 1.49, 0.90, -2.11, 0.68, -1.23, -0.04, -1.60, -0.75, 1.65]
                             token 0   token 1   token 2   ...   token 9
        These are the model's predictions for what comes after "cat" (token 2).
        The model is saying: token 0 has score 1.93, token 5 has score -1.23, etc.

        Step 2: Softmax converts scores to probabilities (sum = 1.0):
        -------
        softmax(logits[0, 0, :]) = [0.299, 0.193, 0.107, 0.005, 0.086, 0.013, ...]
        P(token=0|cat) = 29.9%
        P(token=1|cat) = 19.3%
        P(token=5|cat) = 1.3%   ← this is the correct answer ("sat")
        ...

        Step 3: Cross-entropy extracts probability of the CORRECT token:
        -------
        target = shift_labels[0, 0] = 5  (the token "sat")

        loss = -log(P(token=5|cat))
             = -log(0.013)
             = 4.37

        This is high because the model only gave 1.3% probability to the
        correct answer! As training progresses, the model learns to increase
        the probability of correct tokens, and loss decreases.

        Step 4: Averaging over all positions:
        -------
        Average loss = (loss[0,0] + loss[0,1] + ... + loss[batch,seq]) / total_positions

        During training:
        Step 1: loss = 4.0  (model knows almost nothing)
        Step 10: loss = 2.5  (model learning basic patterns)
        Step 50: loss = 1.0  (model getting better)
        Step 100: loss = 0.03 (model very confident)

        Args:
            logits: Tensor of shape (..., num_classes)
                    Raw scores from the model. Higher value = model thinks
                    this class is more likely.
            targets: Tensor of shape (...) with class indices
                    The correct class index (0 to num_classes-1) for each
                    position. Use -100 for padding positions to ignore.

        Returns:
            Scalar loss value (if reduction="mean" or "sum")
            or loss per position (if reduction="none")
        """
        if logits.dim() < 2:
            raise ValueError("logits must have shape (..., num_classes)")

        original_shape = targets.shape
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        if targets.numel() != logits.size(0):
            raise ValueError("Targets must match logits batch dimensions.")

        # Convert raw scores to log probabilities
        # log_softmax(x) = log(exp(x) / sum(exp(x)))
        # This is numerically stable compared to log(softmax(x))
        log_probs = torch.log_softmax(logits, dim=-1)

        # Mask out padding tokens (targets == -100)
        # These positions don't contribute to loss
        mask = targets != self.ignore_index

        if not torch.any(mask):
            if self.reduction == "none":
                return logits.new_zeros(original_shape)
            return logits.new_zeros(())

        # Gather the log probability of the target class
        # Example: if targets[i] = 5, we want log_probs[i, 5]
        # gather() extracts exactly the target's probability from each row
        safe_targets = targets.clone()
        safe_targets[~mask] = 0  # Replace ignored indices with dummy value
        nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        nll = nll * mask.to(nll.dtype)  # Zero out masked positions

        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "mean":
            denom = mask.sum().to(nll.dtype)
            return nll.sum() / denom
        return nll.view(original_shape)
