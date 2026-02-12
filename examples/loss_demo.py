"""
Concrete example of how loss is calculated for language modeling.

This demonstrates the _compute_loss function step-by-step.
"""

import torch
import torch.nn as nn


def demo_loss_calculation():
    """Show loss calculation with concrete numbers."""

    # ============== STEP 1: Model Input ==============
    print("=" * 60)
    print("STEP 1: Model Input (input_ids)")
    print("=" * 60)

    # Example: "cat sat" with vocab_size=10
    # Token IDs (example): cat=2, sat=5
    batch_size = 2
    seq_len = 3
    vocab_size = 10

    input_ids = torch.tensor([
        [2, 5, 3],  # "cat sat mat"
        [4, 1, 6],  # "dog the bed"
    ])

    print(f"Input shape: {input_ids.shape} = (batch={batch_size}, seq_len={seq_len})")
    print(f"Input tokens:\n{input_ids}\n")

    # ============== STEP 2: Model Output (logits) ==============
    print("=" * 60)
    print("STEP 2: Model Forward Pass → Logits")
    print("=" * 60)

    # In reality, model produces these. Here we simulate them.
    # Shape: (batch_size, seq_len, vocab_size)
    # Each position has a distribution over all vocab tokens

    # Simulated logits (small values for demonstration)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    print(f"Logits shape: {logits.shape} = (batch={batch_size}, seq_len={seq_len}, vocab={vocab_size})")
    print(f"Example: logits[0, 0, :] = predictions for token after 'cat' (position 0, batch 0)")
    print(f"  {logits[0, 0, :].numpy()}")
    print(f"  These are raw scores (not probabilities)\n")

    # ============== STEP 3: Shift for Next-Token Prediction ==============
    print("=" * 60)
    print("STEP 3: Shift Logits and Labels")
    print("=" * 60)
    print("Goal: Token at position t predicts token at position t+1")

    # Original
    print(f"Original input_ids:\n{input_ids}\n")

    # After shifting
    shift_logits = logits[..., :-1, :].contiguous()  # Drop last column
    shift_labels = input_ids[..., 1:].contiguous()    # Drop first column

    print(f"shift_logits shape: {shift_logits.shape} = (batch={batch_size}, seq_len={seq_len-1}, vocab={vocab_size})")
    print(f"shift_labels shape: {shift_labels.shape} = (batch={batch_size}, seq_len={seq_len-1})")
    print(f"\nshift_labels (targets):\n{shift_labels}")
    print("\nWhat this means:")
    print("  - Position 0 must predict position 1's token")
    print("  - Position 1 must predict position 2's token")
    print(f"  - Example batch 0: [2,5,3] → predict [5,3]")
    print(f"    - token '2' (cat) should predict '5' (sat)")
    print(f"    - token '5' (sat) should predict '3' (mat)\n")

    # ============== STEP 4: Flatten ==============
    print("=" * 60)
    print("STEP 4: Flatten for CrossEntropyLoss")
    print("=" * 60)

    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    print(f"shift_logits_flat shape: {shift_logits_flat.shape} = (batch*seq_len={batch_size*(seq_len-1)}, vocab={vocab_size})")
    print(f"shift_labels_flat shape: {shift_labels_flat.shape} = (batch*seq_len={batch_size*(seq_len-1)},)")
    print(f"\nshift_labels_flat: {shift_labels_flat.numpy()}")
    print(f"\nEach row in shift_logits_flat predicts corresponding value in shift_labels_flat\n")

    # ============== STEP 5: Compute Loss ==============
    print("=" * 60)
    print("STEP 5: Cross-Entropy Loss")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(shift_logits_flat, shift_labels_flat)

    # Manual calculation for one position
    print("CrossEntropyLoss formula per position:")
    print("  L = -log(softmax(logits)[target_class])")
    print("\nFor position [0, 0] (batch 0, token 'cat' predicting 'sat'=5):")

    logit_vec = shift_logits[0, 0, :]
    target = shift_labels[0, 0]

    # Softmax
    probs = torch.softmax(logit_vec, dim=0)
    print(f"  Logits: {logit_vec.numpy()}")
    print(f"  Softmax probs (sum={probs.sum().item():.4f}):")
    print(f"    {probs.numpy()}")
    print(f"  Target token: {target}")
    print(f"  Probability of target: {probs[target].item():.6f}")
    print(f"  Loss contribution: {-torch.log(probs[target]).item():.6f}")

    print(f"\nAverage loss over all {batch_size*(seq_len-1)} positions: {loss.item():.6f}")
    print("\nLower is better:")
    print("  - Loss = 0 → perfect predictions (p=1.0 for all targets)")
    print("  - Loss = 2.3 → random predictions (p=0.1 for vocab_size=10)")
    print("  - Loss = 10.0 → terrible predictions (p≈0.00005 for targets)")

    # Show training effect
    print("\n" + "=" * 60)
    print("TRAINING EFFECT")
    print("=" * 60)
    print("During training, loss decreases because:")
    print("  1. Logits for correct tokens increase")
    print("  2. Logits for wrong tokens decrease")
    print("  3. Softmax distribution becomes more peaked")

    # Simulate "better" logits
    print("\nExample: Better predictions (lower loss)")
    better_logits = shift_logits_flat.clone()
    for i, target in enumerate(shift_labels_flat):
        better_logits[i, target] += 5.0  # Boost correct token

    better_loss = criterion(better_logits, shift_labels_flat)
    print(f"  Original loss: {loss.item():.6f}")
    print(f"  After boosting correct tokens: {better_loss.item():.6f}")
    print(f"  Loss decreased by {loss.item() - better_loss.item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    demo_loss_calculation()
