

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size



def extract_attention_weights(model, dataloader, device, num_samples=4):
    """
    Extract encoder/decoder attention weights from the model.

    Args:
        model: Trained Seq2SeqTransformer model.
        dataloader: DataLoader providing test samples.
        device: Torch device.
        num_samples: Number of batches to sample from.

    Returns:
        Dictionary with encoder, decoder self, and cross attention tensors.
    """
    model.eval()
    results = {
        "encoder_attention": [],
        "decoder_self_attention": [],
        "decoder_cross_attention": [],
        "inputs": [],
        "targets": [],
    }

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            B = inputs.size(0)

            # Decoder input (prepend start token = 0)
            start_tok = torch.zeros((B, 1), dtype=torch.long, device=device)
            dec_in = torch.cat([start_tok, targets[:, :-1]], dim=1)

            # Forward pass
            _ = model(inputs, dec_in)

            # Extract attention tensors (saved during forward)
            enc_attn = [layer.self_attn.last_attention for layer in model.encoder_layers]
            dec_self = [layer.self_attn.last_attention for layer in model.decoder_layers]
            dec_cross = [layer.cross_attn.last_attention for layer in model.decoder_layers]

            results["encoder_attention"].append(enc_attn)
            results["decoder_self_attention"].append(dec_self)
            results["decoder_cross_attention"].append(dec_cross)
            results["inputs"].append(inputs[0].cpu().numpy())
            results["targets"].append(targets[0].cpu().numpy())

            samples_collected += 1

    return results


def visualize_attention_pattern(attn, title, save_path):
    """
    Visualize a multi-head attention map as heatmaps.

    Args:
        attn: Attention tensor [H, L_q, L_k] or [B, H, L_q, L_k]
        title: Title string for the figure.
        save_path: File path to save the figure.
    """
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu()

    # Normalize shape
    if attn.dim() == 4:
        attn = attn[0]  # [H, Lq, Lk]
    elif attn.dim() == 2:
        attn = attn.unsqueeze(0)
    elif attn.dim() != 3:
        attn = attn.view(1, attn.size(-2), attn.size(-1))

    num_heads, Lq, Lk = attn.shape
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        a = attn[i]
        im = axes[i].imshow(a, aspect='auto', vmin=0.0, vmax=1.0, cmap='Blues')
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_head_specialization(attention_data, output_dir):
    """
    Compute average entropy per encoder head as a simple specialization metric.
    Lower entropy indicates more "focused" or specialized attention.

    Args:
        attention_data: Dict from extract_attention_weights().
        output_dir: Directory to save head analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_stats = {}

    for sample_idx, layer_list in enumerate(attention_data["encoder_attention"]):
        for li, attn in enumerate(layer_list):
            if attn is None:
                continue
            attn = torch.as_tensor(attn)
            if attn.dim() == 4:
                # Average across batch and query positions
                p = attn.mean(dim=(0, 2))  # [H, Lk]
                entropy = (-p * (p + 1e-12).log()).sum(dim=-1).mean().item()
                head_stats[f"sample{sample_idx}_encoder_L{li}"] = float(entropy)

    with open(output_dir / "head_analysis.json", "w") as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy by comparing predicted sequences with targets.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["input"].to(device)
            tgt = batch["target"].to(device)
            pred = model.generate(src, max_len=tgt.size(1))
            same = (pred == tgt).all(dim=1)
            correct += same.sum().item()
            total += src.size(0)
    return correct / total if total > 0 else 0.0


def plot_head_importance(results, save_path):
    """
    Plot bar chart of accuracy drop for each head ablation.
    """
    baseline = results["baseline"]
    heads, drops = [], []
    for k, v in results.items():
        if k != "baseline":
            heads.append(k)
            drops.append(v)

    plt.figure(figsize=(max(10, 0.6 * len(heads)), 5))
    plt.bar(heads, drops)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy Drop")
    plt.title("Head Importance (Ablation)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform encoder head ablation study.

    For each head, zero out its corresponding output projection weights in W_O,
    re-evaluate accuracy, and record the drop relative to the baseline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline:.2%}")
    results = {"baseline": baseline}

    for li, layer in enumerate(model.encoder_layers):
        attn = layer.self_attn
        H = attn.num_heads
        d = attn.d_k
        W_backup = attn.W_O.weight.data.clone()

        for hi in range(H):
            start, end = hi * d, (hi + 1) * d
            attn.W_O.weight.data[:, start:end] = 0.0
            acc = evaluate_model(model, dataloader, device)
            drop = baseline - acc
            results[f"encoder_L{li}_H{hi}"] = float(drop)
            attn.W_O.weight.data.copy_(W_backup)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_head_importance(results, output_dir / "head_importance.png")
    return results


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=4):
    """
    Save visualization of a few sample predictions (input, target, predicted).
    """
    outdir = Path(output_dir) / "attention_patterns" / "example_cases"
    outdir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        cnt = 0
        for batch in dataloader:
            if cnt >= num_examples:
                break
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model.generate(x, max_len=y.size(1))

            x_str = " ".join(map(str, x[0].cpu().numpy().tolist()))
            y_str = "".join(map(str, y[0].cpu().numpy().tolist()))
            p_str = "".join(map(str, pred[0].cpu().numpy().tolist()))
            correct = torch.equal(pred[0], y[0])

            fig, ax = plt.subplots(figsize=(9, 3))
            ax.axis("off")
            ax.text(0.02, 0.75, f"Input : {x_str}", fontsize=10)
            ax.text(0.02, 0.50, f"Target: {y_str}", fontsize=10)
            ax.text(0.02, 0.25, f"Pred  : {p_str}", fontsize=10)
            ax.text(0.02, 0.05, f"Correct: {correct}", fontsize=10)
            plt.tight_layout()
            plt.savefig(outdir / f"example_{cnt+1}.png", dpi=150)
            plt.close()
            cnt += 1




def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("--model-path", required=True, help="Path to saved model (best_model.pth)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="results", help="Results directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model and weights
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print(f"Loaded model from {args.model_path}")

    # Load test data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    outdir = Path(args.output_dir)
    (outdir / "attention_patterns" / "head_heatmaps").mkdir(parents=True, exist_ok=True)
    (outdir / "head_analysis").mkdir(parents=True, exist_ok=True)

    print("Extracting attention weights...")
    attn_data = extract_attention_weights(model, test_loader, args.device, num_samples=4)

    print("Saving encoder attention heatmaps...")
    for sample_i, layer_list in enumerate(attn_data["encoder_attention"]):
        for li, attn in enumerate(layer_list):
            if attn is None:
                continue
            save_path = outdir / "attention_patterns" / "head_heatmaps" / f"sample{sample_i}_L{li}.png"
            visualize_attention_pattern(attn, f"Encoder Layer {li}", save_path)

    analyze_head_specialization(attn_data, outdir / "head_analysis")
    ablation_study(model, test_loader, args.device, outdir / "head_analysis")
    visualize_example_predictions(model, test_loader, args.device, outdir, num_examples=4)

    print(f"Analysis complete! Results saved to {outdir}")


if __name__ == "__main__":
    main()



