"""
Training script for sequence-to-sequence addition model.
"""

import json
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def compute_accuracy(outputs, targets, pad_token=0):
  
    preds = outputs.argmax(dim=-1)                
    mask = (targets != pad_token)                  
    same = (preds == targets) | (~mask)             
    seq_correct = same.all(dim=1).float().mean().item()
    return seq_correct


def _make_padding_mask(tokens, pad_token, for_encoder=True):
 
    B, L = tokens.shape
    key_ok = (tokens != pad_token).unsqueeze(1).unsqueeze(2) 
    if for_encoder:
      
        pass
    return key_ok  # bool


def _make_tgt_mask(tgt, pad_token, device):
   
    B, L = tgt.shape
    causal = create_causal_mask(L, device=device).bool()      # [1,1,L,L]，bool
    key_ok = (tgt != pad_token).unsqueeze(1).unsqueeze(2)     # [B,1,1,L]，bool

    return causal & key_ok


def train_epoch(model, dataloader, criterion, optimizer, device):
  
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pad_token = model.vocab_size - 1  

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['input'].to(device)     # [B, L_in]
        targets = batch['target'].to(device)   # [B, L_out]

        
        start_col = torch.zeros((targets.size(0), 1), dtype=torch.long, device=device)
        dec_in = torch.cat([start_col, targets[:, :-1]], dim=1)   # [B, L_out]

        src_mask = _make_padding_mask(inputs, pad_token, for_encoder=True)   # [B,1,1,L_in]
        tgt_mask = _make_tgt_mask(dec_in, pad_token, device)                 # [B,1,L_out,L_out]

  
        logits = model(inputs, dec_in, src_mask=src_mask, tgt_mask=tgt_mask)  # [B, L_out, V]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

       
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

   
        acc = compute_accuracy(logits, targets, pad_token=pad_token)
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pad_token = model.vocab_size - 1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            start_col = torch.zeros((targets.size(0), 1), dtype=torch.long, device=device)
            dec_in = torch.cat([start_col, targets[:, :-1]], dim=1)

            src_mask = _make_padding_mask(inputs, pad_token, for_encoder=True)
            tgt_mask = _make_tgt_mask(dec_in, pad_token, device)

            logits = model(inputs, dec_in, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            acc = compute_accuracy(logits, targets, pad_token=pad_token)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train addition transformer')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

  
    train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(args.device)


    pad_token = vocab_size - 1
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)


    best_val_acc = -1.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2%}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

   
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2%}")


    model.load_state_dict(torch.load(out_dir / 'best_model.pth', map_location=args.device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")


    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Results saved to {out_dir}")


if __name__ == '__main__':
    main()

