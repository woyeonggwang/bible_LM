import torch
import time
import math
from model import LanguageModel, encode, decode, content, device, block_size, ckpt_path


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


ids = encode(content)
data = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
chars_per_token = len(content) / len(data)
print(f'토큰 수: {len(data)}, 토큰당 글자 수: {chars_per_token:.2f}')
batch_size = 64
max_steps = 15000
eval_interval = 500
lr = 1e-3
warmup_steps = 500
min_lr = lr / 10
def get_lr(step):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

model = LanguageModel().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
t0 = time.time()  
best_val = float('inf') 
for step in range(1, max_steps + 1):
    cur_lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = cur_lr
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
              f"val(글자당) {losses['val'] / chars_per_token:.4f}, lr {cur_lr:.2e}")
        if losses['val'] < best_val:                     
            best_val = losses['val']                     
            torch.save(model.state_dict(), ckpt_path)    
            print('  → 최저 val, 저장') 
model.eval()
print(f'학습 시간: {(time.time() - t0) / 60:.1f}분')
start = torch.zeros((1, 1), dtype = torch.long, device = device)
out = model.generate(start, max_new_tokens = 300)
print(decode(out[0].tolist()))
