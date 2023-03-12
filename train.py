# read all text into a string

# vocab size: [colors] + special tokens
# should be like 10 + 5

# tokenize the input text
# already have tokens!
# tokenization trade off:
# shorter vocab -> longer sequences
# larger vocab -> shorter sequences
# "hi there" char level encoding -> 8 tokens
# "hi there" gpt2 encoding, 50k vocab size -> 3 tokens


from load_data import load_data, pad_elements
from gpt import ARCFormerModel

import torch

x_train, y_train, x_val, y_val = load_data()

x_train = torch.tensor(pad_elements(x_train, 800), dtype=torch.long)
x_val = torch.tensor(pad_elements(x_val, 800), dtype=torch.long)
y_train = torch.tensor(pad_elements(y_train, 200), dtype=torch.long)
y_val = torch.tensor(pad_elements(y_val, 200), dtype=torch.long)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)



max_iters = 500
eval_interval = 20
eval_iters = 20
learning_rate = 3e-4

batch_size = 8
device = 'cpu'

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    x = x_train if split == 'train' else x_val
    y = y_train if split == 'train' else y_val
    
    ix = torch.randint(len(x), (batch_size,))

    xt = torch.stack([x[i] for i in ix])
    yt = torch.stack([y[i] for i in ix])
    x, y = xt.to(device), yt.to(device)
    return x, y










model =ARCFormerModel()
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            #print('estimate loss', X.shape, Y.shape)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


