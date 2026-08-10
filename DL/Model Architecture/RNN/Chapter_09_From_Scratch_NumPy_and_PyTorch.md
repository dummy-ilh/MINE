# Chapter 9: Building It From Scratch — NumPy, Then PyTorch

## The goal

Every number in this chapter was actually run and verified against Chapters 2 and 3's hand calculations. If you implement this yourself and get different numbers, something's off — use this as your ground truth.

Same setup throughout:

$$x = [1.0,\ 2.0,\ 3.0], \quad y^* = [0.6,\ 0.8,\ 1.0]$$
$$W_{xh}=0.5,\ W_{hh}=0.8,\ b_h=0.1,\ W_{hy}=1.0,\ b_y=0.0$$

## Part 1: NumPy — forward pass

This is a direct translation of the equations from Chapter 2. No shortcuts, no libraries beyond NumPy — every line maps to one term in $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}+b_h)$.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y_star = np.array([0.6, 0.8, 1.0])

W_xh, W_hh, b_h = 0.5, 0.8, 0.1
W_hy, b_y = 1.0, 0.0

h = [0.0]          # h[0] = h_0
z_list = []
y_hat = []

for t in range(3):
    z = W_xh * x[t] + W_hh * h[-1] + b_h
    z_list.append(z)
    h.append(np.tanh(z))
    y_hat.append(W_hy * h[-1] + b_y)

for t in range(3):
    print(f"t={t+1}: z={z_list[t]:.4f}  h={h[t+1]:.4f}  y_hat={y_hat[t]:.4f}")
```

**Verified output:**
```
t=1: z=0.6000  h=0.5370  y_hat=0.5370
t=2: z=1.5296  h=0.9104  y_hat=0.9104
t=3: z=2.3283  h=0.9812  y_hat=0.9812
```

This matches Chapter 2's hand-computed table (0.537, 0.910, 0.981 — small differences are just rounding in the hand calculation).

## Part 2: NumPy — manual BPTT

This directly implements the $D_t$ recursion from Chapter 3: $D_t = (1-h_t^2)(h_{t-1} + W_{hh}\cdot D_{t-1})$.

```python
# D_t = dh_t/dW_hh, computed via the Chapter 3 recursion
D = [0.0]  # D_0 = 0
for t in range(3):
    h_curr, h_prev = h[t+1], h[t]
    Dt = (1 - h_curr**2) * (h_prev + W_hh * D[-1])
    D.append(Dt)

errors = [y_hat[t] - y_star[t] for t in range(3)]

dL_dWhh = sum(errors[t] * D[t+1] for t in range(3))
print(f"dL/dW_hh = {dL_dWhh:.6f}")
```

**Verified output:** `dL/dW_hh = 0.009459` — matches Chapter 3's hand-computed **0.0095** almost exactly.

The same pattern gives you the other three parameters — swap what's added at the "direct path" term inside the recursion:

```python
# dL/dW_xh: swap h_prev for x[t] in the recursion
Dx = [0.0]
for t in range(3):
    Dx.append((1 - h[t+1]**2) * (x[t] + W_hh * Dx[-1]))
dL_dWxh = sum(errors[t] * Dx[t+1] for t in range(3))

# dL/db_h: swap h_prev for 1 (bias has no "input" to multiply)
Db = [0.0]
for t in range(3):
    Db.append((1 - h[t+1]**2) * (1 + W_hh * Db[-1]))
dL_dbh = sum(errors[t] * Db[t+1] for t in range(3))

# dL/dW_hy and dL/db_y don't need the recursion — they only touch the readout layer
dL_dWhy = sum(errors[t] * h[t+1] for t in range(3))
dL_dby  = sum(errors[t] for t in range(3))
```

**Verified output for all five gradients:**

| Parameter | Gradient |
|---|---|
| $\partial L/\partial W_{hh}$ | 0.009459 |
| $\partial L/\partial W_{xh}$ | 0.001409 |
| $\partial L/\partial b_h$ | -0.015990 |
| $\partial L/\partial W_{hy}$ | 0.048198 |
| $\partial L/\partial b_y$ | 0.028593 |

## Part 3: PyTorch — the same computation via autograd

The point of this section: build the *exact same* forward pass manually (not using `nn.RNN`, so every operation is visible), let PyTorch's autograd compute the backward pass, and confirm it matches Part 2 exactly — that's your proof the manual BPTT derivation in Chapter 3 was correct.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y_star = torch.tensor([0.6, 0.8, 1.0])

W_xh = torch.tensor(0.5, requires_grad=True)
W_hh = torch.tensor(0.8, requires_grad=True)
b_h  = torch.tensor(0.1, requires_grad=True)
W_hy = torch.tensor(1.0, requires_grad=True)
b_y  = torch.tensor(0.0, requires_grad=True)

h = torch.tensor(0.0)
losses = []
for t in range(3):
    z = W_xh * x[t] + W_hh * h + b_h
    h = torch.tanh(z)
    y_hat = W_hy * h + b_y
    losses.append(0.5 * (y_hat - y_star[t])**2)

L = sum(losses)
L.backward()   # autograd computes every gradient in one call

print(f"dL/dW_hh = {W_hh.grad.item():.6f}")
print(f"dL/dW_xh = {W_xh.grad.item():.6f}")
print(f"dL/db_h  = {b_h.grad.item():.6f}")
print(f"dL/dW_hy = {W_hy.grad.item():.6f}")
print(f"dL/db_y  = {b_y.grad.item():.6f}")
```

**Verified output:**
```
dL/dW_hh = 0.009459
dL/dW_xh = 0.001409
dL/db_h  = -0.015990
dL/dW_hy = 0.048198
dL/db_y  = 0.028593
```

**Identical to Part 2, to 6 decimal places.** This confirms the recursion from Chapter 3 is exactly what autograd is doing under the hood — `.backward()` isn't magic, it's mechanically applying the same total-derivative recursion, just automatically and for arbitrary graphs instead of one you derived by hand.

## Part 4: the idiomatic PyTorch way (`nn.RNNCell`)

In practice, you won't hand-write the recurrence — you'll use `nn.RNNCell` (one step) or `nn.RNN` (a whole sequence at once). Here's the one-step version, matching the same math:

```python
import torch.nn as nn

cell = nn.RNNCell(input_size=1, hidden_size=1)
# cell.weight_ih ~ W_xh, cell.weight_hh ~ W_hh, cell.bias_ih + cell.bias_hh ~ b_h

h = torch.zeros(1, 1)
for t in range(3):
    x_t = torch.tensor([[x[t]]])
    h = cell(x_t, h)   # one recurrence step, weights shared automatically across the loop
```

`nn.RNN` does this same loop internally for a whole sequence in one call — useful in practice, but Parts 1–3 above are what's actually happening inside it.

## What you should take away from all 9 chapters

1. An RNN is one small shared function, applied in a loop, carrying a hidden state forward (Ch 1–2).
2. BPTT is ordinary backprop on the unrolled graph, with gradients for shared weights summed across every timestep they touched (Ch 3) — and this is *exactly* what PyTorch's `.backward()` computes automatically.
3. That summed gradient is a product of many factors across time, which is why it can vanish or explode over long sequences (Ch 4).
4. Clipping and truncation are compute/stability patches; gating (LSTM/GRU) is the actual architectural fix (Ch 5).
5. The same mechanism supports 7 different input/output shapes, each with different gradient-flow characteristics (Ch 6).
6. You should be able to derive, explain, and defend all of the above in an interview (Ch 7).
7. RNNs occasionally belong on tabular data — specifically when rows are really a variable-length entity history (Ch 8).
8. And now — you've implemented it twice, by hand and via autograd, and confirmed they agree to 6 decimal places.

---

**Curriculum complete.**
