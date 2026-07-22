# Vanilla RNN — Complete Interview Reference (L4→L6)

*A self-contained reference: mechanics, full BPTT derivation with eigenvalue analysis, time-series and NLP applications, diagnostics, and a tiered interview Q&A bank. No LSTM/GRU internals — comparison only, for positioning.*

---

## How to use this document

- **Modules 1–5**: core content, rebuilt with full rigor and numeric traces you can recompute by hand.
- **Q&A Bank** at the end of every module: **Tier 1 (L4 — definitional)**, **Tier 2 (L5 — derivation/application)**, **Tier 3 (L6 — systems & tradeoffs)**.
- Every numeric example is traceable — you should be able to redo the arithmetic on a whiteboard in an interview.

---

# Module 1 — Core Fundamentals & Vanilla RNN Mechanics

## 1.1 Why Feedforward Networks Fail on Sequences

A feedforward network computes $y = f(Wx + b)$ — one fixed-size input, one fixed-size output, no notion of "before" or "after." Two structural failures for sequential data:

1. **Fixed input size.** A sentence of 5 words and a sentence of 50 words can't share one input layer.
2. **No memory.** Each input is processed in isolation — $P(y \mid x_t)$ instead of $P(y \mid x_1, \dots, x_t)$.

```
Feedforward:                         Recurrent:
  X ──► [Hidden] ──► Y                x_t ──► [h_t] ──► y_t
  (no notion of order)                          │
                                                 ▼ (h_t feeds into h_{t+1})
```

The RNN's fix: maintain a **hidden state** $h_t$ that is recomputed at every step from both the new input and the *previous* hidden state — a compressed summary of everything seen so far.

## 1.2 Unrolling the RNN

```
          y_1                    y_2                    y_3
           ▲                      ▲                      ▲
           │ W_hy                 │ W_hy                 │ W_hy
       ┌───────┐    W_hh      ┌───────┐    W_hh      ┌───────┐
h_0 ─► │  h_1  │ ───────────► │  h_2  │ ───────────► │  h_3  │
       └───────┘              └───────┘              └───────┘
           ▲                      ▲                      ▲
           │ W_xh                 │ W_xh                 │ W_xh
          x_1                    x_2                    x_3
```

Three weight matrices, **shared across every time step** (this weight-sharing is *the* defining property of an RNN — it's what lets one model handle variable-length sequences):

| Matrix | Role |
|---|---|
| $W_{xh}$ | input → hidden |
| $W_{hh}$ | previous hidden → current hidden |
| $W_{hy}$ | hidden → output |

## 1.3 The Core Equations

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) \tag{1}$$
$$\hat{y}_t = W_{hy} h_t + b_y \quad \text{(or softmax for classification)} \tag{2}$$

**Why $\tanh$ and not ReLU?** Two reasons that come up often in interviews:
- $\tanh$ is bounded in $(-1, 1)$, which keeps the hidden state from blowing up purely from repeated additive accumulation (ReLU has no upper bound, so an unrolled RNN with ReLU is much more exploding-gradient-prone).
- $\tanh$ is zero-centered, which historically gives better-conditioned optimization than sigmoid.

Modern practice does use ReLU-based RNNs (with careful initialization — see §2.5), but $\tanh$ is the vanilla default and the one interviewers expect by default.

## 1.4 Full Numeric Walkthrough (scalar case)

Setup: $D_{in}=1$, $D_{out}=1$, $h_0=0$, $W_{xh}=0.5$, $W_{hh}=0.8$, $W_{hy}=2.0$, biases $=0$. Input $x=[1.0, 2.0]$.

**t=1:**
$$h_1 = \tanh(0.5\cdot1.0 + 0.8\cdot0) = \tanh(0.5) \approx 0.462$$
$$\hat y_1 = 2.0 \times 0.462 = 0.924$$

**t=2:**
$$h_2 = \tanh(0.5\cdot2.0 + 0.8\cdot0.462) = \tanh(1.3696) \approx 0.878$$
$$\hat y_2 = 2.0 \times 0.878 = 1.756$$

$h_2$ mathematically contains a trace of $x_1$ (through the $0.8 \times h_1$ term) — this is the entire mechanism of "memory" in one line.

## 1.5 Vector Case ($D_{in}=3, D_{out}=2$)

$$W_{xh}=\begin{bmatrix}0.2 & 0.1 & -0.3\\0.5 & 0.0 & 0.4\end{bmatrix},\quad W_{hh}=\begin{bmatrix}0.6 & -0.1\\0.2 & 0.8\end{bmatrix},\quad b_h=\begin{bmatrix}0.1\\0.0\end{bmatrix}$$

$x_1 = [1.0, 2.0, 0.5]^T$:

$$W_{xh}x_1 = \begin{bmatrix}0.2+0.2-0.15\\0.5+0+0.2\end{bmatrix}=\begin{bmatrix}0.25\\0.70\end{bmatrix}, \quad h_1=\tanh\begin{bmatrix}0.35\\0.70\end{bmatrix}=\begin{bmatrix}0.336\\0.604\end{bmatrix}$$

$x_2=[0.5,1.0,2.0]^T$, carrying $h_1$ forward:

$$W_{xh}x_2=\begin{bmatrix}-0.40\\1.05\end{bmatrix},\quad W_{hh}h_1=\begin{bmatrix}0.1412\\0.5504\end{bmatrix}$$
$$h_2=\tanh\begin{bmatrix}-0.1588\\1.6004\end{bmatrix}=\begin{bmatrix}-0.1575\\0.9217\end{bmatrix}$$

## 1.6 Shape Cheat Sheet

| Tensor | Shape | Notes |
|---|---|---|
| $X$ (full batch) | $(B, T, D_{in})$ | $B$=batch, $T$=seq len |
| $x_t$ | $(B, D_{in})$ | slice at step $t$ |
| $W_{xh}$ | $(D_{in}, D_{out})$ | shared across $t$ and batch |
| $W_{hh}$ | $(D_{out}, D_{out})$ | shared |
| $h_t$ | $(B, D_{out})$ | |
| $W_{hy}$ | $(D_{out}, K)$ | $K$=output dim |

$$h_t = \tanh(\underbrace{x_t}_{(B,D_{in})} @ \underbrace{W_{xh}}_{(D_{in},D_{out})} + \underbrace{h_{t-1}}_{(B,D_{out})} @ \underbrace{W_{hh}}_{(D_{out},D_{out})} + b_h)$$

## 1.7 Parameter Count (interview favorite)

Total learnable parameters, independent of sequence length $T$:

$$\#params = \underbrace{D_{in}\cdot D_{out}}_{W_{xh}} + \underbrace{D_{out}^2}_{W_{hh}} + \underbrace{D_{out}}_{b_h} + \underbrace{D_{out}\cdot K}_{W_{hy}} + \underbrace{K}_{b_y}$$

This is *the* reason RNNs generalize to arbitrary-length sequences — parameter count is constant in $T$, unlike an MLP that would need $T \times D_{in}$ input weights.

### Module 1 — Q&A Bank

**Tier 1 (L4 — definitional)**
1. Why can't a standard feedforward network process a variable-length sequence directly?
2. What does "weight sharing across time steps" mean, and why does the RNN need it?
3. What is the role of the hidden state $h_t$? Is it a fixed-size vector, and does its size depend on sequence length?
4. Why is $\tanh$ the default nonlinearity in vanilla RNNs rather than ReLU or sigmoid?
5. Given $D_{in}=10, D_{out}=64, K=1$, how many total parameters does a vanilla RNN cell + output layer have?

**Tier 2 (L5 — derivation/application)**
1. Derive the shape of every intermediate tensor in the forward pass for batch size 32, sequence length 20, input dim 8, hidden dim 128.
2. Show numerically (pick your own small weights) that $h_2$ has a nonzero gradient with respect to $x_1$. What does this prove about "memory"?
3. Why is $h_t$ described as a "compressed summary" — what information is necessarily lost at each step, and why can't that be avoided in the vanilla architecture?
4. If you doubled $D_{out}$ (hidden dim) with everything else fixed, how does parameter count scale? Which matrix dominates the parameter count as $D_{out}$ grows large?

**Tier 3 (L6 — systems & tradeoffs)**
1. You have a production system doing real-time sensor forecasting on an edge device with hidden dim constrained to 16 due to memory limits. Discuss the modeling tradeoffs of such a small hidden state versus latency/memory budgets.
2. Contrast the inductive bias of an RNN's parameter-sharing-over-time with a 1D CNN's parameter-sharing-over-space. When would each be the more natural prior?
3. Why does the RNN's fixed-size hidden state impose a hard theoretical limit on how much sequence information can be retained, regardless of training quality? (Hint: think information-theoretically about compressing an unboundedly long sequence into a fixed-size vector.)

---

# Module 2 — Training Vanilla RNNs & Gradient Dynamics

## 2.1 BPTT: The Setup

Unroll the network, treat each time step like a "layer," and backpropagate as usual — but every unrolled layer shares the *same* physical weights.

```
          L_1        L_2        L_3
           ▲          ▲          ▲
          y_1        y_2        y_3
           ▲          ▲          ▲
h_0 ──►  [h_1] ───► [h_2] ───► [h_3]
           ▲          ▲          ▲
          x_1        x_2        x_3
```

Total loss: $L = \sum_{t=1}^{T} L_t$. Because all $W_{hh}, W_{xh}, W_{hy}$ are literally the *same* tensor reused at every step, the true gradient is the **sum of the gradient contributions from every time step**:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T}\frac{\partial L_t}{\partial W_{hh}}$$

## 2.2 Deriving the Vanishing/Exploding Gradient (rigorously)

Focus on how $W_{hh}$ affected loss at $t=3$:

$$\frac{\partial L_3}{\partial W_{hh}} = \frac{\partial L_3}{\partial h_3}\cdot\frac{\partial h_3}{\partial W_{hh}}$$

But $h_3$ depends on $W_{hh}$ **both directly** (in the equation for $h_3$) **and indirectly** through $h_2$ (which itself used $W_{hh}$), which depends on $h_1$, etc. Summing all paths:

$$\frac{\partial L_3}{\partial W_{hh}} = \sum_{k=1}^{3}\frac{\partial L_3}{\partial h_3}\cdot\frac{\partial h_3}{\partial h_k}\cdot\frac{\partial h_k}{\partial W_{hh}}$$

The critical term is $\dfrac{\partial h_3}{\partial h_k}$ — how a hidden state 2 steps back affects the current one. By the chain rule:

$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^{T}\frac{\partial h_j}{\partial h_{j-1}}$$

### The Jacobian term, expanded properly

From $h_j = \tanh(W_{xh}x_j + W_{hh}h_{j-1}+b_h)$:

$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(1-\tanh^2(z_j)\big)\cdot W_{hh}$$

where $z_j = W_{xh}x_j + W_{hh}h_{j-1}+b_h$ is the pre-activation. Call $D_j = \text{diag}(1-\tanh^2(z_j))$ — a diagonal matrix whose entries are **always in $[0,1]$** (since $\tanh' \le 1$, with equality only at $z=0$).

So the full backward Jacobian over $T$ steps is:

$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^{T} D_j \, W_{hh}$$

**This is the crux of the vanishing/exploding gradient problem**: you are multiplying together $T-1$ copies of $D_j W_{hh}$. Since $D_j \le 1$ always shrinks things, the entire fate of the gradient rests on the **spectral radius (largest eigenvalue magnitude) of $W_{hh}$**:

- If $|\lambda_{max}(W_{hh})| \cdot \|D_j\|< 1$ → the product shrinks geometrically → **vanishing gradient**.
- If $|\lambda_{max}(W_{hh})|$ is large enough to counteract the $D_j$ shrinkage → **exploding gradient**.

In practice, because $D_j \le 1$ always pulls the product down, **vanishing is the much more common failure mode** than exploding for vanilla RNNs — exploding usually needs deliberately large or poorly-initialized $W_{hh}$.

## 2.3 Scalar Intuition (concrete numbers)

Ignoring $\tanh'$ and treating $W_{hh}$ as scalar, over $T=100$ steps:

| $W_{hh}$ | $(W_{hh})^{100}$ | Regime |
|---|---|---|
| 0.8 | $\approx 2.03\times10^{-10}$ | vanishing |
| 0.95 | $\approx 0.0059$ | vanishing (slower) |
| 1.0 | $1.0$ | stable (unstable in practice — knife-edge) |
| 1.05 | $\approx 131.5$ | mild exploding |
| 1.2 | $\approx 8.28\times10^{7}$ | severe exploding |

Notice how close $0.95$ and $1.05$ are as *numbers*, yet after 100 steps they diverge by nearly 5 orders of magnitude — this is the "knife-edge" nature of recurrent dynamics that makes vanilla RNN training so fragile, and is exactly why interviewers ask you to reason about eigenvalues rather than just "weights being small or big."

## 2.4 Complexity Analysis (frequently asked, rarely covered)

For a sequence of length $T$, hidden dim $H$, input dim $D$:

- **Per-step compute:** $O(D H + H^2)$ (the two matmuls in Eq. 1) — dominated by $H^2$ for large hidden dims.
- **Total forward pass:** $O(T(DH+H^2))$.
- **Memory for BPTT:** you must **cache every intermediate hidden state** $h_1, \dots, h_T$ (and pre-activations) to compute gradients on the backward pass → $O(TH)$ memory, growing linearly with sequence length.
- **Critical structural limitation:** the recurrence $h_t = f(h_{t-1})$ makes step $t$ depend on step $t-1$'s output — **this computation is inherently sequential and cannot be parallelized across the time dimension**, unlike a Transformer's self-attention, which computes all positions' representations in parallel. This is the single biggest architectural reason the field moved away from RNNs toward attention-based models for long sequences at scale — not just the vanishing gradient, but the wall-clock training cost of sequential unrolling.

## 2.5 Mitigation Strategies

| Strategy | Mechanism | Fixes |
|---|---|---|
| **Gradient clipping** | Rescale $\mathbf g \leftarrow \min\!\left(1,\frac{c}{\|\mathbf g\|}\right)\mathbf g$ if $\|\mathbf g\|>c$ | Exploding only |
| **Truncated BPTT** | Backprop only $k$ steps instead of full $T$ | Compute/memory blowup, exploding gradients over huge $T$ (doesn't fix vanishing — actually caps how far back you *can* learn) |
| **Orthogonal/identity init of $W_{hh}$** | Forces $|\lambda(W_{hh})|\approx 1$ at initialization | Both, at the start of training (doesn't guarantee it stays that way) |
| **Gated architectures (LSTM/GRU)** | Additive/gated memory cell bypasses repeated matrix multiplication | The structural cause of vanishing (out of scope here — see comparison table §5.4) |

### Manual BPTT — a fully worked 2-step numeric example (no autograd)

Using §1.4's setup ($W_{xh}=0.5, W_{hh}=0.8, W_{hy}=2.0$, all biases 0), suppose true targets are $y_1^*=1.0, y_2^*=2.0$ and loss is MSE, $L_t = \tfrac12(\hat y_t - y_t^*)^2$.

**Forward** (from §1.4): $h_1=0.462, \hat y_1=0.924$; $h_2=0.878, \hat y_2=1.756$.

**Step 1 — output-layer gradients:**
$$\frac{\partial L_1}{\partial \hat y_1} = \hat y_1 - y_1^* = 0.924-1.0=-0.076,\qquad \frac{\partial L_2}{\partial \hat y_2}=1.756-2.0=-0.244$$

**Step 2 — gradient w.r.t. $W_{hy}$** (accumulate over both steps, since $W_{hy}$ is shared):
$$\frac{\partial L}{\partial W_{hy}} = \sum_t \frac{\partial L_t}{\partial \hat y_t}\cdot h_t = (-0.076)(0.462) + (-0.244)(0.878) = -0.0351 - 0.2142 = -0.2493$$

**Step 3 — backprop into hidden states.** $\partial \hat y_t/\partial h_t = W_{hy}=2.0$, so the *direct* loss gradient reaching each $h_t$ is:
$$\delta_2^{direct} = \frac{\partial L_2}{\partial \hat y_2}\cdot W_{hy} = -0.244\times2.0=-0.488$$
$$\delta_1^{direct} = \frac{\partial L_1}{\partial \hat y_1}\cdot W_{hy} = -0.076\times2.0=-0.152$$

**Step 4 — propagate $\delta_2$ backward into $h_1$** through the recurrence. Recall $1-\tanh^2(z_2) = 1-h_2^2 = 1-0.878^2=0.229$:
$$\delta_1^{from\ t=2} = \delta_2^{direct}\cdot(1-h_2^2)\cdot W_{hh} = -0.488\times0.229\times0.8 = -0.0894$$

**Step 5 — total gradient reaching $h_1$** (direct + carried-back from $t=2$):
$$\delta_1^{total} = \delta_1^{direct} + \delta_1^{from\ t=2} = -0.152 + (-0.0894) = -0.2414$$

**Step 6 — gradient w.r.t. $W_{hh}$** at each step, using $1-h_1^2=1-0.462^2=0.7866$:
$$\frac{\partial L}{\partial W_{hh}}\Big|_{t=2} = \delta_2^{direct}\cdot(1-h_2^2)\cdot h_1 = -0.488\times0.229\times0.462=-0.0516$$
$$\frac{\partial L}{\partial W_{hh}}\Big|_{t=1} = \delta_1^{total}\cdot(1-h_1^2)\cdot h_0 = -0.2414\times0.7866\times0 = 0 \ \ (\text{since } h_0=0)$$
$$\frac{\partial L}{\partial W_{hh}} = -0.0516 + 0 = -0.0516$$

This is exactly what `loss.backward()` computes internally — walking it by hand once is the best interview prep there is for "explain BPTT" questions, since it forces you to track *which* gradient is direct vs. carried-back at each node.

### Module 2 — Q&A Bank

**Tier 1 (L4)**
1. What is BPTT, and how does it differ from standard backpropagation in a feedforward net?
2. In one sentence, what causes vanishing gradients in vanilla RNNs?
3. What does gradient clipping do, and which problem (vanishing or exploding) does it solve?
4. Why do we need to cache all intermediate hidden states during the forward pass before running BPTT?

**Tier 2 (L5)**
1. Derive $\partial h_T/\partial h_1$ symbolically and explain why it involves a *product* of Jacobians rather than a sum.
2. Explain precisely why $\text{diag}(1-\tanh^2(z))$ has entries in $[0,1]$, and why that means $\tanh$ itself contributes to (but doesn't singlehandedly cause) vanishing.
3. Given $W_{hh}$ has eigenvalues $\{0.9, 1.3, -0.5\}$, what do you expect to happen to gradients over 50 steps, and why does the *largest-magnitude* eigenvalue dominate the long-run behavior?
4. Walk through, with your own small numeric example, how the gradient w.r.t. $W_{hh}$ accumulates contributions from every time step, not just the final one.
5. Why does truncated BPTT reduce compute/memory but *not* actually solve vanishing gradients over the untruncated horizon?

**Tier 3 (L6)**
1. You're training a vanilla RNN and observe the loss oscillating wildly then hitting NaN after a few hundred steps. Walk through your full debugging process, including what you'd log and what hyperparameters you'd change first.
2. Compare the vanishing gradient problem's root cause to why very deep feedforward networks (pre-ResNet) also suffered from vanishing gradients. What's structurally the same, and what's different about the RNN case (hint: shared weights vs. per-layer weights)?
3. Explain why orthogonal initialization of $W_{hh}$ helps *at initialization* but doesn't guarantee gradient stability throughout training. What causes eigenvalues to drift as training proceeds?
4. In a system requiring both training-time efficiency and inference-time streaming (one token at a time, no re-computation), argue for or against a vanilla RNN versus a Transformer, considering the sequential-dependency argument from §2.4.

---

# Module 3 — Vanilla RNNs for Time Series

## 3.1 Sliding Window Framing

Raw series → supervised (X, y) pairs via a lookback window $L$:

Raw: `[10, 12, 15, 18, 20, 22, 25, 28, 30]`, $L=3$:

| Window | $X$ | $y$ |
|---|---|---|
| 0 | `[10,12,15]` | `18` |
| 1 | `[12,15,18]` | `20` |
| 2 | `[15,18,20]` | `22` |
| 3 | `[18,20,22]` | `25` |

Batched shape: $(B, L, D_{in})$ — e.g. $(32, 30, 3)$ for 32 windows, 30-day lookback, 3 features.

## 3.2 Many-to-One vs. Many-to-Many

```
Many-to-One:                          Many-to-Many:
x_1─►[h_1]                             x_1─►[h_1]─►[Linear]─►ŷ_2
x_2─►[h_2]                             x_2─►[h_2]─►[Linear]─►ŷ_3
x_3─►[h_3]─►[Linear]─►ŷ_4              x_3─►[h_3]─►[Linear]─►ŷ_4
(only final h used)                    (every h used)
```

**Interview trap:** many-to-many time-series forecasting is *not* the same as seq2seq — it's synchronized (prediction at every input step), whereas seq2seq has a separate encode phase then decode phase (§4 in the original BPTT-by-architecture breakdown, kept below in §3.4).

## 3.3 From-Scratch PyTorch (Many-to-One)

```python
import torch
import torch.nn as nn

class CustomVanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_dim))
        self.W_hy = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b_y  = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)
        return h_t @ self.W_hy + self.b_y   # many-to-one: only final h_t used
```

Training loop with gradient clipping:

```python
model = CustomVanillaRNN(input_dim=3, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

optimizer.zero_grad()
y_pred = model(X_batch)                 # (B, T, D_in) -> (B, 1)
loss = criterion(y_pred, y_true)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 3.4 BPTT Flow Depends on Architecture

| Architecture | Loss points | Gradient entry | Example |
|---|---|---|---|
| Many-to-One | single $L_T$ | enters only at $T$, flows back to 1 | sentiment classification, next-step forecast |
| Many-to-Many (sync) | $L_t$ every step | enters at every step, accumulates backward | POS tagging, continuous forecasting |
| Seq2Seq (unsync) | $L_t$ over decoder steps only | flows through decoder, then into encoder via context vector | machine translation |
| One-to-Many | $L_t$ over generated steps | flows back through generated sequence to the seed input | image captioning |

For many-to-many (sync), the gradient reaching $h_3$ is a **sum of the local loss gradient and the gradient carried back from $h_4$**:
$$\text{grad at } h_3 = \frac{\partial L_3}{\partial h_3} + \frac{\partial L_{later}}{\partial h_4}\cdot\frac{\partial h_4}{\partial h_3}$$
— this is exactly the mechanism used in the manual BPTT walkthrough in §2.5, generalized to more steps.

### Module 3 — Q&A Bank

**Tier 1 (L4)**
1. Why must raw time series be reframed into (X, y) windows before training an RNN?
2. What's the difference between many-to-one and many-to-many output configurations?
3. Why is gradient clipping applied *after* `loss.backward()` but *before* `optimizer.step()`?

**Tier 2 (L5)**
1. For many-to-many synchronized forecasting, explain why the gradient at an intermediate hidden state is a *sum* of two terms, and derive what each term represents.
2. If lookback window $L$ is too small, what specific failure mode do you expect at inference time, and how would you diagnose it from validation metrics alone?
3. You have highly non-stationary time series (trend + seasonality). Discuss how you'd preprocess before feeding into a vanilla RNN, and why the vanishing gradient problem interacts badly with long seasonal periods (e.g., yearly seasonality with daily granularity → sequence length 365).

**Tier 3 (L6)**
1. Design a production forecasting system where lookback needs to span 500 steps, but you know vanilla RNN effective memory is ~10 steps. What architectural or feature-engineering changes would you propose, and why?
2. Compare many-to-many synchronized RNN forecasting to a direct multi-step forecasting approach (train separate models per horizon). Discuss error accumulation in autoregressive multi-step RNN forecasting vs. the direct approach.

---

# Module 4 — Vanilla RNNs for Text & NLP

## 4.1 Text → Tensors Pipeline

```
"hello" → vocab {h:0,e:1,l:2,o:3} → indices [0,1,2,2,3] → one-hot OR embedding
```

**One-hot:** sparse, dimension = vocab size, no notion of similarity between tokens.
**Learned embeddings:** dense, trainable lookup table, dimension $E$ chosen independently of vocab size — this is what's used in practice.

Shape progression: $(B, T) \xrightarrow{\text{embedding}} (B, T, E)$ — matches the general $(B,T,D_{in})$ shape used throughout this document.

## 4.2 Classic Tasks

- **Text classification (many-to-one):** only final $h_T$ feeds a linear+softmax head.
- **Character-level language modeling (many-to-many, autoregressive):** predict the *next* token at every step; at generation time, feed the model's own output back in as the next input.

```
Target:      'e'        'l'        'l'        'o'
              ▲          ▲          ▲          ▲
          Softmax    Softmax    Softmax    Softmax
              │          │          │          │
h_0 ─────► [h_1] ────► [h_2] ────► [h_3] ────► [h_4]
              ▲          ▲          ▲          ▲
            'h'        'e'        'l'        'l'
```

## 4.3 From-Scratch Character RNN

```python
import torch
import torch.nn as nn

class CharVanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.W_xh = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h_prev=None):
        batch_size, seq_len = x.shape
        embeds = self.embedding(x)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        logits_list, h_t = [], h_prev
        for t in range(seq_len):
            h_t = torch.tanh(self.W_xh(embeds[:, t, :]) + self.W_hh(h_t))
            logits_list.append(self.fc_out(h_t))
        return torch.stack(logits_list, dim=1), h_t
```

Training step (input shifted right by one relative to target — the standard LM setup):

```python
input_text, target_text = text[:-1], text[1:]
logits, _ = model(input_tensor)
loss = criterion(logits.view(-1, vocab_size), target_tensor.view(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Autoregressive sampling:** feed seed char → get $h_1$ + distribution over next char → sample → feed sampled char + $h_1$ back in → repeat. This is exactly the "one-to-many" BPTT pattern applied at *inference* time (no gradient involved, but the same recurrence).

### Module 4 — Q&A Bank

**Tier 1 (L4)**
1. Why do we use embeddings instead of one-hot vectors for text input to an RNN?
2. In character-level language modeling, why is the target sequence the input sequence shifted by one position?
3. What loss function is standard for next-token prediction, and why (vs. MSE)?

**Tier 2 (L5)**
1. Explain why $(B,T,E)$ for text and $(B,T,D_{in})$ for time series are structurally identical from the RNN's point of view — what does this imply about code reuse?
2. During autoregressive generation, why does sampling (vs. always taking argmax) matter for output diversity, and how does temperature scaling affect the softmax distribution before sampling?
3. For sentiment classification (many-to-one) over long reviews (200+ tokens), explain concretely why vanilla RNNs underperform on this task, connecting it back to the ~5-10 step effective memory horizon from Module 5.

**Tier 3 (L6)**
1. You need to deploy a character-level generator with strict latency budgets on-device. Argue for vanilla RNN vs. a small Transformer, considering both the sequential-inference argument (§2.4) and the effective-memory-horizon argument (§5.1).
2. Explain how teacher forcing (feeding ground-truth $y_{t-1}$ during training vs. the model's own prediction) creates a train/inference mismatch ("exposure bias"), and why this problem is specific to autoregressive sequence models generally, not just RNNs.

---

# Module 5 — Diagnostics, Practical Limits & Positioning vs. Gated Architectures

## 5.1 Effective Memory Horizon

Despite theoretically unbounded context, vanilla RNNs have an **empirical effective memory of ~5-10 time steps** before gradient contributions become negligible — a direct consequence of the repeated $D_j W_{hh}$ multiplication derived in §2.2.

```
Step 1   Step 2  ... Step 10   Step 11 ... Step 50
  │        │             │         │            │
  ▼        ▼             ▼         ▼            ▼
[High impact]      [Fading context]    [~Zero context]
```

## 5.2 Diagnostic Tooling: Gradient Norm Tracking

```python
def compute_grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    return total_sq ** 0.5

loss.backward()
raw_norm = compute_grad_norm(model)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if raw_norm < 1e-4:
    print(f"[Warning] Vanishing gradient: norm={raw_norm:.6f}")
elif raw_norm > 10.0:
    print(f"[Warning] Exploding gradient: norm={raw_norm:.6f}")
```

**More advanced diagnostic (L5/L6-level):** track gradient norm **per time step** (not just globally) by hooking into each $h_t$ — if you plot $\|\partial L/\partial h_t\|$ against $t$, a healthy RNN shows gradual decay; a severely vanishing one shows a near step-function collapse to ~0 within just a few steps back from $T$.

## 5.3 Diagnostic Matrix

| Symptom | Cause | Sign | Fix |
|---|---|---|---|
| Loss → NaN/Inf | Exploding gradients | grad norm > 100, rising | Gradient clipping (max_norm=1.0) |
| Ignores long-range context | Vanishing gradients | grad norm ≈ 0 for early steps | Shorten sequence / truncated BPTT / switch to gated architecture |
| Training stalls early | Poor init | activations collapse to 0 | Orthogonal init of $W_{hh}$ |
| Overfits short sequences | High param density relative to data | val loss diverges from train loss | Dropout, weight decay |

## 5.4 Vanilla RNN vs. LSTM/GRU — Positioning Table

*(Comparison only — internal gate mechanics are out of scope for this document, but you should be able to name this table cold in an interview.)*

| | Vanilla RNN | GRU | LSTM |
|---|---|---|---|
| Matmuls per step | 1 | 3 | 4 |
| Gates | none | reset, update | input, forget, output |
| Vanishing gradient | severe (multiplicative only) | mitigated (additive update path) | mitigated (additive cell state + forget gate) |
| Params (rough, same $H$) | $O(H^2)$ | $O(3H^2)$ | $O(4H^2)$ |
| Effective memory | ~5-10 steps | ~50-100 steps | ~50-100+ steps |
| Inference latency | lowest | medium | highest (of the three) |
| When to prefer | tinyML/edge, ultra-short sequences, latency-critical streaming | good default gated RNN, cheaper than LSTM, similar performance in practice | when you need maximal long-range gating control (input/output/forget separated) |

**The one-line interview answer for "why LSTM over vanilla RNN":** LSTMs replace the *multiplicative* recurrence (repeated $W_{hh}$ multiplication, which vanishes/explodes) with an *additive* cell-state update gated by learned scalars — additive paths preserve gradient magnitude far better than multiplicative ones across many time steps.

## 5.5 Where Vanilla RNNs Still Win

1. **Latency:** 1 matmul/step vs. 3-4 for gated variants.
2. **Edge/tinyML deployment:** smallest possible parameter footprint.
3. **True streaming inference:** naturally suited to processing one sample at a time without recomputation, when context needed is genuinely short.

## 5.6 Master Formula Sheet

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}+b_h) \qquad \hat y_t = W_{hy}h_t+b_y$$
$$\frac{\partial h_T}{\partial h_1} = \prod_{j=2}^{T}\text{diag}(1-\tanh^2(z_j))\,W_{hh}$$
$$\mathbf g \leftarrow \min\!\left(1,\frac{c}{\|\mathbf g\|}\right)\mathbf g \quad \text{(gradient clipping)}$$
$$\#params = D_{in}D_{out} + D_{out}^2 + D_{out} + D_{out}K + K$$

### Module 5 — Q&A Bank

**Tier 1 (L4)**
1. Why is vanilla RNN's effective memory only ~5-10 steps despite theoretically unbounded recurrence?
2. What's one concrete symptom that tells you your RNN is suffering from exploding (not vanishing) gradients?
3. Name one scenario where a vanilla RNN is still the *better* engineering choice over an LSTM.

**Tier 2 (L5)**
1. Explain in one sentence why LSTMs/GRUs mitigate vanishing gradients structurally — what's different about their recurrence compared to Eq. (1)?
2. Design a lightweight diagnostic you'd add to a training loop to catch vanishing gradients *before* validation metrics reveal the problem.
3. Given the parameter-count table in §5.4, if latency budget allows only 1.5× the vanilla RNN's per-step compute, could you fit a GRU? Justify with the multiplier.

**Tier 3 (L6)**
1. You inherit a production model that's a vanilla RNN quietly underperforming on long documents. Leadership wants a fix shipped this sprint without a full re-architecture. What incremental changes would you try first, in what order, and what would make you escalate to "we need to switch to LSTM/Transformer"?
2. Walk through the complete failure-mode-to-fix decision tree you'd use live in an interview whiteboard session, starting from "my vanilla RNN's loss just went to NaN."
3. Defend, with the eigenvalue/spectral-radius argument from §2.2, why "just use a bigger hidden dimension" does *not* solve vanishing gradients — what specifically about $D_{out}$ increasing does and doesn't change about the $W_{hh}$ eigenvalue spectrum?

---

## Final Consolidated Checklist

1. **Mechanics:** shared weights across time, $h_t$ = compressed running summary, parameter count independent of $T$.
2. **BPTT:** gradient at any weight = sum over all time steps' local contributions; recurrent Jacobian is a *product* of per-step Jacobians → this product's stability is governed by $W_{hh}$'s eigenvalues.
3. **Vanishing >> exploding** in practice for vanilla RNNs, because $\tanh'\le 1$ always pulls the product down; exploding requires the eigenvalues to be large enough to overcome that.
4. **Complexity:** $O(T(DH+H^2))$ compute, $O(TH)$ memory, and — critically — **inherently sequential**, which is the deeper reason the field moved to attention.
5. **Effective memory ≈ 5-10 steps** — know this number cold, it's asked constantly.
6. **LSTM/GRU fix vanishing via additive (not multiplicative) recurrence** — you don't need their internals for this doc, but you must be able to say this sentence correctly.
7. Always be able to **derive BPTT by hand for a 2-3 step toy example** — this is the single highest-leverage whiteboard exercise for this topic.
