# Chapter 3: The LSTM Forward Pass — Full Numerical Walkthrough

## The setup

Same toy sequence as always: $x = [1.0, 2.0, 3.0]$. Same scalar simplification as the RNN curriculum — every quantity here is a single number, not a vector, so you can verify every step by hand.

New weights, one set per gate (arbitrary, for illustration — not trained):

| Gate | $W_x$ | $W_h$ | $b$ |
|---|---|---|---|
| Forget | 0.5 | 0.3 | 0.1 |
| Input | 0.4 | 0.2 | 0.0 |
| Candidate | 0.5 | 0.8 | 0.1 |
| Output | 0.6 | 0.4 | 0.0 |

(The candidate row reuses the same numbers as the vanilla RNN's $W_{xh}, W_{hh}, b_h$ from the earlier curriculum — purely so you can compare $\tilde{c}_t$ here to $h_t$ there later.)

Starting state: $h_0 = 0$, $c_0 = 0$.

## Step 1: $t=1$, $x_1=1.0$

**Forget gate:**
$$z_f = (0.5)(1.0) + (0.3)(0) + 0.1 = 0.6 \quad\Rightarrow\quad f_1 = \sigma(0.6) \approx 0.646$$

**Input gate:**
$$z_i = (0.4)(1.0) + (0.2)(0) + 0.0 = 0.4 \quad\Rightarrow\quad i_1 = \sigma(0.4) \approx 0.599$$

**Candidate cell state:**
$$z_c = (0.5)(1.0) + (0.8)(0) + 0.1 = 0.6 \quad\Rightarrow\quad \tilde{c}_1 = \tanh(0.6) \approx 0.537$$

**Cell state update:**
$$c_1 = f_1 \cdot c_0 + i_1 \cdot \tilde{c}_1 = (0.646)(0) + (0.599)(0.537) \approx 0.322$$

**Output gate:**
$$z_o = (0.6)(1.0) + (0.4)(0) + 0.0 = 0.6 \quad\Rightarrow\quad o_1 = \sigma(0.6) \approx 0.646$$

**Hidden state:**
$$h_1 = o_1 \cdot \tanh(c_1) = (0.646)(\tanh(0.322)) = (0.646)(0.311) \approx 0.201$$

Notice: $c_0=0$ means the forget gate has nothing to act on yet — same as the vanilla RNN's $D_1=0$ moment from Chapter 3 of the RNN curriculum. All of step 1's memory comes from the input gate letting in the new candidate.

## Step 2: $t=2$, $x_2=2.0$

$$z_f = (0.5)(2.0)+(0.3)(0.201)+0.1 = 1.160 \quad\Rightarrow\quad f_2 = \sigma(1.160)\approx0.761$$
$$z_i = (0.4)(2.0)+(0.2)(0.201)+0.0 = 0.840 \quad\Rightarrow\quad i_2 = \sigma(0.840)\approx0.699$$
$$z_c = (0.5)(2.0)+(0.8)(0.201)+0.1 = 1.261 \quad\Rightarrow\quad \tilde{c}_2 = \tanh(1.261)\approx0.851$$
$$c_2 = f_2 c_1 + i_2\tilde{c}_2 = (0.761)(0.322)+(0.699)(0.851) \approx 0.245+0.595 = 0.839$$
$$z_o = (0.6)(2.0)+(0.4)(0.201)+0.0 = 1.280 \quad\Rightarrow\quad o_2=\sigma(1.280)\approx0.783$$
$$h_2 = o_2\tanh(c_2) = (0.783)(\tanh(0.839)) = (0.783)(0.686) \approx 0.536$$

Notice the forget gate is now higher (0.761 vs. 0.646) — it's letting more of the accumulated memory through as the sequence continues.

## Step 3: $t=3$, $x_3=3.0$

$$z_f = (0.5)(3.0)+(0.3)(0.536)+0.1 = 1.761 \quad\Rightarrow\quad f_3=\sigma(1.761)\approx0.853$$
$$z_i = (0.4)(3.0)+(0.2)(0.536)+0.0 = 1.307 \quad\Rightarrow\quad i_3=\sigma(1.307)\approx0.787$$
$$z_c = (0.5)(3.0)+(0.8)(0.536)+0.1 = 2.029 \quad\Rightarrow\quad \tilde{c}_3=\tanh(2.029)\approx0.966$$
$$c_3 = f_3 c_2 + i_3\tilde{c}_3 = (0.853)(0.839)+(0.787)(0.966) \approx 0.716+0.760 = 1.477$$
$$z_o = (0.6)(3.0)+(0.4)(0.536)+0.0 = 2.015 \quad\Rightarrow\quad o_3=\sigma(2.015)\approx0.882$$
$$h_3 = o_3\tanh(c_3) = (0.882)(\tanh(1.477)) = (0.882)(0.901) \approx 0.795$$

## Everything in one table (verified by code)

| $t$ | $f_t$ | $i_t$ | $\tilde{c}_t$ | $c_t$ | $o_t$ | $h_t$ |
|---|---|---|---|---|---|---|
| 1 | 0.646 | 0.599 | 0.537 | 0.322 | 0.646 | 0.201 |
| 2 | 0.761 | 0.699 | 0.851 | 0.839 | 0.783 | 0.536 |
| 3 | 0.853 | 0.787 | 0.966 | 1.477 | 0.882 | 0.795 |

## Compare to the vanilla RNN on the same input

| $t$ | Vanilla RNN $h_t$ | LSTM $h_t$ | LSTM $c_t$ |
|---|---|---|---|
| 1 | 0.537 | 0.201 | 0.322 |
| 2 | 0.910 | 0.536 | 0.839 |
| 3 | 0.981 | 0.795 | 1.477 |

Two things worth noticing:

1. **The cell state $c_t$ is not bounded between -1 and 1** the way $h_t$ is — it grew past 1.0 by step 3 ($c_3\approx1.477$). This is expected: $c_t$ is a running sum of gated contributions, never squashed directly. Only when it's read out (via $\tanh(c_t)$ in the $h_t$ equation) does it get bounded again.
2. **The LSTM's $h_t$ values are smaller than the vanilla RNN's** here — that's a coincidence of these particular (untrained, arbitrary) weights, not a general rule. Once trained, both architectures adjust their weights to fit the task; the point of this comparison is architecture, not which produces "bigger" numbers.

## What to carry forward

Every gate value in the table above is a sigmoid output between 0 and 1 — you can sanity-check any LSTM computation just by checking gate values are in that range and $\tilde{c}_t$, $h_t$ are in $(-1,1)$ while $c_t$ is unbounded.

## What's ahead

Chapter 4 traces a gradient backward through this exact forward pass and shows numerically why $c_t$'s "mostly-additive, no full squash" update lets gradients survive much further back than the vanilla RNN's product-of-many-shrinking-factors did in Chapter 1.

---

**One-line summary:** the LSTM forward pass computes four gate/candidate values from $(x_t, h_{t-1})$ at each step, uses the forget and input gates to blend the old cell state with new candidate content into $c_t$, then uses the output gate to decide how much of $\tanh(c_t)$ becomes the new hidden state $h_t$.
