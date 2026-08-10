# Chapter 6: The GRU Forward Pass — Full Numerical Walkthrough

## The setup

Same toy sequence, same scalar simplification: $x=[1.0, 2.0, 3.0]$.

| Gate/candidate | $W_x$ | $W_h$ | $b$ |
|---|---|---|---|
| Reset | 0.5 | 0.3 | 0.1 |
| Update | 0.4 | 0.2 | 0.0 |
| Candidate | 0.5 | 0.8 | 0.1 |

(Same candidate weights as the vanilla RNN and LSTM chapters, again purely so the numbers are directly comparable.)

Starting state: $h_0=0$.

## Step 1: $t=1$, $x_1=1.0$

**Reset gate:**
$$z_r = (0.5)(1.0)+(0.3)(0)+0.1 = 0.6 \quad\Rightarrow\quad r_1=\sigma(0.6)\approx0.646$$

**Update gate:**
$$z_z = (0.4)(1.0)+(0.2)(0)+0.0 = 0.4 \quad\Rightarrow\quad z_1=\sigma(0.4)\approx0.599$$

**Candidate** (note: $r_1 \cdot h_0 = r_1\cdot0=0$ regardless of $r_1$'s value, since there's no previous state to reset yet):
$$z_c = (0.5)(1.0)+(0.8)(0)+0.1 = 0.6 \quad\Rightarrow\quad \tilde{h}_1=\tanh(0.6)\approx0.537$$

**Final hidden state:**
$$h_1 = (1-z_1)h_0 + z_1\tilde{h}_1 = (0.401)(0)+(0.599)(0.537) \approx 0.322$$

Worth noticing: at $t=1$, since $h_0=0$, the candidate $\tilde{h}_1$ reduces to exactly the vanilla RNN's $z_1$ computation from Chapter 2 of the RNN curriculum — same 0.6, same $\tanh(0.6)\approx0.537$. The reset gate has nothing to act on yet.

## Step 2: $t=2$, $x_2=2.0$

$$z_r = (0.5)(2.0)+(0.3)(0.322)+0.1 = 1.197 \quad\Rightarrow\quad r_2=\sigma(1.197)\approx0.768$$
$$z_z = (0.4)(2.0)+(0.2)(0.322)+0.0 = 0.864 \quad\Rightarrow\quad z_2=\sigma(0.864)\approx0.704$$

Reset-gated previous state: $r_2\cdot h_1 = (0.768)(0.322)\approx0.247$

$$z_c = (0.5)(2.0)+(0.8)(0.247)+0.1 = 1.298 \quad\Rightarrow\quad \tilde{h}_2=\tanh(1.298)\approx0.861$$
$$h_2 = (1-z_2)h_1+z_2\tilde{h}_2 = (0.296)(0.322)+(0.704)(0.861) \approx 0.095+0.606 = 0.701$$

## Step 3: $t=3$, $x_3=3.0$

$$z_r = (0.5)(3.0)+(0.3)(0.701)+0.1 = 1.810 \quad\Rightarrow\quad r_3=\sigma(1.810)\approx0.859$$
$$z_z = (0.4)(3.0)+(0.2)(0.701)+0.0 = 1.340 \quad\Rightarrow\quad z_3=\sigma(1.340)\approx0.793$$

Reset-gated previous state: $r_3\cdot h_2 = (0.859)(0.701)\approx0.603$

$$z_c = (0.5)(3.0)+(0.8)(0.603)+0.1 = 2.082 \quad\Rightarrow\quad \tilde{h}_3=\tanh(2.082)\approx0.969$$
$$h_3 = (1-z_3)h_2+z_3\tilde{h}_3 = (0.207)(0.701)+(0.793)(0.969) \approx 0.145+0.769 = 0.914$$

## Everything in one table (verified by code)

| $t$ | $r_t$ | $z_t$ | $\tilde{h}_t$ | $h_t$ |
|---|---|---|---|---|
| 1 | 0.646 | 0.599 | 0.537 | 0.322 |
| 2 | 0.768 | 0.704 | 0.861 | 0.701 |
| 3 | 0.859 | 0.793 | 0.969 | 0.914 |

## Three-way comparison: vanilla RNN vs. LSTM vs. GRU

| $t$ | Vanilla RNN $h_t$ | LSTM $h_t$ | GRU $h_t$ |
|---|---|---|---|
| 1 | 0.537 | 0.201 | 0.322 |
| 2 | 0.910 | 0.536 | 0.701 |
| 3 | 0.981 | 0.795 | 0.914 |

All three architectures see the exact same input sequence. The numbers differ because the gating mechanisms (or lack thereof) change how much of each new input gets blended with prior state — this is architecture producing genuinely different computations, not just relabeled math. Notice GRU's values sit between the vanilla RNN's and LSTM's here — again, an artifact of these particular untrained weights, not a general property.

## What to notice about the update gate specifically

Look at $z_t$ across the three steps: 0.599 → 0.704 → 0.793. It's climbing — the network (with these arbitrary weights) is leaning more toward "take the new candidate" as the sequence progresses. Compare this to $(1-z_t)$, the weight on the *old* state: 0.401 → 0.296 → 0.207, shrinking. This single number, $z_t$, is doing the job that took *two* separate gates (forget + input) in LSTM — the direct consequence of GRU's coupled-gate design from Chapter 5.

## What's ahead

Chapter 7 covers bidirectional variants — BiLSTM and BiGRU — running this same toy sequence forward *and* backward, then concatenating. Chapter 8 is the head-to-head comparison of when to use vanilla RNN vs. LSTM vs. GRU vs. bidirectional variants.

---

**One-line summary:** the GRU forward pass computes reset and update gates from $(x_t, h_{t-1})$, uses the reset gate to control how much old state feeds the new candidate, and uses the update gate to linearly blend old state and candidate into the final $h_t$ — one gate doing what took two in LSTM.
