# Chapter 7: Bidirectional Variants — BiLSTM & BiGRU

## The motivation

Every architecture so far reads left to right — $h_t$ only knows about $x_1, \dots, x_t$. It has no idea what comes *after* timestep $t$. For some tasks that's exactly right (you can't peek at the future when generating text one word at a time). But for many tasks, the entire sequence is available upfront, and **future context matters just as much as past context**.

Example: filling in a masked word — "The ___ was barking loudly" — you need what came before ("The") *and* what came after ("was barking loudly") to guess "dog." A left-to-right-only model is blind to the second half.

## The fix: run two RNNs, one each direction

A bidirectional RNN (of any gated flavor — vanilla, LSTM, or GRU) is really **two separate recurrent networks**, with their own independent weights:

- A **forward** network, reading $x_1 \to x_2 \to x_3$, producing $\overrightarrow{h_1}, \overrightarrow{h_2}, \overrightarrow{h_3}$ — exactly what Chapters 3 and 6 already computed.
- A **backward** network, reading the sequence in reverse, $x_3 \to x_2 \to x_1$, producing its own hidden states, which get **realigned** to the original timestep order: $\overleftarrow{h_1}, \overleftarrow{h_2}, \overleftarrow{h_3}$.

At each timestep $t$, the two are concatenated:

$$h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$$

That's the entire idea. No new gate math — BiLSTM is just "run an LSTM forward, run a second LSTM backward, glue their outputs together at each timestep." Same for BiGRU with GRUs.

## Important: realignment

The backward network processes $x_3$ first, then $x_2$, then $x_1$ — so its *first* computed hidden state corresponds to the *last* real timestep. Before concatenating, you have to reverse the backward network's output list so its values line up with the same $t$ as the forward network's.

```
Forward pass:   x1 -> x2 -> x3        (produces h_fwd at t=1,2,3, in order)
Backward pass:  x3 -> x2 -> x1        (produces h_bwd at t=3,2,1, in that order)
Realign:        reverse the backward list -> h_bwd at t=1,2,3
Concatenate:    h_t = [h_fwd_t ; h_bwd_t]   for each t
```

## Worked example: BiLSTM on the toy sequence

Using the exact same LSTM weights and gate equations from Chapters 2–3, run the forward pass on $x=[1.0,2.0,3.0]$ (already computed in Chapter 3), and a **second** pass on the reversed sequence $x_{\text{rev}}=[3.0,2.0,1.0]$ — using, for simplicity here, the same weight values for both directions. (In a real trained BiLSTM, the forward and backward networks have their **own independent weights**, learned separately — reusing weights here is purely to keep the arithmetic traceable; note this explicitly since it's a simplification, not a general rule.)

**Forward hidden states** (from Chapter 3): $\overrightarrow{h_1}=0.201,\ \overrightarrow{h_2}=0.536,\ \overrightarrow{h_3}=0.795$

**Backward pass**, run on $x_{\text{rev}}=[3.0,2.0,1.0]$ starting fresh from $h_0=0,c_0=0$, using the identical LSTM equations:

| reverse-step | input | raw backward $h$ |
|---|---|---|
| 1st | $x_3=3.0$ | 0.614 |
| 2nd | $x_2=2.0$ | 0.670 |
| 3rd | $x_1=1.0$ | 0.523 |

Realigned to original timestep order (reverse the list — the 1st computed value belongs to $t=3$, the 3rd computed value belongs to $t=1$):

$$\overleftarrow{h_1}=0.523, \quad \overleftarrow{h_2}=0.670, \quad \overleftarrow{h_3}=0.614$$

**Concatenated BiLSTM output at each timestep:**

| $t$ | $\overrightarrow{h_t}$ | $\overleftarrow{h_t}$ | $h_t = [\overrightarrow{h_t};\overleftarrow{h_t}]$ |
|---|---|---|---|
| 1 | 0.201 | 0.523 | [0.201, 0.523] |
| 2 | 0.536 | 0.670 | [0.536, 0.670] |
| 3 | 0.795 | 0.614 | [0.795, 0.614] |

At $t=1$, the forward component only knows about $x_1$ — but the backward component already encodes information from the *entire* sequence (it started from $x_3$). That's the entire value proposition: even the earliest timestep's output is informed by everything that comes later.

## Worked example: BiGRU on the toy sequence

Same procedure, GRU equations from Chapters 5–6.

**Forward hidden states** (from Chapter 6): $\overrightarrow{h_1}=0.322,\ \overrightarrow{h_2}=0.701,\ \overrightarrow{h_3}=0.914$

**Backward pass** on $x_{\text{rev}}=[3.0,2.0,1.0]$:

| reverse-step | input | raw backward $h$ |
|---|---|---|
| 1st | $x_3=3.0$ | 0.816 |
| 2nd | $x_2=2.0$ | 0.856 |
| 3rd | $x_1=1.0$ | 0.708 |

Realigned: $\overleftarrow{h_1}=0.708,\ \overleftarrow{h_2}=0.856,\ \overleftarrow{h_3}=0.816$

**Concatenated BiGRU output:**

| $t$ | $\overrightarrow{h_t}$ | $\overleftarrow{h_t}$ | $h_t=[\overrightarrow{h_t};\overleftarrow{h_t}]$ |
|---|---|---|---|
| 1 | 0.322 | 0.708 | [0.322, 0.708] |
| 2 | 0.701 | 0.856 | [0.701, 0.856] |
| 3 | 0.914 | 0.816 | [0.914, 0.816] |

## What changes about BPTT here

Nothing new mechanically — as covered in the RNN curriculum's Chapter 6 (architecture shapes), the forward and backward networks are **two completely independent BPTT computations**. Gradients don't cross between them during backprop; they only interact where their outputs get concatenated and fed into a shared downstream layer (e.g., a final classifier). Train them exactly as you'd train two separate LSTMs (or GRUs), then combine at the readout.

## The cost

Bidirectional doubles the parameters (two full networks instead of one) and, more importantly, **requires the entire sequence upfront** — you can't run the backward pass until you've seen the last element. This rules out bidirectional architectures for real-time / streaming prediction (e.g., live speech transcription as it's spoken) but makes them a strong choice whenever the full sequence is available before you need an answer (e.g., tagging a complete sentence, classifying a finished document).

## What's ahead

Chapter 8 is the head-to-head decision guide: vanilla RNN vs. LSTM vs. GRU vs. bidirectional variants, plus common tweaks (peephole connections, coupled gates, minimal GRU, stacked BiLSTM).

---

**One-line summary:** a bidirectional RNN (Bi-anything) is two independent networks — one reading forward, one reading backward — whose hidden states are concatenated at each timestep, giving every position access to both past and future context, at the cost of needing the full sequence upfront and roughly double the parameters.
