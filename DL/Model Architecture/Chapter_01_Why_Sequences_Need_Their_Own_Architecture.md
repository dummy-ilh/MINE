# Chapter 1: Why Sequences Need Their Own Architecture

## The problem, in one example

Say you want a network to understand these two phrases:

- "not good"
- "good not"

If you just average the word embeddings and feed that into a normal neural network (an MLP), both phrases produce **the exact same input**. The network can't tell them apart. But they don't mean the same thing.

The issue: an MLP has no concept of order. It sees a bag of numbers, not a sequence.

## The second problem: fixed size

An MLP has a fixed number of input slots. Say 10. But sentences aren't always 10 words. Some are 3, some are 300. Stock histories can be 5 days or 5 years long. Audio clips can be 1 second or 1 hour.

You can't hard-code an input size for something that has no fixed size.

So we need an architecture with two properties:

1. **It handles any length.**
2. **It knows the order things came in.**

## The fix: read one piece at a time, keep a running summary

Think about how you read a sentence. You don't take in the whole thing at once. You read word 1, form an impression. You read word 2, update that impression. Word 3, update again. By the end, you have a summary of the whole sentence — built incrementally.

That "impression I'm carrying as I go" is called the **hidden state**. It's just a vector of numbers that gets updated at every step.

That's the entire idea of a Recurrent Neural Network:

> Read one input. Update a running summary. Repeat.

## The RNN as a loop

Here's the whole architecture, before any equations:

```
hidden_state = 0                      # nothing read yet

for each input x_t in the sequence:
    hidden_state = update(hidden_state, x_t)   # fold new input into the summary
    output = readout(hidden_state)             # optionally produce a prediction here
```

Two functions: `update` (folds a new input into the summary) and `readout` (turns the summary into a prediction). Same two functions, used at every single timestep. That reuse is the key design choice — you're not learning a new function for step 47 versus step 2. One function, applied repeatedly.

## Making `update` concrete

`update` is a small neural network layer. In the simplest ("vanilla") RNN, it looks like this:

$$h_t = \tanh(W_{xh} \, x_t + W_{hh} \, h_{t-1} + b_h)$$

Read it left to right:

- $x_t$ — the input at this timestep.
- $h_{t-1}$ — the summary carried over from the previous timestep.
- $W_{xh}$ — a matrix that says "here's how much the current input matters."
- $W_{hh}$ — a matrix that says "here's how much the previous summary matters, and how to blend it in."
- Add them, add a bias $b_h$, squash with $\tanh$ so values stay bounded between -1 and 1.
- Result: $h_t$, the **new** summary.

That's the entire recurrence. Same $W_{xh}$, $W_{hh}$, $b_h$ are reused at every timestep — this is the "one function, applied repeatedly" idea made concrete.

If you also want an output at each step (like a prediction), add one more line:

$$\hat{y}_t = W_{hy} \, h_t + b_y$$

## Why "recurrent"

Because the function calls itself on its own output. $h_t$ depends on $h_{t-1}$, which depended on $h_{t-2}$, all the way back to the start. Unroll this across, say, 3 timesteps, and you get a chain:

$$h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow h_3$$

It looks like a deep network (one "layer" per timestep), but it isn't really deep in the usual sense — it's the **same small layer**, reused 3 times. That single fact — shared weights across an unrolled chain — is why training an RNN needs a specialized version of backprop. That's Chapter 3.

## The example we'll use for the rest of this curriculum

To keep things concrete, every chapter through Chapter 5 reuses this exact toy sequence:

> **Task:** predict rain probability from 3 days of a "wetness score."
> $x_1 = 1.0, \quad x_2 = 2.0, \quad x_3 = 3.0$

It's deliberately tiny — a single number per day, not a whole embedding vector — so you can hand-compute every number yourself in Chapter 2 and trust the mechanics before we scale up. The math doesn't change with size, only the amount of arithmetic.

## What's ahead

| Chapter | What you'll get |
|---|---|
| 2 | Full forward pass on the toy sequence — every number computed by hand |
| 3 | Backpropagation through time (BPTT), same example |
| 4 | Why gradients vanish/explode over long sequences (shown numerically) |
| 5 | The fixes — clipping, truncated BPTT, and why LSTM/GRU exist |
| 6 | The 7 architectural shapes of RNNs (one-to-many, many-to-many, etc.) and how BPTT differs across them |
| 7 | Interview cheat sheet |
| 8 | RNNs on tabular data |
| 9 | Built from scratch — NumPy, then PyTorch |

---

**One-line summary:** an RNN reads a sequence one step at a time, using the *same* small function at every step to update a running summary vector (the hidden state). Everything from here is detail on top of that idea.
