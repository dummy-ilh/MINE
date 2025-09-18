
What is a Recurrent Neural Network (RNN)?

A Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data or time series data, like text, audio, or video. Unlike a traditional Feedforward Neural Network where information flows in a single direction, an RNN has a "loop" that allows information to persist and be used in future calculations. This gives it a form of "memory." 🧠
Think of it like a human reading a sentence: you understand the meaning of the current word by also remembering the words that came before it. An RNN does the same, using the output from a previous step as an additional input for the current step.
Image of recurrent neural network architecture diagram

To better understand this "loop," we often unroll the RNN over time. This makes it look like a very deep network where each layer corresponds to a time step.
Image of unrolled recurrent neural network diagram


The Math Behind an RNN

The core of an RNN is its ability to update a hidden state at each time step, which acts as the network's memory.
At each time step t, the RNN takes two inputs:
The current input, xt​.
The hidden state from the previous time step, ht−1​.
The new hidden state, ht​, is calculated using a set of learned weights:
ht​=tanh(Whh​ht−1​+Wxh​xt​+bh​)
Where:
Whh​ is the weight matrix for the recurrent hidden state.
Wxh​ is the weight matrix for the input.
bh​ is the bias vector.
The tanh function (or sometimes ReLU) is the activation function.
The output, yt​, at time step t is then calculated based on the new hidden state:
yt​=Why​ht​+by​
This output can be a prediction, a word, or whatever the task requires.

Training an RNN

Training an RNN is similar to training a regular neural network, but it uses a specialized version of backpropagation called Backpropagation Through Time (BPTT). The error at the final time step is propagated backward through the unrolled network, allowing the model to update the weights based on the loss from all time steps.

Challenges with RNNs

Vanishing Gradient Problem: During BPTT, the gradients (used to update weights) can become extremely small. This makes it difficult for the network to learn long-term dependencies, as the updates from early time steps are effectively "lost."
Exploding Gradient Problem: Conversely, the gradients can become extremely large, leading to unstable learning and the network's weights "exploding" to a value of NaN (Not a Number). This is less common but can be managed by gradient clipping, which caps the maximum value of the gradients.

Types of RNNs and Their Evolution

The challenges of the vanishing/exploding gradient problem led to the development of more advanced architectures.

1. Long Short-Term Memory (LSTM)

An LSTM is a specialized RNN that was explicitly designed to handle long-term dependencies. It achieves this by using a "cell state" that runs through the entire chain and a series of "gates" that regulate the flow of information. The gates are special neural networks themselves that use a sigmoid function to output values between 0 and 1, acting as a filter.
Forget Gate: Decides what information to throw away from the cell state.
Input Gate: Decides what new information to store in the cell state.
Output Gate: Decides what part of the cell state to output to the hidden state.
![]("Image of http://googleusercontent.com/image_collection/image_retrieval/735203521162558814")

2. Gated Recurrent Unit (GRU)

A GRU is a simplified version of an LSTM. It combines the forget and input gates into a single update gate and also merges the cell state and hidden state. GRUs are simpler to compute and train but offer comparable performance to LSTMs on many tasks.
Image of gated recurrent unit (GRU) diagram


Applications of RNNs

RNNs, and more specifically LSTMs and GRUs, are used for a wide range of tasks involving sequential data:
Natural Language Processing (NLP): Machine translation, text generation (e.g., creating summaries), and sentiment analysis.
Speech Recognition: Converting audio signals into text.
Time Series Prediction: Forecasting stock prices or weather patterns.
Video Analysis: Understanding actions in video frames.



