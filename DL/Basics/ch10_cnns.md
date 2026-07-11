# Chapter 10: CNNs (Convolution, Pooling, Filters)

---

### 10.1 The Plain-English Picture

A fully connected network that processes a 224×224 RGB image has to treat every pixel as an independent input. That image has 224 × 224 × 3 = 150,528 input values. A first hidden layer with 1,000 neurons needs 150,528 × 1,000 = 150 million weights — just for the first layer. This is computationally absurd, statistically wasteful, and structurally ignorant.

Why structurally ignorant? Because a fully connected layer treats pixel (0,0) and pixel (100,200) as completely independent features with no spatial relationship. But images have structure: nearby pixels are correlated, edges and textures appear locally, and the same feature (a cat's ear, a wheel, the letter "A") can appear anywhere in the image.

Convolutional Neural Networks (CNNs) exploit this structure with three key ideas:

```
THREE CNN INDUCTIVE BIASES
===========================

1. LOCAL CONNECTIVITY
   Each neuron connects to a small local region (receptive field),
   not the entire input. A 3×3 filter sees 9 pixels, not 150,528.
   Assumption: relevant features are local.

2. WEIGHT SHARING (TRANSLATION EQUIVARIANCE)
   The SAME filter is applied at EVERY position in the image.
   One set of 9 weights detects edges everywhere, not just top-left.
   Assumption: the same features can appear anywhere.
   Effect: 150M weights → 9 weights for one feature detector.

3. HIERARCHICAL COMPOSITION
   Early layers detect simple features (edges, colors).
   Later layers combine them into complex ones (textures → parts → objects).
   Assumption: complex features are compositions of simple ones.
```

These three assumptions are correct for natural images, audio spectrograms, and many spatial/sequential signals. They are the reason CNNs dominated computer vision from 2012 onward, reducing the parameter count by orders of magnitude while improving accuracy.

---

### 10.2 The Convolution Operation

Convolution is a mathematical operation that slides a small filter (kernel) across an input, computing a dot product at each position.

```
1D CONVOLUTION (to build intuition)
=====================================

Input:  x = [1, 2, 3, 4, 5]        (length 5)
Filter: f = [1, 0, -1]             (length 3, detects rising edges)

Output: y[i] = Σₖ x[i+k] · f[k]

  y[0] = x[0]·f[0] + x[1]·f[1] + x[2]·f[2]
       = 1·1 + 2·0 + 3·(-1) = 1 + 0 - 3 = -2

  y[1] = x[1]·f[0] + x[2]·f[1] + x[3]·f[2]
       = 2·1 + 3·0 + 4·(-1) = 2 + 0 - 4 = -2

  y[2] = x[2]·f[0] + x[3]·f[1] + x[4]·f[2]
       = 3·1 + 4·0 + 5·(-1) = 3 + 0 - 5 = -2

  y = [-2, -2, -2]

  The filter [1, 0, -1] is a derivative detector: positive where
  values are increasing left-to-right, negative where decreasing.
  Applied everywhere via sliding: this is translation equivariance.

2D CONVOLUTION (images)
========================

Input patch (5×5):              Filter (3×3) — vertical edge detector:
  1  1  1  0  0                   1  0 -1
  1  1  1  0  0                   1  0 -1
  1  1  1  0  0                   1  0 -1
  1  1  1  0  0
  1  1  1  0  0
  (bright on left, dark on right)

Output at position (1,1):
  [1 1 1]   [1  0 -1]
  [1 1 1] ⊙ [1  0 -1]  → element-wise multiply then sum
  [1 1 1]   [1  0 -1]

  = 1·1 + 1·0 + 1·(-1)   (row 0)
  + 1·1 + 1·0 + 1·(-1)   (row 1)
  + 1·1 + 1·0 + 1·(-1)   (row 2)
  = (1+0-1) + (1+0-1) + (1+0-1)
  = 0 + 0 + 0 = 0          ← no edge here (all same value)

Output at position (1,2) — straddling the edge:
  [1 1 0]   [1  0 -1]
  [1 1 0] ⊙ [1  0 -1]  →
  [1 1 0]   [1  0 -1]

  = (1·1 + 1·0 + 0·(-1)) × 3
  = (1 + 0 + 0) × 3 = 3     ← STRONG response at the edge ✓

The filter fires strongly where there's a transition from light to dark.
This is feature detection via learned (or designed) filters.
```

---

### 10.3 Formal Definition: 2D Convolution

```
2D CONVOLUTION — FORMAL
========================

Input feature map:  X ∈ ℝ^(Hᵢₙ × Wᵢₙ × Cᵢₙ)
  Hᵢₙ = input height
  Wᵢₙ = input width
  Cᵢₙ = input channels (3 for RGB, arbitrary for hidden layers)

Filter (kernel):    K ∈ ℝ^(k × k × Cᵢₙ × Cₒᵤₜ)
  k     = kernel size (e.g., 3×3, 5×5)
  Cᵢₙ  = input channels (same as input)
  Cₒᵤₜ = output channels (number of filters = number of features to detect)

Bias:               b ∈ ℝ^Cₒᵤₜ   (one bias per output channel)

Output feature map: Y ∈ ℝ^(Hₒᵤₜ × Wₒᵤₜ × Cₒᵤₜ)

  Y[h, w, c] = b[c] + Σᵢ Σⱼ Σₖ X[h+i, w+j, k] · K[i, j, k, c]

  Where:
    h, w = spatial position in output
    c    = output channel index
    i, j = filter spatial indices (0 to kernel_size-1)
    k    = input channel index

OUTPUT DIMENSIONS:
  With padding P and stride S:
    Hₒᵤₜ = floor((Hᵢₙ + 2P - k) / S) + 1
    Wₒᵤₜ = floor((Wᵢₙ + 2P - k) / S) + 1

  Common cases:
    k=3, P=1, S=1: Hₒᵤₜ = Hᵢₙ  (same-size output, "same" padding)
    k=3, P=0, S=1: Hₒᵤₜ = Hᵢₙ-2 (shrinks by 1 on each side)
    k=3, P=0, S=2: Hₒᵤₜ = (Hᵢₙ-2)/2 (halves spatial size, "stride conv")

PARAMETER COUNT FOR ONE CONV LAYER:
  Weights: k × k × Cᵢₙ × Cₒᵤₜ
  Biases:  Cₒᵤₜ
  Total:   k² × Cᵢₙ × Cₒᵤₜ + Cₒᵤₜ

  Example: first layer of AlexNet
    k=11, Cᵢₙ=3 (RGB), Cₒᵤₜ=96
    Weights: 11 × 11 × 3 × 96 = 34,848
    Compare to FC: 224 × 224 × 3 × 96 = 14,450,688 (414× more!)
```

---

### 10.4 What Filters Learn

In a trained CNN, filters at different layers detect qualitatively different features. This hierarchy is one of the most beautiful empirical findings in deep learning.

```
FEATURE HIERARCHY IN CNNS
===========================

Layer 1 (closest to input):
  Filters learn EDGES and COLORS.
  ┌───────────────────────────────────────────────┐
  │ //// (diagonal edges)   ││││ (vertical edges) │
  │ ──── (horizontal edges) ░░░░ (color blobs)    │
  └───────────────────────────────────────────────┘
  These are essentially Gabor filters — same filters that
  appear in the primary visual cortex (V1) of mammals.
  NOT hand-designed. Emerge purely from training on images.

Layer 2:
  Combinations of edges: CORNERS, CURVES, TEXTURES.
  ┌────────────────────────────────┐
  │  ⌐ (corners)  ∿ (curves)      │
  │  ≋ (textures) ⊞ (grids)       │
  └────────────────────────────────┘

Layer 3-4:
  PARTS: eyes, wheels, windows, fur patches, text segments.

Layer 5 (deep):
  OBJECT CONCEPTS: faces, dogs, cars, text.

Output layer:
  CLASSIFICATION SCORES: one per class.

This hierarchy is not programmed — it emerges from:
  - The convolutional structure (local + shared weights)
  - Backpropagation on labeled images
  - Depth allowing composition of simpler features
```

---

### 10.5 Padding

```
PADDING
========

Problem: without padding, each convolution shrinks the spatial dimensions.
  After many layers: feature maps become tiny.
  Information at the edges is used less often than center pixels.

Solution: add zeros around the border before convolving.

NO PADDING (P=0, "valid" convolution):
  Input: 5×5   Filter: 3×3   Output: 3×3
  ┌─────────┐           ┌───────┐
  │ · · · · ·│           │ · · · │
  │ · · · · ·│  ────►   │ · · · │
  │ · · · · ·│           │ · · · │
  │ · · · · ·│           └───────┘
  │ · · · · ·│
  └─────────┘

SAME PADDING (P=1, "same" convolution):
  Output has same H,W as input.
  For k=3: need P=1. For k=5: need P=2. General: P=(k-1)/2.
  ┌───────────┐           ┌─────────┐
  │ 0 0 0 0 0 0 0│         │ · · · · ·│
  │ 0 · · · · · 0│  ────►  │ · · · · ·│ (same size!)
  │ 0 · · · · · 0│         │ · · · · ·│
  │ 0 · · · · · 0│         │ · · · · ·│
  │ 0 · · · · · 0│         │ · · · · ·│
  │ 0 0 0 0 0 0 0│         └─────────┘
  └───────────┘
  (zeros added around border)

Most modern CNNs use "same" padding for conv layers
and reduce spatial dimensions only through pooling or stride-2 conv.
```

---

### 10.6 Stride

```
STRIDE
=======

Stride S = how many pixels the filter moves per step.
Default S=1: filter moves one pixel at a time.
S=2: filter jumps 2 pixels → output is half the spatial size.

STRIDE 1:                    STRIDE 2:
Input: 6×6, Filter: 3×3     Input: 6×6, Filter: 3×3
Output: 4×4 (P=0)           Output: 2×2 (P=0)

  ┌─┬─┬─┬─┬─┬─┐               ┌─┬─┬─┬─┬─┬─┐
  │█│█│█│ │ │ │  step 1        │█│█│█│ │ │ │  step 1 (top-left)
  ├─┼─┼─┼─┼─┼─┤               ├─┼─┼─┼─┼─┼─┤
  │█│█│█│ │ │ │                │█│█│█│ │ │ │
  ├─┼─┼─┼─┼─┼─┤               ├─┼─┼─┼─┼─┼─┤
  │█│█│█│ │ │ │                │█│█│█│ │ │ │
  ├─┼─┼─┼─┼─┼─┤               ├─┼─┼─┼─┼─┼─┤
  │ │ │ │ │ │ │                │ │ │ │ │ │ │
  └─┴─┴─┴─┴─┴─┘               └─┴─┴─┴─┴─┴─┘
                               (next step: jump 2, land at column 3)

S=2 conv replaces pooling in many modern architectures.
  Advantages over pooling:
    - Learnable downsampling (filter weights are trained)
    - Can learn non-uniform downsampling
  Disadvantages:
    - More parameters than max pooling

Modern trend: use stride-2 conv instead of pooling for downsampling.
Pooling still common for global (spatial average pooling before classifier).
```

---

### 10.7 Pooling

Pooling reduces spatial dimensions while preserving the most important information.

```
MAX POOLING
============

Pool size 2×2, stride 2:
  Takes the MAXIMUM value in each 2×2 region.
  Halves both height and width.

Input (4×4):                  Output (2×2):
  1   3   2   4                 max(1,3,5,6) = 6    max(2,4,8,9) = 9
  5   6   8   9      ──►        max(3,7,2,4) = 7    max(1,8,3,5) = 8
  3   7   2   1
  4   2   3   5                   6    9
                                  7    8

WHY MAX POOLING WORKS:
  "Is this feature present somewhere in this region?"
  Max pooling answers YES (high value) or NO (low value).
  It's an OR operation: feature detected anywhere → keep the signal.
  It doesn't care exactly WHERE in the 2×2 region the feature appears.
  This provides TRANSLATION INVARIANCE within the pooling window.

AVERAGE POOLING
================
  Takes the AVERAGE of each 2×2 region instead of maximum.

  Input:         Output:
  1   3          (1+3+5+6)/4 = 3.75   (2+4+8+9)/4 = 5.75
  5   6    ──►
  
  Less common in hidden layers (max better at preserving strong signals).
  COMMON for GLOBAL AVERAGE POOLING: average over entire feature map.
    Takes a [7×7×512] feature map → [1×1×512] → [512] vector.
    Replaces the flattening + FC approach in modern networks.
    Much fewer parameters, better spatial invariance.

GLOBAL AVERAGE POOLING (GAP)
==============================

  Before GAP:                After GAP:
  Feature map [7×7×512]      Vector [512]
  
  GAP[c] = (1/49) Σₕ Σw X[h, w, c]   (average over spatial dims)
  
  This single vector is then fed to the output classifier (softmax).
  
  ResNet uses GAP instead of fully-connected layers at the end.
  This reduces parameters massively and improves generalization.
```

---

### 10.8 Receptive Field

The receptive field is the region of the original input that influences a neuron's output. Understanding it is critical for designing CNN architectures.

```
RECEPTIVE FIELD GROWTH
=======================

Each 3×3 conv with stride 1 increases receptive field by 2 in each dim.

Layer 0 (input): RF = 1×1  (just the pixel itself)

Layer 1: 3×3 conv → RF = 3×3
  ┌─────┐
  │● ● ●│
  │● ● ●│ ← this 3×3 region of input affects one output pixel
  │● ● ●│
  └─────┘

Layer 2: 3×3 conv → RF = 5×5
  ┌─────────┐
  │● ● ● ● ●│
  │● ● ● ● ●│ ← a 5×5 region of input affects one layer-2 pixel
  │● ● ● ● ●│
  │● ● ● ● ●│
  │● ● ● ● ●│
  └─────────┘

After L layers of 3×3 conv with S=1:
  RF = (2L+1) × (2L+1)
  
  5 layers: RF = 11×11
  15 layers: RF = 31×31
  50 layers: RF = 101×101

EFFECTIVE RECEPTIVE FIELD vs. THEORETICAL:
  Theoretical RF = pixels that COULD influence the output.
  Effective RF   = pixels that SIGNIFICANTLY influence the output.
  
  Due to the multiplicative nature of gradient weighting, the
  effective RF has a Gaussian shape — central pixels contribute
  exponentially more than peripheral ones.
  Effective RF ≈ 0.5 × √L × kernel_size  (much smaller than theoretical)
  
  This is why attention mechanisms (transformers) can be superior:
  every token attends to every other token — global RF from layer 1.

DILATED CONVOLUTION — EXPANDING RF WITHOUT DEPTH
=================================================
Insert gaps (dilation) in the filter:

Dilation rate d=1 (standard):    Dilation rate d=2:
  ● ● ●                            ●   ●   ●
  ● ● ●                            
  ● ● ●                            ●   ●   ●
  RF = 3×3                         
                                   ●   ●   ●
                                   RF = 5×5, but still 3×3=9 weights!

Dilation rate d=4:
  Samples every 4th pixel → RF = 9×9 with 9 weights.
  
Used in: semantic segmentation (DeepLab), WaveNet (audio),
         temporal convolution networks. Allows large RF with few params.
```

---

### 10.9 Depthwise Separable Convolution

A critical efficiency technique used in MobileNet, Xception, and EfficientNet.

```
STANDARD CONVOLUTION vs. DEPTHWISE SEPARABLE
=============================================

Standard convolution: k × k × Cᵢₙ × Cₒᵤₜ parameters
  Simultaneously: (1) mix spatial information, (2) mix channel information.

Depthwise separable: separates these two operations.

STEP 1 — DEPTHWISE CONVOLUTION:
  Apply one separate k×k filter per input channel.
  No mixing across channels yet.
  Parameters: k × k × Cᵢₙ      (one filter per channel)
  Output: same spatial size, Cᵢₙ channels

  Cᵢₙ=3 example:
    Channel 0: [k×k filter] → output channel 0
    Channel 1: [k×k filter] → output channel 1
    Channel 2: [k×k filter] → output channel 2

STEP 2 — POINTWISE CONVOLUTION (1×1 conv):
  Apply 1×1 × Cᵢₙ filters × Cₒᵤₜ to mix channels.
  No spatial mixing — just linear combination across channels.
  Parameters: 1 × 1 × Cᵢₙ × Cₒᵤₜ = Cᵢₙ × Cₒᵤₜ

PARAMETER COUNT COMPARISON:
  Standard conv: k² × Cᵢₙ × Cₒᵤₜ
  Depthwise sep: k² × Cᵢₙ + Cᵢₙ × Cₒᵤₜ

  Ratio = (k² + Cₒᵤₜ) / (k² × Cₒᵤₜ)
        = 1/Cₒᵤₜ + 1/k²
        ≈ 1/k²  for large Cₒᵤₜ

  For k=3, Cₒᵤₜ=256:
    Standard: 9 × 256 = 2304 (per input channel)
    DW Sep:   9 + 256  =  265 (per input channel)
    Reduction: ~8.7× fewer parameters and FLOPs!

MobileNet uses depthwise separable convs throughout.
95%+ reduction in multiply-adds vs. VGG.
Runs in real-time on mobile devices.
```

---

### 10.10 The Full CNN Forward Pass

```
TYPICAL CNN ARCHITECTURE PATTERN
==================================

Input image: [224 × 224 × 3]
     │
     ▼
┌──────────────────────────────────────────┐
│ Conv Block 1:                            │
│   Conv(k=3, C=64, P=1, S=1)             │
│   BatchNorm                              │
│   ReLU                                  │
│   Output: [224 × 224 × 64]              │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│ Downsample (MaxPool 2×2 or stride-2 conv)│
│   Output: [112 × 112 × 64]              │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│ Conv Block 2:                            │
│   Conv(k=3, C=128, P=1, S=1)            │
│   BatchNorm                              │
│   ReLU                                  │
│   Output: [112 × 112 × 128]             │
└──────────────────────────────────────────┘
     │
     ▼
  [More blocks with increasing channels, decreasing spatial dims]
     │
     ▼
┌──────────────────────────────────────────┐
│ Final conv block:                        │
│   Feature map: [7 × 7 × 512]            │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│ Global Average Pooling                   │
│   [7 × 7 × 512] → [512]                │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│ Fully Connected + Softmax               │
│   [512] → [1000]  (ImageNet classes)    │
└──────────────────────────────────────────┘

PATTERN: channels increase (64→128→256→512) as spatial dims decrease.
This keeps computation roughly constant at each stage:
  Stage 1: 224×224×64   ≈ 3.2M activations
  Stage 2: 112×112×128  ≈ 1.6M activations (same as stage 1!)
  Stage 3:  56×56×256   ≈ 0.8M activations
  Stage 4:  28×28×512   ≈ 0.4M activations

The spatial information is gradually compressed into channel depth.
```

---

### 10.11 Worked Numerical Example: One Conv Layer

```
COMPLETE CONV LAYER COMPUTATION
=================================

Setup:
  Input:    X ∈ ℝ^(4×4×1)  (4×4 image, 1 channel)
  Filter:   K ∈ ℝ^(3×3×1)  (one 3×3 filter, 1 input channel)
  Bias:     b = 0.1
  Stride:   S = 1
  Padding:  P = 0

  Output size: Hₒᵤₜ = (4-3)/1 + 1 = 2,  Wₒᵤₜ = 2
  Output: Y ∈ ℝ^(2×2×1)

Input X:           Filter K:
  1  2  1  0        1  0 -1
  3  4  2  1        2  0 -2
  2  1  3  2        1  0 -1
  0  1  2  1        (Sobel-like edge detector)

Bias b = 0.1

COMPUTING OUTPUT AT POSITION (0,0):
  Patch = X[0:3, 0:3] =
    1  2  1
    3  4  2
    2  1  3

  Y[0,0] = Σ (Patch ⊙ K) + b
          = (1×1 + 2×0 + 1×(-1))     row 0
          + (3×2 + 4×0 + 2×(-2))     row 1
          + (2×1 + 1×0 + 3×(-1))     row 2
          + bias

          = (1 + 0 - 1)
          + (6 + 0 - 4)
          + (2 + 0 - 3)
          + 0.1

          = 0 + 2 + (-1) + 0.1
          = 1.1

COMPUTING OUTPUT AT POSITION (0,1):
  Patch = X[0:3, 1:4] =
    2  1  0
    4  2  1
    1  3  2

  Y[0,1] = (2×1 + 1×0 + 0×(-1))
          + (4×2 + 2×0 + 1×(-2))
          + (1×1 + 3×0 + 2×(-1))
          + 0.1

          = (2 + 0 + 0)
          + (8 + 0 - 2)
          + (1 + 0 - 2)
          + 0.1

          = 2 + 6 + (-1) + 0.1
          = 7.1

COMPUTING OUTPUT AT POSITION (1,0):
  Patch = X[1:4, 0:3] =
    3  4  2
    2  1  3
    0  1  2

  Y[1,0] = (3×1 + 4×0 + 2×(-1))
          + (2×2 + 1×0 + 3×(-2))
          + (0×1 + 1×0 + 2×(-1))
          + 0.1

          = (3 + 0 - 2)
          + (4 + 0 - 6)
          + (0 + 0 - 2)
          + 0.1

          = 1 + (-2) + (-2) + 0.1
          = -2.9

COMPUTING OUTPUT AT POSITION (1,1):
  Patch = X[1:4, 1:4] =
    4  2  1
    1  3  2
    1  2  1

  Y[1,1] = (4×1 + 2×0 + 1×(-1))
          + (1×2 + 3×0 + 2×(-2))
          + (1×1 + 2×0 + 1×(-1))
          + 0.1

          = (4 + 0 - 1)
          + (2 + 0 - 4)
          + (1 + 0 - 1)
          + 0.1

          = 3 + (-2) + 0 + 0.1
          = 1.1

OUTPUT Y (before activation):
  Y = [[ 1.1,  7.1],
       [-2.9,  1.1]]

After ReLU:
  Y_relu = [[1.1, 7.1],
            [0.0, 1.1]]

INTERPRETATION:
  Y[0,1] = 7.1 (strong response) → strong vertical edge at position (0,1)
  Y[1,0] = -2.9 (negative) → reversed edge (dark-to-light transition)
            → zeroed by ReLU since this filter detects left-dark-right-light

PARAMETER COUNT:
  Weights: 3×3×1 = 9
  Biases:  1
  Total:   10

  Compare to FC layer: 16 inputs × 4 outputs = 64 weights + 4 biases = 68
  This one conv layer uses 10 parameters but covers 2×2=4 output positions.
  FC would use 68 parameters for the same mapping.
  At larger scales the difference is orders of magnitude.
```

---

### 10.12 Why This Matters — What Breaks If You Get This Wrong

1. **Forgetting padding in deep networks.** Without padding (P=0), each 3×3 conv shrinks H and W by 2. In a 10-layer network: 224 - 20 = 204 pixels remaining. In a 50-layer network: 224 - 100 = 124, and you've lost 45% of your spatial information before any intentional downsampling. Use same-padding (P=1 for k=3) in all conv layers that aren't meant to downsample.

2. **Wrong output size calculation.** The formula `(H + 2P - k) / S + 1` must be an integer. If it isn't, you have an invalid configuration — PyTorch will throw a shape error or silently truncate. Always verify output sizes when building a new architecture, especially with stride-2 or asymmetric padding.

3. **Not using weight sharing mentally.** A 3×3 conv layer with 64 output channels has 3×3×Cᵢₙ×64 parameters — NOT 224×224×Cᵢₙ×64. Confusing this when estimating model size leads to wildly wrong parameter counts and memory estimates. This matters for model deployment on mobile/edge devices.

4. **Applying fully connected layers to spatial feature maps without GAP.** Flattening a [7×7×512] feature map and feeding it to a FC layer gives 7×7×512 = 25,088 inputs → massive FC layer, no spatial invariance. If an object moves slightly in the image, the flattened vector changes completely. GAP provides spatial invariance and far fewer parameters.

5. **Confusing convolution with cross-correlation.** Deep learning frameworks implement cross-correlation (filter not flipped) and call it "convolution." True convolution flips the filter before sliding. For learned filters this doesn't matter (the filter just learns the flipped version). But if you're using a hand-designed filter (e.g., Gaussian blur, Sobel), you need to know which operation your framework uses.

---

### 10.13 Google/Apple-Level Interview Q&A

---

**Q1: "A CNN takes a 224×224×3 image. The first layer is Conv(k=7, C=64, S=2, P=3), followed by MaxPool(k=3, S=2, P=1). What is the output shape? How many parameters are in the conv layer? What is the receptive field of one neuron in the pooling output?"**

*Why this is asked:* Dimension tracking through conv and pooling layers is an essential daily skill for CNN engineers. Getting it wrong causes shape errors or, worse, silent architectural bugs where the network trains but the receptive field is wrong for the task. This is a pure "can you work through the math" question that filters candidates who learned CNNs by reading about them vs those who have actually built them.

**Answer:**

```
CONV LAYER: Conv(k=7, C=64, S=2, P=3)
=========================================
Input:  [224 × 224 × 3]

Output height:
  Hₒᵤₜ = floor((Hᵢₙ + 2P - k) / S) + 1
        = floor((224 + 2×3 - 7) / 2) + 1
        = floor((224 + 6 - 7) / 2) + 1
        = floor(223 / 2) + 1
        = 111 + 1
        = 112

Output width (same calculation): Wₒᵤₜ = 112

Conv output shape: [112 × 112 × 64]

Parameter count:
  Weights: k × k × Cᵢₙ × Cₒᵤₜ = 7 × 7 × 3 × 64 = 9,408
  Biases:  Cₒᵤₜ = 64
  Total:   9,472

Note: this IS the first layer of ResNet (exactly).

MAX POOL: MaxPool(k=3, S=2, P=1)
=========================================
Input:  [112 × 112 × 64]

Output height:
  Hₒᵤₜ = floor((112 + 2×1 - 3) / 2) + 1
        = floor((112 + 2 - 3) / 2) + 1
        = floor(111 / 2) + 1
        = 55 + 1
        = 56

Pool output shape: [56 × 56 × 64]

RECEPTIVE FIELD CALCULATION
=========================================

For a neuron in the pooling output, what region of the original
224×224 input does it see?

Method: work backwards through each layer.

Pooling output neuron sees: 3×3 region of conv output
  (pool size k=3 with S=2, but RF calculation needs the actual extent)
  RF from pool layer = 3×3 (of conv output)

Mapping pool RF back through conv layer (k=7, S=2, P=3):
  A region of size r in the conv output corresponds to a region of
  size (r-1)×S + k in the input.

  r = 3 (pool kernel size)
  S = 2 (conv stride)
  k = 7 (conv kernel size)

  RF in input = (3-1) × 2 + 7 = 4 + 7 = 11

RF of one pooling output neuron = 11×11 in the original image.

Verification:
  - Conv filter size 7×7 covers 7×7 of input at each position
  - But conv has stride 2, so the 3×3 pool region spans:
    columns {0, 2, 4} × stride 2 = {0, 2, 4} of the conv output,
    corresponding to conv input positions {0, 2, 4} × 2 = {0, 4, 8}
    each covering [position, position+6], so total: [0, 14] → 15?
  
  Careful: for stride S, the RF formula is:
    RF_total = RF_lower × S_upper + (k_upper - 1)
    
  Iterating properly:
    RF after pool: RF_pool = 1 (single output) → maps to pool size 3
    RF in conv output: 1 → pool with k=3,S=2 → (1-1)×2+3 = 3
    RF in input: 3 → conv with k=7,S=2 → (3-1)×2+7 = 11

  Final answer: 11×11 receptive field in the original 224×224 image.
  
This is quite small! After just two layers, each output neuron
only "sees" 11×11 / 224×224 = 0.24% of the input image.
This motivates depth: many layers needed to build up global context.
```

---

**Q2: "Explain why a 1×1 convolution is useful. Give three distinct use cases where it appears in modern CNN architectures and explain what computation it performs."**

*Why this is asked:* The 1×1 conv is one of the most powerful and misunderstood operations in CNNs. It appears in Inception, ResNet, MobileNet, and virtually every modern architecture. Candidates who understand it deeply can design efficient architectures; those who don't design bloated ones. This question is a proxy for "do you actually understand what convolutions do?"

**Answer:**

A 1×1 convolution with Cᵢₙ input channels and Cₒᵤₜ output channels performs a linear combination of input channels at each spatial position independently — it is a fully connected layer applied to the channel dimension, replicated across every spatial position.

```
1×1 CONV MATHEMATICS:
  For each position (h,w):
    Y[h, w, c] = Σₖ X[h, w, k] · W[k, c] + b[c]

  This is identical to a FC layer applied to the Cᵢₙ-dimensional
  vector at each (h,w) position, with shared weights across positions.

  Output shape: [H × W × Cₒᵤₜ]  (same H,W, different C)
  Parameters:   1 × 1 × Cᵢₙ × Cₒᵤₜ = Cᵢₙ × Cₒᵤₜ
```

**Use case 1: Channel dimensionality reduction (bottleneck)**

```
In ResNet's bottleneck block:
  Input: [H × W × 256]
  1×1 conv: [H × W × 256] → [H × W × 64]    ← REDUCE channels
  3×3 conv: [H × W × 64]  → [H × W × 64]    ← spatial processing
  1×1 conv: [H × W × 64]  → [H × W × 256]   ← RESTORE channels

The middle 3×3 conv operates on 64 channels instead of 256.
Parameter savings: 3×3×64×64 vs 3×3×256×256 = 36,864 vs 589,824
                 = 16× fewer parameters in the 3×3 layer!

The 1×1 conv compresses channel information before expensive spatial ops.
This is the "bottleneck" design that allows ResNet to be deep.
```

**Use case 2: Increasing expressiveness without spatial cost (Inception)**

```
In Inception modules, different conv sizes run in parallel:
  1×1 conv   → small-scale features
  3×3 conv   → medium-scale features
  5×5 conv   → large-scale features
  Concat all → rich multi-scale features

The 5×5 conv would be too expensive without 1×1 reduction first:
  Input: [28×28×192]
  5×5 conv directly: 5×5×192×32 = 153,600 params
  
  With 1×1 first: 192→16 then 5×5: 1×1×192×16 + 5×5×16×32
                = 3,072 + 12,800 = 15,872 params (10× fewer!)
  
  The 1×1 conv doesn't change spatial structure,
  only reweights channels → pure information bottleneck.
```

**Use case 3: Projection shortcut in residual connections**

```
In ResNet, when a residual block changes channel count:
  Input a^(l):   [H × W × 64]
  Block output:  [H/2 × W/2 × 128]  (stride-2 downsampled, more channels)

  We need to add the skip connection:
  a^(l+1) = F(a^(l)) + a^(l)
  
  But F(a^(l)) ∈ ℝ^(H/2 × W/2 × 128) and a^(l) ∈ ℝ^(H × W × 64).
  Shapes don't match! Can't add them.

  Solution: 1×1 conv with stride 2 as the projection shortcut:
    a^(l)_projected = Conv1×1(a^(l), Cₒᵤₜ=128, S=2)
    shape: [H/2 × W/2 × 128]  ← now matches!
  
  a^(l+1) = F(a^(l)) + a^(l)_projected  ✓

  The 1×1 projection learns the best linear mapping between
  the two channel spaces, adding only Cᵢₙ × Cₒᵤₜ parameters.
```

---

**Q3: "What is translation invariance vs translation equivariance? Which property do CNNs have, which do they not, and why does this distinction matter for tasks like object detection vs image classification?"**

*Why this is asked:* This is a precision question about what CNNs actually do vs what people think they do. Many engineers say "CNNs are translation invariant" — this is wrong. They are equivariant to translation, with invariance only achieved through pooling. The distinction is critical for understanding why object detection (which needs location) uses different architectures than image classification (which doesn't). Companies like Apple (object detection in iOS) and Google (image search) build these systems, and engineers need to understand the properties precisely.

**Answer:**

```
DEFINITIONS
============

Translation equivariance:
  If the input shifts, the output shifts by the same amount.
  Formally: f(T_δ(x)) = T_δ(f(x))
  Where T_δ = translation by δ pixels.

  "If the cat moves right by 10 pixels, the feature map also
   shifts right by 10 pixels."

Translation invariance:
  If the input shifts, the output DOESN'T change.
  Formally: f(T_δ(x)) = f(x)

  "If the cat moves right by 10 pixels, the classification
   output is still 'cat' with the same confidence."
```

**What CNNs have:**

```
CONVOLUTION IS TRANSLATION EQUIVARIANT:
  A conv layer with stride S=1 shifts its output by δ/S when
  the input shifts by δ (discrete version with boundary effects).

  Proof:
    [f * k](x) = Σᵢ f(x-i)·k(i)
    f shifted by δ: f(x-δ)
    [f(·-δ) * k](x) = Σᵢ f(x-δ-i)·k(i) = [f*k](x-δ)
    The output shifts by the same δ. ∎

  This means: if an edge detector finds an edge at position (h,w),
  and the image shifts right by 5 pixels, the detector now finds
  the edge at position (h, w+5). The detection is preserved —
  just relocated. This is equivariance.

MAX POOLING provides LOCAL invariance:
  Max pooling over a 2×2 region: a shift of 1 pixel in the input
  doesn't change the max pooling output (both shifted and original
  inputs may have the same maximum).
  This gives invariance within the pooling window (typically 2×2).

GLOBAL AVERAGE POOLING provides GLOBAL invariance:
  After GAP, a feature map [H×W×C] → [C] vector.
  A shift in the input shifts the feature map, but the AVERAGE
  over all positions stays the same (periodic boundary effects aside).
  → GAP + Classifier is truly translation invariant.
```

**Why the distinction matters for tasks:**

```
IMAGE CLASSIFICATION (want invariance):
  "Is there a cat in this image?" → same answer regardless of cat position.
  
  CNN + GAP + Softmax: CORRECT choice.
  The spatial information is thrown away by GAP.
  We don't need to know WHERE the cat is, just THAT it's there.

OBJECT DETECTION (want equivariance, NOT invariance):
  "Where is the cat and what is its bounding box?"
  
  We NEED to know the spatial location of the cat.
  If we use GAP, we throw away the location information!
  
  Solutions:
  (a) KEEP the spatial feature map (don't do GAP).
      Apply a classifier at every spatial position.
      → Fully Convolutional Networks (FCN), YOLO, SSD.
  
  (b) REGION PROPOSALS: first propose candidate regions,
      then classify each region separately.
      → Faster R-CNN: RPN proposes regions, ROI Pool extracts features.
  
  Both rely on CNN's equivariance: the feature at position (h,w)
  is about the image content near pixel (h×stride, w×stride).
  If an object is at position (h,w), the feature at (h,w) encodes it.

SEMANTIC SEGMENTATION (need equivariance + dense output):
  "What class is every single pixel?"
  
  Need full spatial resolution in output.
  Cannot use striding/pooling (loses resolution).
  Solutions: dilated convolutions (no spatial reduction),
             encoder-decoder (reduce then upsample, U-Net).
  The equivariance of conv ensures spatial correspondence
  between input pixels and output class predictions.

THE KEY INSIGHT:
  Equivariance: "when things shift, so does my representation" → preserves spatial info
  Invariance:   "when things shift, my output doesn't change" → destroys spatial info

  For classification: need invariance → use GAP or strided pooling.
  For localization:   need equivariance → keep spatial maps, no GAP.
  
  Architectures that need both (Mask R-CNN, panoptic segmentation):
  Split into two branches at some layer:
    Classification branch: → GAP → invariant class prediction
    Localization branch:   → preserve spatial → bounding box / mask
```

---

*End of Chapter 10. Chapter 11 (CNN Architectures) coming next.*
