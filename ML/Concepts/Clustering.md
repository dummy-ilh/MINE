# Clustering — Methods & When to Use What

---

## The Core Question

> "Does my data have natural groups, and what's the shape/structure of those groups?"

Your answer to that determines your algorithm.

---

## The Methods Map

### 1. K-Means
**What:** Partition $n$ points into $k$ clusters by minimizing within-cluster variance.

$$\min \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

**Use when:**
- You know $k$
- Clusters are roughly **spherical, similar size**
- Data is large (scales well — $O(nkd)$)
- Speed matters

**Fails when:** Non-convex shapes, unequal cluster sizes, outliers, varying density

---

### 2. DBSCAN
**What:** Finds clusters as **dense regions** separated by low-density areas. No need to specify $k$.

**Use when:**
- You don't know $k$
- Clusters are **arbitrary shape** (rings, crescents)
- You need **outlier detection** (noise points labeled $-1$)
- Varying cluster shapes

**Fails when:** Varying density across clusters, very high dimensions

---

### 3. Hierarchical Clustering (Agglomerative)
**What:** Builds a **dendrogram** — a tree of merges from individual points up to one cluster. Cut the tree at any level to get $k$ clusters.

**Use when:**
- You don't know $k$ in advance
- You want to **explore cluster structure at multiple scales**
- Small-to-medium datasets ($n < 10K$)
- Interpretability matters (dendrogram is human-readable)

**Fails when:** Large datasets ($O(n^2)$ memory, $O(n^3)$ time)

---

### 4. Gaussian Mixture Models (GMM)
**What:** Soft probabilistic version of K-Means. Assumes data is generated from $k$ Gaussian distributions. Each point gets a **probability of belonging** to each cluster.

**Use when:**
- Clusters have **elliptical shapes** (not just spherical)
- You need **soft/probabilistic assignments**
- Clusters have **different sizes and covariances**
- You want a generative model of the data

**Fails when:** Non-Gaussian clusters, too few data points to estimate covariance

---

### 5. Spectral Clustering
**What:** Uses the **eigenvectors of the graph Laplacian** of a similarity graph. Finds clusters by graph connectivity, not Euclidean distance.

**Use when:**
- Clusters are **non-convex, manifold-structured** (rings inside rings)
- You have a **similarity/affinity matrix** already
- Medium datasets

**Fails when:** Very large $n$ (eigendecomposition is $O(n^3)$), choosing right kernel is tricky

---

### 6. Mean Shift
**What:** Each point iteratively moves toward the **mode (peak)** of the density around it. Clusters form at density peaks.

**Use when:**
- You don't know $k$
- Clusters are **blob-shaped** but you don't want to assume Gaussian
- Image segmentation, computer vision

**Fails when:** Large datasets (slow — $O(n^2)$ per iteration), high dimensions

---

## Decision Table

| Scenario | Use |
|---|---|
| Know $k$, large data, spherical clusters | **K-Means** |
| Unknown $k$, arbitrary shapes, need outlier detection | **DBSCAN** |
| Want to explore structure, small data | **Hierarchical** |
| Soft assignments, elliptical clusters | **GMM** |
| Non-convex/manifold structure | **Spectral** |
| Image segmentation, density-based blobs | **Mean Shift** |

---

## One Cheat Line Each

| Algorithm | One line |
|---|---|
| K-Means | Fast, spheres, you pick $k$ |
| DBSCAN | Shapes + outliers, density-based, auto-$k$ |
| Hierarchical | Tree of clusters, explore at any scale |
| GMM | Probabilistic K-Means with ellipses |
| Spectral | Graph-based, handles rings/manifolds |
| Mean Shift | Density peaks, no $k$, slow |

---

Great catch. Let me kill the jargon.

---

## "Spherical Clusters" — What it actually means

Imagine your data plotted on a graph.

**Spherical = clusters that look like blobs. Round. Compact.**

```
Spherical (K-Means happy):    Non-spherical (K-Means fails):

   ●●●        ●●●               ●●●●●●●●●●
  ●●●●●      ●●●●●             ●          ●
   ●●●        ●●●              ●    ●●●   ●
                                ●●●●●●●●●●
  blob A      blob B            ring shape
```

K-Means works by measuring distance to a **center point**. So it can only find clusters that are roughly equal distance from a center in all directions — i.e. round/blob shaped.

If your cluster is a **ring, crescent, spiral, or any weird shape** — K-Means will butcher it because it's trying to draw circles around centers that don't exist.

---

## Plain English translations of ALL the jargons I used

| Jargon | Plain English |
|---|---|
| Spherical clusters | Round blob-shaped groups |
| Arbitrary shape | Any shape — rings, crescents, squiggles |
| Dendrogram | A family tree of how clusters merged |
| Soft assignment | "This point is 70% cluster A, 30% cluster B" |
| Hard assignment | "This point belongs to exactly one cluster" |
| Density | How crowded a region of the plot is |
| Manifold | A curved surface hiding in high dimensions (think: the surface of a donut) |
| Graph Laplacian | A math matrix that captures which points are connected/similar |
| Covariance | How features move together (are tall people also heavy? that's covariance) |
| Generative model | A model that can also **create** new fake data points that look real |
| Mode | The peak — the most crowded point in a region |

---

# Clustering — Plain English Version

---

## What is clustering?

You have a pile of data points. You want to find **natural groups** in them. That's it.

---

## The Methods

---

### 1. K-Means
**What it does:** You tell it how many groups you want. It draws round blobs around the groups.

**Picture:**
```
  ●●●        ●●●        ●●●
 ●●●●●      ●●●●●      ●●●●●
  ●●●        ●●●        ●●●
 Group A    Group B    Group C
```

**Use it when:**
- You already know how many groups you want
- Your groups are roughly round and similar in size
- You have a lot of data and need speed

**Breaks when:**
- Groups are weird shapes (rings, crescents)
- One group is huge and another is tiny
- There are outliers (stray points far from everything)

---

### 2. DBSCAN
**What it does:** Looks for **crowded neighborhoods**. Dense = cluster. Lonely stray points = outliers. You don't tell it how many groups — it figures it out.

**Picture:**
```
●●●●●●●●●●
●          ●     ← ring shape — DBSCAN handles this
●    ●●●   ●
●●●●●●●●●●

      ×  ← stray point, labeled as outlier
```

**Use it when:**
- You don't know how many groups exist
- Groups have weird shapes
- You want to automatically detect outliers

**Breaks when:**
- Some groups are dense, others are sparse — it gets confused
- Very high number of features (crowdedness is hard to measure)

---

### 3. Hierarchical Clustering
**What it does:** Starts with every point as its own group. Then keeps merging the two closest groups, step by step, until everything is one big group. Gives you a **family tree** of merges so you can cut it wherever you want.

**Picture:**
```
        |
    ____|____
   |         |
 __|__      _|_
|     |    |   |
A     B    C   D
```
Cut high → 2 groups. Cut low → 4 groups. You choose.

**Use it when:**
- You don't know how many groups you want
- You want to **explore** — see the structure at many levels
- Your dataset is small (a few thousand points max)

**Breaks when:**
- Large datasets — too slow and uses too much memory

---

### 4. GMM (Gaussian Mixture Models)
**What it does:** Like K-Means but instead of hard "you belong to group A", it says "you're **70% group A, 30% group B**." Also handles oval/stretched groups, not just round ones.

**Picture:**
```
K-Means:           GMM:
●●●●●  ●●●●●      ●●●●●  ●●●●●
● A  ●  ● B ●     ●70% ●  ●30%●  ← this point gets split
●●●●●  ●●●●●      ●●●●●  ●●●●●
```

**Use it when:**
- You want each point to have a **probability** of belonging to each group
- Groups are oval/stretched, not just round
- Groups overlap

**Breaks when:**
- Not enough data to figure out the shape of each group
- Groups are very non-oval (banana shapes etc.)

---

### 5. Spectral Clustering
**What it does:** Builds a **web of connections** between nearby points. Then finds groups by looking at which points are well-connected to each other. Doesn't care about round vs weird shapes.

**Picture:**
```
●—●—●—●—●        ●—●—●
            vs
●—●—●—●—●        ●—●—●

Two separate webs = two groups
```

**Use it when:**
- Groups have weird shapes — rings inside rings, curved blobs
- You already have a "how similar are A and B?" table for your data

**Breaks when:**
- Very large datasets — building the web is slow
- Hard to tune

---

### 6. Mean Shift
**What it does:** Every point looks at its neighborhood and moves toward the most crowded nearby spot. Points that end up at the same crowded peak = same group.

**Picture:**
```
 . . ●● . . .         → → ●● ← ←      all arrows converge
 . .●●●● . . .        → →●●●●← ←      to the peak
 . . ●● . . .         → → ●● ← ←
```

**Use it when:**
- You don't know how many groups exist
- Groups are blob-shaped
- Used a lot in image processing

**Breaks when:**
- Large data — very slow
- Many features

---

## The Cheat Sheet

| Situation | Use |
|---|---|
| I know how many groups, they're round blobs, big data | **K-Means** |
| I don't know how many groups, weird shapes, want outliers flagged | **DBSCAN** |
| I want to explore, small data, don't know how many groups | **Hierarchical** |
| Groups overlap, I want probabilities not hard labels | **GMM** |
| Groups are rings/curves/weird | **Spectral** |
| Image grouping, don't know how many groups | **Mean Shift** |

---
