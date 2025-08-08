Here’s your **Ultimate Clustering Algorithm Summary Table**—designed for **FAANG-level interviews**, high-concept clarity, and real-world applicability.
We cover all the core clustering methods:

| Algorithm                        | When to Use                                                                                                     | Assumptions                                                                                         | Nuances / Issues                                                                                                        | Solutions / Tweaks                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **K-Means**                      | - Clusters are spherical and equally sized<br>- You know or can guess `k`<br>- Fast clustering needed           | - Clusters are convex, isotropic (spherical)<br>- Equal variance<br>- Euclidean distance meaningful | - Sensitive to initialization<br>- Poor with non-spherical shapes<br>- Hard clustering only<br>- Needs `k`              | - Use K-Means++<br>- Try Elbow or Silhouette for `k`<br>- Use multiple runs<br>- Use GMM for soft assignments                            |
| **K-Medoids (PAM)**              | - Small datasets<br>- Robust to noise/outliers<br>- Want interpretability (medoids are real data)               | - Similar to K-Means but no assumption of centroid validity                                         | - Slower than K-Means<br>- Still needs `k`<br>- Still hard clustering                                                   | - Use CLARA for large datasets<br>- Replace L2 with L1 distance if needed                                                                |
| **Gaussian Mixture Model (GMM)** | - You need **soft clustering**<br>- Data is Gaussian-ish<br>- Overlapping clusters                              | - Data comes from a mixture of Gaussians<br>- Cluster shapes: ellipsoids                            | - Requires `k`<br>- Sensitive to initialization<br>- Prone to overfitting                                               | - Use BIC/AIC to pick `k`<br>- Use regularization<br>- Multiple runs                                                                     |
| **DBSCAN**                       | - Unknown number of clusters<br>- Clusters of varying shape/density<br>- Need outlier detection                 | - Clusters are dense areas separated by sparse regions                                              | - Sensitive to `eps` and `minPts`<br>- Struggles with varying densities<br>- High-dimensional data degrades performance | - Use k-distance plots to choose `eps`<br>- Use HDBSCAN for varying densities<br>- Standardize data                                      |
| **HDBSCAN**                      | - Like DBSCAN but with varying density<br>- Need hierarchy or probability of cluster membership                 | - Same as DBSCAN, but no fixed density                                                              | - May produce overlapping hierarchies<br>- No `k` → hard to validate                                                    | - Use cluster stability for selection<br>- Tune min\_cluster\_size carefully                                                             |
| **Mean Shift**                   | - Don’t want to guess `k`<br>- Want to detect **modes** of a density<br>- Works well with image or spatial data | - Points are drawn from a continuous distribution<br>- Clusters = modes in KDE                      | - Slow (O(n²))<br>- Very sensitive to bandwidth<br>- May detect too many clusters                                       | - Use bandwidth heuristics (Scott’s, Silverman’s)<br>- Try quantile-based bandwidth tuning                                               |
| **Agglomerative Hierarchical**   | - Need dendrogram or hierarchy<br>- Don't know `k` in advance<br>- Small-to-medium data                         | - Similarity metric is meaningful<br>- Merge small clusters into large                              | - Dendrogram can be hard to interpret<br>- Sensitive to noise<br>- No correction once merged                            | - Try different linkage methods (Ward, complete, average)<br>- Cut dendrogram at optimal point using inconsistency metrics or Silhouette |
| **Spectral Clustering**          | - Graph-based data<br>- Non-convex or nested clusters<br>- Clusters lie on a manifold                           | - Laplacian matrix captures meaningful structure<br>- Eigenvectors represent low-dim embedding      | - Expensive eigendecomposition<br>- Still needs `k`<br>- Doesn’t scale to huge datasets                                 | - Use sparse graph<br>- Use approximate eigensolvers<br>- Preprocess with PCA                                                            |
| **OPTICS**                       | - You like DBSCAN but want ordering of density<br>- Clusters of different density<br>- Want reachability graph  | - Similar to DBSCAN assumptions<br>- Works by ordering points by density reachability               | - Hard to interpret reachability plot<br>- Not as plug-and-play                                                         | - Use automated clustering from reachability plot<br>- Works well with visualization                                                     |
| **Birch**                        | - Large datasets<br>- Online/incremental clustering<br>- Memory efficient                                       | - Assumes hierarchical CF trees approximate data well                                               | - Doesn’t perform well on non-spherical clusters                                                                        | - Combine with other algorithms (e.g. K-Means on leaf nodes)                                                                             |

---

## 🧠 Bonus: Intuition for Spherical vs Elliptical vs Arbitrary Clusters

| Shape                     | Meaning                                               | Algorithm Fit                            |
| ------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| **Spherical**             | Same spread in all directions, Euclidean norm applies | K-Means, GMM (equal covariance)          |
| **Elliptical**            | Elongated, different variance in directions           | GMM (full covariance), Spectral          |
| **Arbitrary / Irregular** | Non-convex, nested, curved                            | DBSCAN, HDBSCAN, Spectral, Agglomerative |

---

## 🧪 FAANG Interview Tips:

* **K-Means fails** when cluster densities or variances differ. Say: "Because it assumes spherical, equally sized clusters."
* **DBSCAN fails** in high dimensions. Say: "Because density estimation becomes meaningless—curse of dimensionality."
* **Spectral is good** for manifolds. Say: "It reduces the problem to finding connected components in the eigenspace."
* **GMM’s strength** is soft clustering. Say: "It gives the probability of each point belonging to each cluster."
* **Hierarchical's power** is interpretability. Say: "I can trace how clusters merged or split, and cut at optimal levels."

---

Would you like this table exported to **Markdown**, **PDF**, or LaTeX for your master notes?

Or should we now go deeper into **Spectral Clustering or HDBSCAN**?
