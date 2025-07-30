Community detection is a process in **network science and graph theory** used to **identify groups of nodes (communities or clusters)** in a network that are more **densely connected internally** than with the rest of the network.

---

### **1️⃣ What is a Community?**

* A **community** in a network is a set of nodes that:

  * Have **many connections** (edges) among themselves
  * Have **fewer connections** to nodes outside their group

Example:

* In a **social network**, a community could represent:

  * A friend circle
  * A workgroup
  * A family cluster

---

### **2️⃣ What Does Community Detection Do?**

Community detection algorithms **analyze a graph** and **group related nodes** based on their connection patterns.

It helps to:

* **Find structure** in large, complex networks
* **Simplify the network** by identifying clusters
* **Reveal hidden relationships** between nodes

---

### **3️⃣ Why is it Useful?**

Community detection is widely used in:

* **Social media analysis:** Detect friend groups or influencer clusters
* **Fraud detection:** Identify suspiciously tight-knit groups
* **Biology:** Find functional modules in protein interaction networks
* **Recommender systems:** Suggest items based on user clusters

---

### **4️⃣ Example**

Imagine a network of emails exchanged in a company:

* Nodes = employees
* Edges = email exchanges
* Community detection can automatically group:

  * HR team
  * Sales team
  * Tech team

These clusters appear **naturally** without prior labeling.

---

Would you like me to make a **small visual diagram of community detection** (before vs. after clustering) for easier understanding?
If you have **client-advisor communication data** (e.g., emails, messages, call logs), **community detection** can help you **uncover patterns and relationships** within those communications.

---

## **1️⃣ Possible Applications of Community Detection**

### **(a) Identify Relationship Clusters**

* **Goal:** See which clients and advisors form **tight communication groups**.
* **Use case:** Helps you know which advisors frequently work together or which clients rely on a particular group.
* **Example:** Advisor A, B, and Client X form a small cluster because they exchange emails often.

---

### **(b) Detect Information Flow Patterns**

* **Goal:** Understand **how information travels** between clients and advisors.
* **Use case:** Detect isolated clients (potential neglect) or over-connected advisors (possible bottlenecks).
* **Example:** Some clients are only connected via one advisor; others interact with multiple advisors.

---

### **(c) Spot Hidden Communities or Unusual Links**

* **Goal:** Find **unexpected communication clusters**.
* **Use case:** Useful in **compliance and risk**—detects unapproved communication lines or **cross-team interactions** not following policy.

---

### **(d) Segment Clients by Advisor Networks**

* **Goal:** Use detected communities to **group clients** with similar advisory networks.
* **Use case:** Helps with **targeted marketing, cross-selling**, or identifying underserved groups.

---

### **(e) Support Workload Balancing**

* **Goal:** Analyze advisor clusters to **identify overloaded advisors** or **clients lacking attention**.
* **Use case:** Improve client experience by redistributing communication.

---

## **2️⃣ How You’d Do It (High Level)**

1. **Build a Communication Graph**

   * **Nodes:** Clients and Advisors
   * **Edges:** Communication frequency or strength (emails, calls, chats)

2. **Apply Community Detection Algorithms**

   * **Examples:**

     * Louvain Method (modularity optimization)
     * Girvan-Newman (edge betweenness)
     * Label Propagation

3. **Analyze Detected Communities**

   * Identify natural clusters of clients and advisors
   * Spot isolated nodes or unusual communication patterns

4. **Take Action**

   * Optimize client-advisor allocation
   * Flag anomalous communication clusters
   * Create better collaboration or service models

---

Would you like me to **sketch a diagram (example graph)** showing how client-advisor communication gets grouped using community detection? (e.g., before = messy network, after = 3 clear client clusters).
import pandas as pd
import spacy
import networkx as nx

# -----------------------------
# 1. Load Structured Data
# -----------------------------
crm = pd.DataFrame([
    ["C101", "Alice Wong", "Gold", "APAC", "David Smith", "Premium Insurance"],
    ["C102", "Bob Allen", "Silver", "NA", "David Smith", "Home Loan"],
    ["C103", "Maria Evans", "Platinum", "APAC", "Sarah Johnson", "Wealth Portfolio"],
    ["C104", "John Reyes", "Gold", "EMEA", "Linda Brown", "Retirement Plan"]
], columns=["ClientID", "Name", "Tier", "Region", "Advisor", "Product"])

complaints = pd.DataFrame([
    ["T701", "C101", "Overcharged premium fee", "High", "Pending"],
    ["T702", "C103", "Portfolio losses > 10%", "Critical", "Open"],
    ["T703", "C104", "Retirement calc mismatch", "Medium", "Closed"]
], columns=["TicketID", "ClientID", "Issue Summary", "Severity", "Status"])

products = pd.DataFrame([
    ["Premium Insurance", "Medium"],
    ["Home Loan", "Low"],
    ["Wealth Portfolio", "High"],
    ["Retirement Plan", "Medium"]
], columns=["Product", "Risk"])

emails = [
    {"id": "E1", "text": "Hi David, I'm Alice Wong. I need an urgent upgrade of my Premium Insurance plan after a billing error."},
    {"id": "E2", "text": "This is Bob Allen. I need help with my Home Loan repayment; payment portal not working."},
    {"id": "E3", "text": "Maria Evans here. My Wealth Portfolio lost value last month. Thinking of moving funds to safer options."},
    {"id": "E4", "text": "John Reyes. Please review my Retirement Plan calculation; seems off compared to last year."}
]

nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 2. Build Knowledge Graph
# -----------------------------
G = nx.MultiDiGraph()

# Add CRM
for _, row in crm.iterrows():
    G.add_node(row["Name"], type="Client", Tier=row["Tier"], Region=row["Region"])
    risk = products.loc[products["Product"] == row["Product"], "Risk"].values[0]
    G.add_node(row["Product"], type="Product", Risk=risk)
    G.add_edge(row["Name"], row["Product"], relation="uses_product")
    G.add_node(row["Advisor"], type="Advisor")
    G.add_edge(row["Name"], row["Advisor"], relation="has_advisor")

# Add complaints
for _, row in complaints.iterrows():
    client_name = crm.loc[crm["ClientID"] == row["ClientID"], "Name"].values[0]
    G.add_node(row["TicketID"], type="Complaint", Severity=row["Severity"], Status=row["Status"])
    G.add_edge(client_name, row["TicketID"], relation="raised_ticket")

# Add emails (unstructured)
for email in emails:
    doc = nlp(email["text"])
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    products_mentioned = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
    G.add_node(email["id"], type="Email")
    for p in persons:
        if p not in G.nodes:
            G.add_node(p, type="Client")
        G.add_edge(p, email["id"], relation="sent_email")
    for prod in products_mentioned:
        if prod not in G.nodes:
            G.add_node(prod, type="Product")
        G.add_edge(email["id"], prod, relation="mentions_product")

# -----------------------------
# 3. Query function
# -----------------------------
def get_high_value_risk_clients(G, emails_data):
    results = []
    for node, data in G.nodes(data=True):
        if data.get('type') == "Client" and data.get('Tier') in ["Gold", "Platinum"] and data.get('Region') == "APAC":
            client = node

            # Complaints with pending/open status
            complaints_nodes = [
                nbr for nbr in G.neighbors(node)
                if any(e.get("relation") == "raised_ticket" for e in G.get_edge_data(node, nbr).values())
                and G.nodes[nbr].get("Status") in ["Pending", "Open"]
            ]
            if not complaints_nodes:
                continue

            # High/Medium risk products
            products_nodes = [
                nbr for nbr in G.neighbors(node)
                if any(e.get("relation") == "uses_product" for e in G.get_edge_data(node, nbr).values())
                and G.nodes[nbr].get("Risk") in ["High", "Medium"]
            ]
            if not products_nodes:
                continue

            # Emails mentioning issues
            emails_nodes = [
                nbr for nbr in G.neighbors(node)
                if any(e.get("relation") == "sent_email" for e in G.get_edge_data(node, nbr).values())
            ]
            issue_flag = False
            for e in emails_data:
                if e["id"] in emails_nodes and any(word in e["text"].lower() for word in ["error", "lost", "issue", "problem"]):
                    issue_flag = True

            if issue_flag:
                results.append({
                    "Client": client,
                    "Products": products_nodes,
                    "Complaints": complaints_nodes
                })
    return results

# -----------------------------
# 4. Run the business query
# -----------------------------
results = get_high_value_risk_clients(G, emails)

print("🔍 High-value clients in APAC with risk factors:")
for r in results:
    print(f"\nClient: {r['Client']}")
    print(f"Products: {', '.join(r['Products'])}")
    print(f"Complaints: {', '.join(r['Complaints'])}")
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
from collections import defaultdict

# -----------------------------
# 1️⃣ Create New Dataset
# -----------------------------
emails = [
    ("Client_A", "Advisor_1"), ("Client_B", "Advisor_1"), ("Client_C", "Advisor_1"),
    ("Client_D", "Advisor_2"), ("Client_E", "Advisor_2"), ("Client_F", "Advisor_2"),
    ("Client_G", "Advisor_3"), ("Client_H", "Advisor_3"),
    ("Client_I", "Advisor_4"), ("Client_J", "Advisor_4"),
    # Cross-communications to create overlaps
    ("Client_B", "Advisor_2"),    # Shared advisor (potential confusion)
    ("Client_E", "Advisor_3"),    # Shared advisor (cross-team interaction)
    ("Client_H", "Advisor_4"),    # Shared advisor
    ("Client_C", "Advisor_3"),    # Shared advisor
    ("Client_K", "Advisor_4"),    # New client
    ("Client_L", "Advisor_1"),    # New client
]

# -----------------------------
# 2️⃣ Build Graph
# -----------------------------
G = nx.Graph()
for client, advisor in emails:
    if G.has_edge(client, advisor):
        G[client][advisor]['weight'] += 1
    else:
        G.add_edge(client, advisor, weight=1)

# -----------------------------
# 3️⃣ Apply Louvain Community Detection
# -----------------------------
partition = community_louvain.best_partition(G)

# Group nodes into communities
communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

print("📌 Detected Communities:")
for comm_id, members in communities.items():
    print(f"Community {comm_id}: {members}")

# -----------------------------
# 4️⃣ Business Insights
# -----------------------------
# Find attrition, fraud, opportunities
insights = []

for comm_id, members in communities.items():
    clients = [m for m in members if "Client" in m]
    advisors = [m for m in members if "Advisor" in m]
    
    # Attrition Risk: client linked to multiple advisors
    for client in clients:
        linked_advisors = [nbr for nbr in G.neighbors(client) if "Advisor" in nbr]
        if len(linked_advisors) > 1:
            insights.append((client, "⚠️ Attrition Risk: Multiple advisors causing confusion."))

    # Fraud/Compliance Risk: advisors sharing multiple clients unusually
    for advisor in advisors:
        linked_clients = [nbr for nbr in G.neighbors(advisor) if "Client" in nbr]
        if len(linked_clients) > 4:  # unusual load threshold
            insights.append((advisor, "🔍 Possible oversight/fraud: unusually high client load."))

    # Opportunity: healthy large communities (≥3 clients per advisor)
    if len(clients) >= 3 and len(advisors) == 1:
        insights.append((advisors[0], "💡 Opportunity: Strong client base for upselling."))

# Print business insights
print("\n📢 Business Insights:")
for entity, msg in insights:
    print(f"{entity}: {msg}")

# -----------------------------
# 5️⃣ Visualize Communities
# -----------------------------
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
colors = [partition[n] for n in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3,
        node_size=1200, font_size=9, font_weight='bold')
plt.title("Detected Client-Advisor Communication Communities", fontsize=14)
plt.show()



Let's start **from scratch** and build a **clear, end-to-end understanding of community detection (CommDetection)** and how it applies to **client-advisor communication data**.

---

# **1️⃣ What is Community Detection?**

Community Detection is a **graph analysis technique** that **automatically finds groups (communities) of nodes** that are:

* **Highly connected internally** (many edges between them)
* **Weakly connected externally** (fewer edges to other nodes)

It **reveals natural structures** in a network **without predefined labels**.

---

## **2️⃣ Why is it Different from Clustering?**

While **clustering** (e.g., k-means) groups data based on features in a table (numeric similarity), **community detection:**

* Works **directly on graph structure**, not raw attributes
* Uses **connections (edges)** as the main driver
* Can find **overlapping and non-linear communities** that feature-based clustering may miss

Example difference:

| Feature                  | Clustering       | Community Detection            |
| ------------------------ | ---------------- | ------------------------------ |
| Data Input               | Tabular features | Network graph                  |
| Basis for grouping       | Distance metrics | Edge density/graph structure   |
| Predefined number needed | Often yes (k)    | No, finds groups automatically |
| Handles relationships    | ❌ No             | ✅ Yes                          |

---

# **3️⃣ Why Use Community Detection in Communication Data?**

In **client-advisor communication networks**:

* **Nodes:** Clients, Advisors, Products, Emails
* **Edges:** Email exchanges, shared advisors, interactions

Community detection helps answer **business-critical questions**:

* Are there **hidden client clusters** that share advisors or issues?
* Are some clients **underserved or isolated**?
* Are there **cross-advisor overlaps** that cause confusion or compliance risk?
* Can we identify **natural client segments** for upselling or retention strategies?

---

# **4️⃣ Simple Example**

### **Dataset (emails)**

| Sender    | Receiver   |
| --------- | ---------- |
| Client\_A | Advisor\_1 |
| Client\_B | Advisor\_1 |
| Client\_C | Advisor\_2 |
| Client\_B | Advisor\_2 |
| Client\_D | Advisor\_3 |

Graph:

* B is linked to both Advisor\_1 and Advisor\_2 (overlap).
* A and B form a community with Advisor\_1.
* B and C form another with Advisor\_2.

Community detection will **find these clusters automatically** instead of you having to define them.

---

# **5️⃣ How Do We Add Community Detection to a Graph?**

Steps:

1. **Build the Graph**

   * Create **nodes** for clients and advisors
   * Create **edges** for each communication (email, chat, call)

2. **Choose a Community Detection Algorithm**

   * **Louvain:** Finds modularity-based communities efficiently
   * **Girvan-Newman:** Uses edge betweenness but slower
   * **Label Propagation:** Fast and simple

3. **Run the Algorithm**

   * It assigns a **community ID** to each node.

4. **Analyze Results**

   * Each community = **group of clients and advisors** frequently interacting.
   * Overlaps, isolated clients, and large dense clusters can be flagged.

---

# **6️⃣ The Logic (How Louvain Works in Simple Terms)**

* **Goal:** Maximize a metric called **modularity (Q)** = how well a network divides into communities compared to random connections.
* **Process:**

  1. Start with each node as its own community.
  2. Merge nodes with neighbors if it **increases modularity** (stronger connection).
  3. Repeat until **no further improvement** is possible.
  4. Output = set of **communities** with high internal density and weak external links.

Think of it as:

> "Who talks most with whom? Group them if they talk a lot compared to the rest of the network."

---

# **7️⃣ Small Example Code**

```python
import networkx as nx
import community.community_louvain as community_louvain

emails = [
    ("Client_A", "Advisor_1"), ("Client_B", "Advisor_1"),
    ("Client_C", "Advisor_2"), ("Client_B", "Advisor_2"),
    ("Client_D", "Advisor_3")
]

# Build graph
G = nx.Graph()
G.add_edges_from(emails)

# Detect communities
partition = community_louvain.best_partition(G)

# Print results
for node, comm in partition.items():
    print(f"{node} -> Community {comm}")
```

### **Output**

```
Client_A -> Community 0
Client_B -> Community 0
Advisor_1 -> Community 0
Client_C -> Community 1
Advisor_2 -> Community 1
Client_D -> Community 2
Advisor_3 -> Community 2
```

✅ Communities formed based on communication patterns.

---

# **8️⃣ Business Use Cases**

* **Opportunities:** Identify **dense advisor-client clusters** for **cross-selling** or **relationship deepening**.
* **Attrition Risk:** Clients connected to **many advisors** or **few weak connections** → **confusion or neglect** risk.
* **Fraud/Compliance Risk:** **Unexpected advisor overlaps** (clients handled by multiple teams) may indicate **rule violations or misrouting**.

---

Would you like me to create a **single script** (new dataset + Louvain detection + flagged risks & opportunities) so you can see **business insights directly from communication data** in one run?
Sure! Let's break down the **Louvain Community Detection Algorithm** in a clear, structured way.

---

# **1️⃣ What is Louvain Community Detection?**

Louvain is a **graph-based algorithm** designed to find **communities** (groups of nodes that are densely connected internally but sparsely connected externally) **without knowing the number of communities in advance**.

It is **fast, scalable, and works well on large real-world networks**, which is why it’s widely used for social networks, communication networks, fraud networks, etc.

---

# **2️⃣ Key Idea**

The Louvain algorithm tries to **maximize modularity (Q)**, which measures:

* How many connections **exist inside communities**, versus
* How many connections **would exist if the graph was random**.

A high modularity value (close to 1) means:

* Many edges inside communities
* Few edges between communities
  → The communities are **well-formed and meaningful**.

---

# **3️⃣ How Louvain Works (Step-by-Step)**

Louvain runs in **two main phases**, repeatedly, until no improvement is possible:

---

### **Phase 1 – Local Movement of Nodes**

1. **Start:** Put each node in its **own community** (so we have N communities initially).
2. **Iterate over all nodes:**

   * Temporarily remove the node from its community.
   * Calculate the **gain in modularity** if the node is moved to each neighboring community.
   * Move the node to the **community that gives the highest modularity gain** (if > 0).
3. Repeat until **no individual node movement increases modularity**.

✅ At the end of this step, we have **small communities** where each node sits in the "best" place locally.

---

### **Phase 2 – Community Aggregation**

1. Build a **new, smaller network**:

   * Each community from Phase 1 becomes a **single node**.
   * Edges between communities are **weighted sums of connections** between their members.
2. Repeat **Phase 1** on this new network.

This process **hierarchically merges communities** until:

* Modularity can't be improved further.

---

## **4️⃣ Visual Example**

Imagine a **communication network**:

* Clients: A, B, C, D, E
* Advisors: 1, 2, 3

Edges (emails):

* A ↔ 1, B ↔ 1, B ↔ 2, C ↔ 2, D ↔ 3, E ↔ 3

The algorithm will:

* See that A-B-1 have many internal connections → merge them.
* See that B-C-2 are also tightly linked → merge them.
* Keep D-3 and E-3 as another community.
* The final modularity score is **maximized** when we have **two communities**, not one big or many tiny ones.

---

# **5️⃣ Why Louvain Works Well**

* **Automatic:** Finds number of communities itself (no need for k).
* **Efficient:** Works on networks with millions of nodes.
* **Hierarchical:** Produces communities at **different resolution levels** (can zoom in/out).
* **Weighted Graph Support:** Can use **frequency of emails** or **number of calls** as edge weights.

---

# **6️⃣ Intuition (Plain English)**

Think of it like **social circles in an office**:

* People email each other.
* Some groups of people talk frequently → natural "communities".
* Louvain "guesses" these circles by:

  * Looking at **who talks most to whom**,
  * Grouping them if it **makes the network more connected internally** than expected by chance.
* It keeps refining these circles until **no better grouping exists**.

---

# **7️⃣ Business Example**

In a **client-advisor network**:

* Clients emailing **only one advisor** → form clean communities.
* Clients emailing **multiple advisors** → cause overlapping communities (confusion).
* Louvain helps **map these patterns automatically**, supporting:

  * **Attrition risk detection:** Confused clients with multiple advisor links.
  * **Opportunity detection:** Dense, loyal communities of clients tied to one advisor.
  * **Fraud/compliance alerts:** Advisors in unexpected communities handling unrelated clients.

---

Would you like me to prepare a **diagram** (graph with nodes and colors) that visually shows **how Louvain merges communities step-by-step**, using a **small 6-node communication network example**? This would make the process visually intuitive.


Got it. Let’s make this **small, clear, and with a bit of math** to show exactly how Louvain works.

---

# **1️⃣ Problem Setup**

We have **5 nodes** in a communication graph:

* Clients: **A, B, C**
* Advisors: **X, Y**

Edges (emails):

* A ↔ X (weight 1)
* B ↔ X (weight 1)
* B ↔ Y (weight 1)
* C ↔ Y (weight 1)

This network can be visualized as:

```
A -- X
B -- X
B -- Y
C -- Y
```

We want to **detect communities automatically**.

---

# **2️⃣ What Louvain Tries to Do**

It maximizes **Modularity (Q)**:

$$
Q = \frac{1}{2m} \sum_{ij}\Big[A_{ij} - \frac{k_i k_j}{2m}\Big] \delta(c_i, c_j)
$$

Where:

* $A_{ij}$ = weight of edge between nodes $i,j$
* $k_i$ = degree (total weight) of node $i$
* $2m$ = sum of all edge weights
* $\delta(c_i, c_j)$ = 1 if $i,j$ are in same community, else 0

Intuition:

* $A_{ij}$: actual connections
* $k_i k_j / 2m$: expected connections if edges were random
* $Q$: **gain in “surprise connectivity”** when nodes are in the same community.

Goal → group nodes so that **Q is as large as possible**.

---

# **3️⃣ Louvain Steps on This Example**

### **Step 1: Start**

Each node is its **own community**:

* {A}, {B}, {C}, {X}, {Y}

---

### **Step 2: Try moving nodes**

We pick **B**, which talks to **X** and **Y**.

* **Option 1:** Merge B with X

  * Compute modularity gain (ΔQ):

$$
ΔQ = \big[\frac{1}{2m} (2*1) - \frac{(k_B+k_X)^2 - k_B^2 - k_X^2}{(2m)^2}\big]
$$

After simplifying (m=4 total edges):

* ΔQ ≈ positive value.

* **Option 2:** Merge B with Y

  * Similar calculation, also positive.

The algorithm chooses the **best positive ΔQ** (say B→X).

---

### **Step 3: Repeat for others**

* A merges with X (emails mostly with X)
* C merges with Y
* Communities now:

  * {A,B,X}, {C,Y}

---

### **Step 4: Build new graph**

Now we have **2 super-nodes**:

* Community 1: A,B,X
* Community 2: C,Y
  With 1 edge between them (B-Y connection).

No further merging improves Q → **stop**.

---

# **4️⃣ Final Communities**

* **Community 1:** {A, B, X} → clients A, B mostly handled by advisor X
* **Community 2:** {C, Y} → client C handled by advisor Y

---

# **5️⃣ Intuition Recap**

* Louvain checks if **moving a node to a neighbor’s community** increases modularity (more internal edges than expected).
* Builds **bigger and bigger groups** that are **more connected than random chance**.
* Stops when no further modularity gain is possible.

---

# **6️⃣ Business Interpretation**

* Advisor X has a **dense community** (A, B → one main contact).
* Advisor Y has a **loyal single client** (C).
* If B also talks a lot to Y (two advisors), Louvain may find an **overlap** or split communities differently → indicating **attrition risk or confusion**.

---

Would you like me to make a **simple 6-node figure (color-coded)** showing these **steps visually** (starting separate → merging communities → final 2 groups)? This would make Louvain **very clear in one glance**.
