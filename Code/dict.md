## 🐍 Python Dictionary — All Key Commands

### 📦 Create
```python
d = {"name": "Alice", "age": 25, "score": 90}
d = dict(name="Alice", age=25)  # alternate way
```

---

### ➕ Add / Update
```python
d["city"] = "Delhi"        # add new key
d["age"] = 26              # update existing key
d.update({"age": 26, "city": "Delhi"})  # update multiple
```

---

### 🔍 Search / Access
```python
d["name"]            # direct access (KeyError if missing)
d.get("name")        # safe access → returns None if missing
d.get("xyz", "N/A")  # default value if key not found

"age" in d           # check if key exists → True/False
```

---

### ❌ Delete
```python
del d["age"]          # delete by key
d.pop("age")          # delete + returns value
d.pop("age", None)    # safe pop (no error if missing)
d.popitem()           # removes last inserted key-value pair
d.clear()             # empty the whole dict
```

---

### 🔁 Loop / Traverse
```python
for key in d:
    print(key, d[key])

for key, val in d.items():   # best way
    print(key, val)

for key in d.keys():   print(key)
for val in d.values(): print(val)
```

---

### 🔃 Sort by Value
```python
d = {"b": 3, "a": 1, "c": 2}

# Sort by value → ascending
sorted_d = dict(sorted(d.items(), key=lambda x: x[1]))
# {'a': 1, 'c': 2, 'b': 3}

# Sort by value → descending
sorted_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
# {'b': 3, 'c': 2, 'a': 1}

# Sort by key
sorted_d = dict(sorted(d.items()))
```

---

### 🔽 Filter
```python
d = {"a": 10, "b": 3, "c": 7, "d": 1}

# Keep only values > 5
filtered = {k: v for k, v in d.items() if v > 5}
# {'a': 10, 'c': 7}

# Keep only specific keys
keys_to_keep = ["a", "c"]
filtered = {k: v for k, v in d.items() if k in keys_to_keep}
```

---

### 🧮 Other Useful Operations
```python
len(d)              # number of keys

d.copy()            # shallow copy

# Merge two dicts (Python 3.9+)
d3 = d1 | d2

# Merge older way
d3 = {**d1, **d2}

# Set default (adds key only if not present)
d.setdefault("age", 0)
```

---

### 🧠 Quick Cheat Sheet

| Operation | Command |
|---|---|
| Add/Update | `d[key] = val` |
| Safe Access | `d.get(key, default)` |
| Delete | `del d[key]` / `d.pop(key)` |
| Search key | `key in d` |
| Sort by value | `sorted(d.items(), key=lambda x: x[1])` |
| Filter | `{k:v for k,v in d.items() if condition}` |
| Loop | `for k, v in d.items()` |
| Merge | `d1 \| d2` |

