Let's break down each of these dynamic programming problems. I'll provide a clear problem statement, examples, and then walk through the different solution approaches (recursion, memoization, tabulation, optimized space if applicable, and alternatives), highlighting time/space complexity and intuitive insights.

-----

## Basic Dynamic Programming Problems

### 1\. Fibonacci Numbers

**Problem Statement:**
Given an integer $n$, return the $n$-th Fibonacci number. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1.
$F(0) = 0$
$F(1) = 1$
$F(n) = F(n-1) + F(n-2)$ for $n \> 1$

**Example:**
Input: $n = 5$
Output: $5$ ($0, 1, 1, 2, 3, 5$)

-----

#### Recursion (Brute Force)

**Intuitive Insight:** The definition of Fibonacci numbers is inherently recursive. To find $F(n)$, we directly apply the formula.

**Code:**

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# Example usage:
print(f"Fibonacci(5) (Recursive): {fib_recursive(5)}") # Output: 5
print(f"Fibonacci(0) (Recursive): {fib_recursive(0)}") # Output: 0
print(f"Fibonacci(1) (Recursive): {fib_recursive(1)}") # Output: 1
```

**Time Complexity:** $O(2^n)$ - Each call branches into two more calls, leading to an exponential number of computations due to redundant calculations of subproblems (e.g., `fib(2)` is calculated multiple times when computing `fib(5)`).
**Space Complexity:** $O(n)$ - Due to the recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** We observe overlapping subproblems in the recursive solution. We can store the results of expensive function calls and return the cached result when the same inputs occur again.

**Code:**

```python
def fib_memoization(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    
    memo[n] = fib_memoization(n - 1, memo) + fib_memoization(n - 2, memo)
    return memo[n]

# Example usage:
print(f"Fibonacci(5) (Memoization): {fib_memoization(5)}") # Output: 5
print(f"Fibonacci(10) (Memoization): {fib_memoization(10)}") # Output: 55
```

**Time Complexity:** $O(n)$ - Each Fibonacci number from 0 to $n$ is computed only once.
**Space Complexity:** $O(n)$ - For the memoization table (dictionary/array) and the recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Instead of starting from $n$ and going down, we can build up the solution from the base cases ($F(0)$ and $F(1)$) to $F(n)$. This eliminates recursion overhead and stack space.

**Code:**

```python
def fib_tabulation(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        
    return dp[n]

# Example usage:
print(f"Fibonacci(5) (Tabulation): {fib_tabulation(5)}") # Output: 5
print(f"Fibonacci(10) (Tabulation): {fib_tabulation(10)}") # Output: 55
```

**Time Complexity:** $O(n)$ - A single loop iterates $n$ times.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** To calculate $F(n)$, we only need $F(n-1)$ and $F(n-2)$. We don't need to store all previous Fibonacci numbers. We can use just two variables to keep track of the last two values.

**Code:**

```python
def fib_optimized_space(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage:
print(f"Fibonacci(5) (Optimized Space): {fib_optimized_space(5)}") # Output: 5
print(f"Fibonacci(10) (Optimized Space): {fib_optimized_space(10)}") # Output: 55
```

**Time Complexity:** $O(n)$ - A single loop iterates $n$ times.
**Space Complexity:** $O(1)$ - Only a constant number of variables are used.

-----

#### Alternatives (Mathematical Formula - Binet's Formula)

**Intuitive Insight:** There's a closed-form expression for the $n$-th Fibonacci number, derived using linear recurrence relation techniques.

**Code:**

```python
import math

def fib_binet(n):
    if n < 0:
        return "Invalid input for Fibonacci (n must be non-negative)"
    if n <= 1:
        return n
    
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    
    # Using round for integer result due to floating point inaccuracies
    return round((phi**n - psi**n) / math.sqrt(5))

# Example usage:
print(f"Fibonacci(5) (Binet's Formula): {fib_binet(5)}") # Output: 5
print(f"Fibonacci(10) (Binet's Formula): {fib_binet(10)}") # Output: 55
```

**Time Complexity:** $O(\\log n)$ - Due to the `math.pow` operation.
**Space Complexity:** $O(1)$

-----

### 2\. Tribonacci Numbers

**Problem Statement:**
Given an integer $n$, return the $n$-th Tribonacci number. The Tribonacci sequence is similar to the Fibonacci sequence, but each number is the sum of the three preceding ones.
$T(0) = 0$
$T(1) = 1$
$T(2) = 1$
$T(n) = T(n-1) + T(n-2) + T(n-3)$ for $n \> 2$

**Example:**
Input: $n = 4$
Output: $4$ ($0, 1, 1, 2, 4$)

-----

#### Recursion (Brute Force)

**Intuitive Insight:** Direct application of the recursive definition.

**Code:**

```python
def trib_recursive(n):
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    return trib_recursive(n - 1) + trib_recursive(n - 2) + trib_recursive(n - 3)

# Example usage:
print(f"Tribonacci(4) (Recursive): {trib_recursive(4)}") # Output: 4
print(f"Tribonacci(0) (Recursive): {trib_recursive(0)}") # Output: 0
print(f"Tribonacci(1) (Recursive): {trib_recursive(1)}") # Output: 1
print(f"Tribonacci(2) (Recursive): {trib_recursive(2)}") # Output: 1
```

**Time Complexity:** $O(3^n)$ - Exponential due to redundant calculations.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store computed Tribonacci numbers to avoid re-calculating them.

**Code:**

```python
def trib_memoization(n, memo={}):
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    if n in memo:
        return memo[n]
    
    memo[n] = trib_memoization(n - 1, memo) + trib_memoization(n - 2, memo) + trib_memoization(n - 3, memo)
    return memo[n]

# Example usage:
print(f"Tribonacci(4) (Memoization): {trib_memoization(4)}") # Output: 4
print(f"Tribonacci(7) (Memoization): {trib_memoization(7)}") # Output: 24
```

**Time Complexity:** $O(n)$ - Each Tribonacci number up to $n$ is computed once.
**Space Complexity:** $O(n)$ - For the memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build up the `dp` array from base cases to $T(n)$.

**Code:**

```python
def trib_tabulation(n):
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        
    return dp[n]

# Example usage:
print(f"Tribonacci(4) (Tabulation): {trib_tabulation(4)}") # Output: 4
print(f"Tribonacci(7) (Tabulation): {trib_tabulation(7)}") # Output: 24
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** To calculate $T(n)$, we only need $T(n-1)$, $T(n-2)$, and $T(n-3)$. We can use three variables.

**Code:**

```python
def trib_optimized_space(n):
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    a, b, c = 0, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c

# Example usage:
print(f"Tribonacci(4) (Optimized Space): {trib_optimized_space(4)}") # Output: 4
print(f"Tribonacci(7) (Optimized Space): {trib_optimized_space(7)}") # Output: 24
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(1)$ - Constant number of variables.

-----

### 3\. Lucas Numbers

**Problem Statement:**
Given an integer $n$, return the $n$-th Lucas number. The Lucas numbers are a sequence where each number is the sum of the two preceding ones, similar to Fibonacci, but with different starting values.
$L(0) = 2$
$L(1) = 1$
$L(n) = L(n-1) + L(n-2)$ for $n \> 1$

**Example:**
Input: $n = 5$
Output: $11$ ($2, 1, 3, 4, 7, 11$)

-----

#### Recursion (Brute Force)

**Intuitive Insight:** Directly apply the recursive definition with the specific base cases for Lucas numbers.

**Code:**

```python
def lucas_recursive(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    return lucas_recursive(n - 1) + lucas_recursive(n - 2)

# Example usage:
print(f"Lucas(5) (Recursive): {lucas_recursive(5)}") # Output: 11
print(f"Lucas(0) (Recursive): {lucas_recursive(0)}") # Output: 2
print(f"Lucas(1) (Recursive): {lucas_recursive(1)}") # Output: 1
```

**Time Complexity:** $O(2^n)$ - Exponential due to redundant calculations.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store computed Lucas numbers to avoid re-calculating them.

**Code:**

```python
def lucas_memoization(n, memo={}):
    if n == 0:
        return 2
    if n == 1:
        return 1
    if n in memo:
        return memo[n]
    
    memo[n] = lucas_memoization(n - 1, memo) + lucas_memoization(n - 2, memo)
    return memo[n]

# Example usage:
print(f"Lucas(5) (Memoization): {lucas_memoization(5)}") # Output: 11
print(f"Lucas(8) (Memoization): {lucas_memoization(8)}") # Output: 47
```

**Time Complexity:** $O(n)$ - Each Lucas number up to $n$ is computed once.
**Space Complexity:** $O(n)$ - For the memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build up the `dp` array from base cases to $L(n)$.

**Code:**

```python
def lucas_tabulation(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 2
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        
    return dp[n]

# Example usage:
print(f"Lucas(5) (Tabulation): {lucas_tabulation(5)}") # Output: 11
print(f"Lucas(8) (Tabulation): {lucas_tabulation(8)}") # Output: 47
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** To calculate $L(n)$, we only need $L(n-1)$ and $L(n-2)$. We can use just two variables.

**Code:**

```python
def lucas_optimized_space(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage:
print(f"Lucas(5) (Optimized Space): {lucas_optimized_space(5)}") # Output: 11
print(f"Lucas(8) (Optimized Space): {lucas_optimized_space(8)}") # Output: 47
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(1)$ - Constant number of variables.

-----

### 4\. Climbing Stairs

**Problem Statement:**
You are climbing a staircase. It takes $n$ steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Example:**
Input: $n = 3$
Output: $3$
Explanation:

1.  1 step + 1 step + 1 step
2.  1 step + 2 steps
3.  2 steps + 1 step

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To reach step $n$, you must have come from either step $n-1$ (by taking 1 step) or step $n-2$ (by taking 2 steps). The total ways to reach $n$ is the sum of ways to reach $n-1$ and ways to reach $n-2$. This is exactly the Fibonacci sequence\!

**Code:**

```python
def climb_stairs_recursive(n):
    if n == 0: # Base case: 1 way to be at step 0 (do nothing) or consider 0 steps to reach
        return 1 
    if n < 0: # Invalid state, cannot climb negative steps
        return 0
    
    return climb_stairs_recursive(n - 1) + climb_stairs_recursive(n - 2)

# Example usage:
print(f"Climbing Stairs(3) (Recursive): {climb_stairs_recursive(3)}") # Output: 3
print(f"Climbing Stairs(1) (Recursive): {climb_stairs_recursive(1)}") # Output: 1 (1 way: 1 step)
print(f"Climbing Stairs(2) (Recursive): {climb_stairs_recursive(2)}") # Output: 2 (1+1, 2)
```

**Time Complexity:** $O(2^n)$ - Exponential, similar to Fibonacci.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store the results of subproblems to avoid recomputing.

**Code:**

```python
def climb_stairs_memoization(n, memo={}):
    if n == 0:
        return 1
    if n < 0:
        return 0
    if n in memo:
        return memo[n]
    
    memo[n] = climb_stairs_memoization(n - 1, memo) + climb_stairs_memoization(n - 2, memo)
    return memo[n]

# Example usage:
print(f"Climbing Stairs(3) (Memoization): {climb_stairs_memoization(3)}") # Output: 3
print(f"Climbing Stairs(10) (Memoization): {climb_stairs_memoization(10)}") # Output: 89
```

**Time Complexity:** $O(n)$ - Each subproblem is computed once.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build up the solution from the base cases (reaching step 0, step 1, etc.) to step $n$.

**Code:**

```python
def climb_stairs_tabulation(n):
    if n == 0:
        return 1
    if n == 1:
        return 1 # Or dp[0]=1, dp[1]=1. Both work depending on interpretation.
    
    dp = [0] * (n + 1)
    dp[0] = 1 # There's 1 way to reach step 0 (do nothing)
    dp[1] = 1 # There's 1 way to reach step 1 (take 1 step from step 0)
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        
    return dp[n]

# Example usage:
print(f"Climbing Stairs(3) (Tabulation): {climb_stairs_tabulation(3)}") # Output: 3
print(f"Climbing Stairs(10) (Tabulation): {climb_stairs_tabulation(10)}") # Output: 89
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** Only the previous two results are needed.

**Code:**

```python
def climb_stairs_optimized_space(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    
    a, b = 1, 1 # a is ways to reach i-2, b is ways to reach i-1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage:
print(f"Climbing Stairs(3) (Optimized Space): {climb_stairs_optimized_space(3)}") # Output: 3
print(f"Climbing Stairs(10) (Optimized Space): {climb_stairs_optimized_space(10)}") # Output: 89
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(1)$ - Constant variables.

-----

### 5\. Climbing Stairs with 3 Moves

**Problem Statement:**
You are climbing a staircase. It takes $n$ steps to reach the top. Each time you can either climb 1, 2, or 3 steps. In how many distinct ways can you climb to the top?

**Example:**
Input: $n = 3$
Output: $4$
Explanation:

1.  1+1+1
2.  1+2
3.  2+1
4.  3

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To reach step $n$, you could have taken 1 step from $n-1$, 2 steps from $n-2$, or 3 steps from $n-3$. Summing these possibilities gives the total ways. This is the Tribonacci sequence.

**Code:**

```python
def climb_stairs_3_moves_recursive(n):
    if n == 0:
        return 1
    if n < 0:
        return 0
    
    return (climb_stairs_3_moves_recursive(n - 1) + 
            climb_stairs_3_moves_recursive(n - 2) + 
            climb_stairs_3_moves_recursive(n - 3))

# Example usage:
print(f"Climbing Stairs with 3 moves(3) (Recursive): {climb_stairs_3_moves_recursive(3)}") # Output: 4
print(f"Climbing Stairs with 3 moves(4) (Recursive): {climb_stairs_3_moves_recursive(4)}") # Output: 7
```

**Time Complexity:** $O(3^n)$ - Exponential.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store computed results to avoid redundant calculations.

**Code:**

```python
def climb_stairs_3_moves_memoization(n, memo={}):
    if n == 0:
        return 1
    if n < 0:
        return 0
    if n in memo:
        return memo[n]
    
    memo[n] = (climb_stairs_3_moves_memoization(n - 1, memo) + 
               climb_stairs_3_moves_memoization(n - 2, memo) + 
               climb_stairs_3_moves_memoization(n - 3, memo))
    return memo[n]

# Example usage:
print(f"Climbing Stairs with 3 moves(3) (Memoization): {climb_stairs_3_moves_memoization(3)}") # Output: 4
print(f"Climbing Stairs with 3 moves(4) (Memoization): {climb_stairs_3_moves_memoization(4)}") # Output: 7
```

**Time Complexity:** $O(n)$ - Each subproblem computed once.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build up the `dp` array.

**Code:**

```python
def climb_stairs_3_moves_tabulation(n):
    if n == 0:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 1 # Base case: 1 way to reach step 0
    
    # Initialize dp[1] and dp[2] carefully based on problem interpretation
    # If starting from 0:
    # dp[1] = dp[0] = 1 (1 way: 1 step)
    # dp[2] = dp[1] + dp[0] = 1 + 1 = 2 (1+1, 2)
    # This is consistent if n is allowed to be 0, 1, 2 etc.
    if n >= 1:
        dp[1] = 1
    if n >= 2:
        dp[2] = dp[1] + dp[0] # From step 1 (1 step) or step 0 (2 steps)
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        
    return dp[n]

# Example usage:
print(f"Climbing Stairs with 3 moves(3) (Tabulation): {climb_stairs_3_moves_tabulation(3)}") # Output: 4
print(f"Climbing Stairs with 3 moves(4) (Tabulation): {climb_stairs_3_moves_tabulation(4)}") # Output: 7
print(f"Climbing Stairs with 3 moves(1) (Tabulation): {climb_stairs_3_moves_tabulation(1)}") # Output: 1
print(f"Climbing Stairs with 3 moves(2) (Tabulation): {climb_stairs_3_moves_tabulation(2)}") # Output: 2
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** Only the last three results are needed.

**Code:**

```python
def climb_stairs_3_moves_optimized_space(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2 # 1+1, 2
    
    a, b, c = 1, 1, 2 # a=dp[i-3], b=dp[i-2], c=dp[i-1]
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c

# Example usage:
print(f"Climbing Stairs with 3 moves(3) (Optimized Space): {climb_stairs_3_moves_optimized_space(3)}") # Output: 4
print(f"Climbing Stairs with 3 moves(4) (Optimized Space): {climb_stairs_3_moves_optimized_space(4)}") # Output: 7
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(1)$ - Constant variables.

-----

### 6\. Weighted Climbing Stairs

**Problem Statement:**
You are climbing a staircase. It takes $n$ steps to reach the top. You are given an array `cost` where `cost[i]` is the cost of taking the $i$-th step. You can either climb 1 or 2 steps. You can start from step 0 or step 1. You need to find the minimum cost to reach the top of the floor (which is one step beyond the last index in `cost`).

**Example:**
Input: `cost = [10, 15, 20]`
Output: $15$
Explanation:
Option 1: Start at index 1, take two steps. Cost = $15$.
Option 2: Start at index 0, take one step, then one step. Cost = $10 + 20 = 30$.
Option 3: Start at index 0, take two steps. Cost = $10 + 15 = 25$.
Minimum cost is 15.

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To reach step $i$, you could have come from step $i-1$ (paying `cost[i-1]`) or step $i-2$ (paying `cost[i-2]`). We want the minimum cost, so we choose the path that has the minimum total cost. The "top" is effectively `n` steps, where `n` is `len(cost)`.

**Code:**

```python
def min_cost_climbing_stairs_recursive(cost):
    n = len(cost)

    # Helper function to find min cost to reach a given step
    def find_min_cost(step):
        if step < 0:
            return 0 # No cost if we are before the starting steps
        if step == 0 or step == 1:
            return cost[step] # Cost to land on these steps

        # Min cost to reach 'step' is cost[step] + min(cost to reach step-1, cost to reach step-2)
        # Note: The problem asks for cost to reach *top of floor*, which is one step *beyond* the last index.
        # This implies we consider the cost of *landing* on step i.
        # A common interpretation is that the `cost[i]` is paid *after* landing on step `i`.
        # So to reach 'top' (index `n`), you must have come from `n-1` or `n-2`.
        # The cost is then `min(find_min_cost(n-1), find_min_cost(n-2))`
        # Let's adjust the recurrence to match the standard interpretation of the problem.

        # The problem statement typically implies that `cost[i]` is paid *to take* step `i`.
        # If we are at step `i`, and we take 1 or 2 steps, what's the cost?
        # Let's re-interpret: `dp[i]` is the minimum cost to reach *step i*.
        # To reach `n` (the top), you either came from `n-1` or `n-2`.
        # The base cases would be the cost of starting at 0 or 1.

        # Let's consider `dp[i]` as the minimum cost to reach step `i`.
        # The goal is to reach `n` (after the last step `n-1`).
        # So we need `min(dp[n-1], dp[n-2])` where `dp[i]` is the cost to reach `i`.

        # Let's define `dp[i]` as the minimum cost to reach the *i-th step*.
        # The actual steps are from 0 to n-1. The "top" is effectively step n.
        # So, to reach step `i` (where `i` can be `n`), you must have come from `i-1` or `i-2`.
        # The cost to reach `i` is `cost[i]` + minimum cost to reach `i-1` or `i-2`.
        # For the "top" (index `n`), there is no cost, just the cost to arrive.

        # Let's use `memo` to store the minimum cost to reach a certain index `i`.
        # We need to reach `n`. The steps are `0, 1, ..., n-1`.
        # To reach `n`, we came from `n-1` or `n-2`.
        # cost to reach `n` = min(cost to reach `n-1`, cost to reach `n-2`)
        # cost to reach `i` = cost[i] + min(cost to reach `i-1`, cost to reach `i-2`) (for i >= 2)

        if step == n: # Reached the top (one step beyond last index)
            return 0
        if step > n: # Out of bounds
            return float('inf')

        cost_current = cost[step] if step >= 0 else 0

        # Option 1: Take 1 step from current position
        cost_one_step = find_min_cost(step + 1)
        # Option 2: Take 2 steps from current position
        cost_two_steps = find_min_cost(step + 2)

        return cost_current + min(cost_one_step, cost_two_steps)

    # We can start from step 0 or step 1.
    return min(find_min_cost(0), find_min_cost(1))

# This recursive solution above is a bit complex to set up correctly
# due to the "starting from 0 or 1" and "reaching top (n)" aspects.

# A more standard recursive definition for this problem:
# `dp[i]` represents the minimum cost to reach step `i`.
# The last step is `n-1`. The "top" is `n`.
# We need `min(dp[n-1], dp[n-2])` to reach the top.
# `dp[i] = cost[i] + min(dp[i-1], dp[i-2])`

def min_cost_climbing_stairs_recursive_simplified(cost):
    n = len(cost)

    # memo = {} # For memoization later

    def solve(i):
        # Base cases:
        # If we are at index n (the "top"), the cost is 0 as we've already paid to get there.
        if i == n:
            return 0
        # If we are at n+1, we overshot, so return infinity.
        if i > n:
            return float('inf')

        # If we are at a valid step (0 to n-1), we pay cost[i]
        # and then choose the minimum path from (i+1) or (i+2).
        cost_to_take_current_step = cost[i]

        return cost_to_take_current_step + min(solve(i + 1), solve(i + 2))

    # We can start from step 0 or step 1.
    return min(solve(0), solve(1))


# Example usage:
# print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Recursive): {min_cost_climbing_stairs_recursive_simplified([10, 15, 20])}") # Output: 15
# This `_simplified` version is also tricky because it calculates cost *from* a step.
# The common DP state is `dp[i]` = min cost to reach step `i`.

# Let's use the standard DP state for recursion:
# `dp[i]` = minimum cost to reach step `i`.
# Target: `min(dp[n-1], dp[n-2])` because we can land on either the last step or the second to last step and then take our final 1 or 2 steps to go *beyond* the array.

def min_cost_climbing_stairs_recursive_correct(cost):
    n = len(cost)

    # Define a helper function to calculate the minimum cost to reach step 'i'
    # 'i' here refers to the index in the cost array.
    # The 'top' is considered to be one step *after* the last index, i.e., index 'n'.
    # So we want to find the minimum cost to reach index 'n'.
    
    # Memoization dictionary to store results
    memo = {}

    def get_min_cost(i):
        # Base cases:
        # If we are trying to reach index 0 or 1, the cost is the cost at that index.
        # This is because we can start directly from step 0 or step 1.
        if i == 0:
            return cost[0]
        if i == 1:
            return cost[1]
        
        # If the result is already memoized, return it
        if i in memo:
            return memo[i]

        # Recurrence relation:
        # The minimum cost to reach step 'i' is the cost of step 'i'
        # plus the minimum of (cost to reach step 'i-1', cost to reach step 'i-2').
        # We need to consider that for `n`, there is no `cost[n]`.
        # If `i` is out of bounds (past `n`), it should be handled differently.

        # Let's think about `dp[i]` as the minimum cost to get *to* step `i`.
        # And we want the minimum cost to get *to the end* (index `n`).

        # The standard way to model this is `dp[i]` = minimum cost to reach `i` steps.
        # So `dp[n]` is the target.
        # `dp[0]` = 0 (cost to reach the bottom, before any steps)
        # `dp[1]` = 0 (cost to reach step 1 from bottom, no cost from step itself for now)
        # Or, `dp[i]` means minimum cost to reach the *i-th floor* (where steps are 0 to n-1, floor is n).
        
        # Let `dp[i]` be the minimum cost to reach the `i`-th step (index `i-1` in `cost`).
        # `dp[0]` = 0
        # `dp[1]` = cost[0]
        # `dp[2]` = min(cost[0] + cost[1], cost[1])  -- this is getting confusing.

        # Let's use the most common interpretation:
        # `dp[i]` is the minimum cost to reach *just after* step `i`.
        # So `dp[n]` is the answer.
        # To reach `dp[i]`, we could have come from `i-1` or `i-2`.
        # `dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`
        # Base cases: `dp[0] = 0`, `dp[1] = 0` (cost to reach "before" first step or "before" second step)

        # This makes the recursion start from the "end".
        # This is tricky for recursive brute force without memoization.

        # The "reaching the top" means reaching `n` steps, where `cost` has `n` elements.
        # This means reaching an imaginary `n`-th step.
        # Let `f(k)` be the minimum cost to reach step `k`.
        # `f(k) = cost[k] + min(f(k-1), f(k-2))` (if k is not 0 or 1)
        # `f(0) = cost[0]`
        # `f(1) = cost[1]`
        # The answer is `min(f(n-1), f(n-2))` (cost to reach top *from* last two steps).

        # Let's use this definition, but with memoization for recursion.
        
        # Base cases
        if i >= n: # If we are at or beyond the "top" (index n), no further cost is added from this point
            return 0

        if i in memo:
            return memo[i]

        # Calculate cost if we are at step i, and move one or two steps
        cost_for_this_step = cost[i]

        # Option 1: Take 1 step from current step `i`
        cost_from_one_step = get_min_cost(i + 1)
        
        # Option 2: Take 2 steps from current step `i`
        cost_from_two_steps = get_min_cost(i + 2)

        memo[i] = cost_for_this_step + min(cost_from_one_step, cost_from_two_steps)
        return memo[i]

    # We can start from step 0 or step 1.
    # The minimum cost to reach the top will be the minimum of starting from step 0
    # or starting from step 1.
    return min(get_min_cost(0), get_min_cost(1))

# Example usage:
print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Recursive Correct - but still slow): {min_cost_climbing_stairs_recursive_correct([10, 15, 20])}") # Output: 15
# This recursive approach, while correct, implicitly explores paths and can be exponential.
# For brute force, we really need to build the tree.
# The previous recursive solution for "Climbing Stairs" was simple because it counted ways, not minimum cost.

# Let's consider a simpler recursive structure for brute force:
# `f(i)` = minimum cost to reach step `i` (from the beginning)
# `f(i) = cost[i] + min(f(i-1), f(i-2))`
# This isn't quite right because `cost[i]` is paid *to take* step `i`.
# The problem statement: "cost[i] is the cost of taking the i-th step."
# This means if you land on step `i`, you incur `cost[i]`.
# We want to reach the floor *after* the last step.

# Let `dp[i]` be the min cost to reach step `i`.
# `dp[0]` = cost[0]
# `dp[1]` = cost[1]
# For `i >= 2`: `dp[i] = cost[i] + min(dp[i-1], dp[i-2])`
#  `min(dp[n-1], dp[n-2])`

def min_cost_climbing_stairs_recursive_final_brute(cost):
    n = len(cost)
    
    def solve(i):
        # Base cases:
        if i < 0:
            return 0 # No cost before starting
        if i == 0 or i == 1:
            return cost[i] # Cost to land on these initial steps

        # Recursive step: Cost to land on step i is cost[i]
        # plus the minimum cost to land on step i-1 or step i-2.
        return cost[i] + min(solve(i - 1), solve(i - 2))

    # The problem asks for the minimum cost to reach the top of the floor,
    # which implies going *past* the last step (index n-1).
    # So we want to find the minimum cost to step onto an imaginary 'n'th step.
    # This means either coming from step n-1 or step n-2.
    return min(solve(n - 1), solve(n - 2))

print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Recursive Brute Force): {min_cost_climbing_stairs_recursive_final_brute([10, 15, 20])}") # Output: 15
print(f"Min Cost Climbing Stairs ([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) (Recursive Brute Force): {min_cost_climbing_stairs_recursive_final_brute([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])}") # Output: 6
```

**Time Complexity:** $O(2^n)$ - Exponential due to repeated calculations.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** We observe overlapping subproblems. Store the minimum cost to reach each step.

**Code:**

```python
def min_cost_climbing_stairs_memoization(cost):
    n = len(cost)
    memo = {}

    def solve(i):
        if i < 0:
            return 0
        if i == 0 or i == 1:
            return cost[i]
        
        if i in memo:
            return memo[i]
        
        memo[i] = cost[i] + min(solve(i - 1), solve(i - 2))
        return memo[i]

    return min(solve(n - 1), solve(n - 2))

# Example usage:
print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Memoization): {min_cost_climbing_stairs_memoization([10, 15, 20])}") # Output: 15
print(f"Min Cost Climbing Stairs ([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) (Memoization): {min_cost_climbing_stairs_memoization([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])}") # Output: 6
```

**Time Complexity:** $O(n)$ - Each `solve(i)` is computed once.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build the `dp` array where `dp[i]` is the minimum cost to reach step `i`.

**Code:**

```python
def min_cost_climbing_stairs_tabulation(cost):
    n = len(cost)
    
    # dp[i] will store the minimum cost to reach step i.
    # Since we can start from index 0 or 1, dp[0] = cost[0] and dp[1] = cost[1].
    # The 'top' is considered to be one step *after* the last element.
    # So we need to consider the cost to reach index `n`.
    # A cleaner way is to let `dp[i]` be the minimum cost to reach a position *before* step `i`.
    # Or, let `dp[i]` be the min cost to reach step `i` (meaning, you've just stepped onto `i`).
    
    # Let's adjust the definition to be: `dp[i]` is the minimum cost to reach the *i-th floor*.
    # So `dp[0]` is cost to reach floor 0 (before step 0).
    # `dp[1]` is cost to reach floor 1 (after step 0).
    # `dp[n]` is cost to reach floor n (after step n-1). This is our target.
    
    # `dp` array size `n+1` for `n` floors (from 0 to n)
    dp = [0] * (n + 1)
    
    # Base cases:
    # To reach floor 0 (before step 0), cost is 0.
    # To reach floor 1 (after step 0), we can start from 0 and pay cost[0], or start from 1.
    # This standard problem often defines `dp[i]` as the minimum cost to *climb to* step `i`.
    # `dp[i]` is the min cost to reach a position just *after* step `i-1`.
    
    # A standard and clean approach for this problem:
    # `dp[i]` = minimum cost to reach step `i` (index `i-1` in `cost` array).
    # We want to find `min(dp[n-1], dp[n-2])`.
    # This is for when `cost[i]` is paid *to land on* step `i`.

    # Let's use `dp[i]` as the minimum cost to reach the *i-th index in the cost array*.
    # Then the final answer will be `min(dp[n-1], dp[n-2])`.
    
    # Initialize dp array with size n.
    dp = [0] * n
    
    # Base cases
    if n >= 1:
        dp[0] = cost[0]
    if n >= 2:
        dp[1] = cost[1]
    
    # Fill dp table
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
        
    # The result is the minimum cost to reach the last step (n-1) or the second to last step (n-2)
    # because from either of these, we can take the final 1 or 2 steps to "the top" (after array).
    # This implies that reaching the *top* doesn't incur additional cost beyond reaching the last step.
    return min(dp[n - 1], dp[n - 2])

# Another common interpretation, often found in LeetCode solutions:
# `dp[i]` is the minimum cost to reach `i` steps.
# The `cost` array has `n` elements. The "top" is `n`.
# `dp[0] = 0` (cost to reach start)
# `dp[1] = 0` (cost to reach start) (if allowed to start from 0 or 1)
# `dp[i]` = min( `dp[i-1] + cost[i-1]`, `dp[i-2] + cost[i-2]` )

def min_cost_climbing_stairs_tabulation_preferred(cost):
    n = len(cost)
    
    # dp[i] represents the minimum cost to reach index 'i' (which is actually step i+1).
    # Or, dp[i] is the minimum cost to reach the 'i-th floor'
    # where floor 0 is before cost[0], floor 1 is after cost[0], etc.
    # The 'top' is floor 'n'.
    
    # Initialize dp array of size n+1 (for floors 0 to n)
    dp = [0] * (n + 1)
    
    # Base cases:
    # Cost to reach floor 0 (before step 0) is 0.
    # Cost to reach floor 1 (after step 0, taking 1 step from floor 0) is 0.
    # This allows us to "start" at 0 or 1 without initial cost from outside.
    # The cost `cost[i]` is incurred when you *land on* step `i`.
    
    # For this problem: `dp[i]` = min cost to reach *just beyond* index `i-1`.
    # So `dp[len(cost)]` is the answer.
    # `dp[0]` = 0
    # `dp[1]` = 0
    
    # Iterate from index 2 up to n (the "top" floor)
    for i in range(2, n + 1):
        # To reach floor `i`, we could have come from:
        # 1. Floor `i-1` by taking a 1-step from `i-1` (cost `cost[i-1]`)
        # 2. Floor `i-2` by taking a 2-step from `i-2` (cost `cost[i-2]`)
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        
    return dp[n]

# Example usage:
print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Tabulation Preferred): {min_cost_climbing_stairs_tabulation_preferred([10, 15, 20])}") # Output: 15
print(f"Min Cost Climbing Stairs ([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) (Tabulation Preferred): {min_cost_climbing_stairs_tabulation_preferred([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])}") # Output: 6
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** We only need the previous two `dp` values to calculate the current one.

**Code:**

```python
def min_cost_climbing_stairs_optimized_space(cost):
    n = len(cost)
    
    # dp_prev_2 represents dp[i-2]
    # dp_prev_1 represents dp[i-1]
    dp_prev_2 = 0
    dp_prev_1 = 0
    
    for i in range(2, n + 1):
        current_dp = min(dp_prev_1 + cost[i - 1], dp_prev_2 + cost[i - 2])
        dp_prev_2 = dp_prev_1
        dp_prev_1 = current_dp
        
    return dp_prev_1

# Example usage:
print(f"Min Cost Climbing Stairs ([10, 15, 20]) (Optimized Space): {min_cost_climbing_stairs_optimized_space([10, 15, 20])}") # Output: 15
print(f"Min Cost Climbing Stairs ([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) (Optimized Space): {min_cost_climbing_stairs_optimized_space([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])}") # Output: 6
```

**Time Complexity:** $O(n)$ - Single loop.
**Space Complexity:** $O(1)$ - Constant variables.

-----

### 7\. Maximum Segments

**Problem Statement:**
You are given an integer $n$ and a list of integers `cuts` representing possible segment lengths. You want to cut a rope of length $n$ into pieces such that the length of each piece is one of the `cuts` values. Find the maximum number of pieces you can get. If it's not possible to cut the rope into any valid pieces, return -1.

**Example:**
Input: $n = 5$, `cuts = [2, 3]`
Output: $2$
Explanation: You can cut the rope into pieces of length 2 and 3 (2+3=5), giving 2 pieces. You cannot cut it into 2+2 (remaining 1, not in cuts) or 3+3 (exceeds 5).

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To find the maximum segments for length $n$, we can try to cut off each possible `cut` length. If we cut off `c` from $n$, we are left with $n-c$. We then need to find the maximum segments for $n-c$. We take the maximum among all possible first cuts.

**Code:**

```python
def max_segments_recursive(n, cuts):
    # Base case: if length is 0, we have 0 segments
    if n == 0:
        return 0
    # Base case: if length is negative, this path is invalid, return a very small number
    # to ensure it's not chosen in max()
    if n < 0:
        return float('-inf')

    max_pieces = float('-inf')
    for c in cuts:
        result = max_segments_recursive(n - c, cuts)
        if result != float('-inf'): # Only consider valid paths
            max_pieces = max(max_pieces, 1 + result)
    
    return max_pieces

# Example usage:
print(f"Max Segments (5, [2, 3]) (Recursive): {max_segments_recursive(5, [2, 3])}") # Output: 2
print(f"Max Segments (7, [2, 3, 5]) (Recursive): {max_segments_recursive(7, [2, 3, 5])}") # Output: 3 (2+2+3 or 2+5)
print(f"Max Segments (4, [5]) (Recursive): {max_segments_recursive(4, [5])}") # Output: -inf (will be handled as -1)

# A small adjustment to return -1 if not possible
def max_segments_recursive_adjusted(n, cuts):
    res = max_segments_recursive(n, cuts)
    return res if res != float('-inf') else -1

print(f"Max Segments (4, [5]) (Recursive Adjusted): {max_segments_recursive_adjusted(4, [5])}") # Output: -1
```

**Time Complexity:** $O(k^n)$ where $k$ is the number of `cuts`. Exponential.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store the maximum segments for each length `i` in a memoization table.

**Code:**

```python
def max_segments_memoization(n, cuts, memo={}):
    if n == 0:
        return 0
    if n < 0:
        return float('-inf') # Use -inf to indicate an invalid path
    
    if n in memo:
        return memo[n]
    
    max_pieces = float('-inf')
    for c in cuts:
        result = max_segments_memoization(n - c, cuts, memo)
        if result != float('-inf'):
            max_pieces = max(max_pieces, 1 + result)
            
    memo[n] = max_pieces
    return max_pieces

def max_segments_memoization_adjusted(n, cuts):
    res = max_segments_memoization(n, cuts, {}) # Pass a new memo dict for each call
    return res if res != float('-inf') else -1

# Example usage:
print(f"Max Segments (5, [2, 3]) (Memoization): {max_segments_memoization_adjusted(5, [2, 3])}") # Output: 2
print(f"Max Segments (7, [2, 3, 5]) (Memoization): {max_segments_memoization_adjusted(7, [2, 3, 5])}") # Output: 3
print(f"Max Segments (4, [5]) (Memoization): {max_segments_memoization_adjusted(4, [5])}") # Output: -1
print(f"Max Segments (10, [3, 5]) (Memoization): {max_segments_memoization_adjusted(10, [3, 5])}") # Output: 4 (5+5 or 3+3+3+1(invalid))
```

**Time Complexity:** $O(n \\times k)$ where $k$ is the number of `cuts`. Each length from 1 to $n$ is computed once, and for each, we iterate through $k$ cuts.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build a `dp` array where `dp[i]` stores the maximum segments for length `i`.

**Code:**

```python
def max_segments_tabulation(n, cuts):
    # dp[i] will store the maximum number of segments for a rope of length i.
    # Initialize with -1 to signify that a length is not reachable, except for dp[0].
    dp = [-1] * (n + 1)
    dp[0] = 0 # 0 segments for a rope of length 0
    
    for i in range(1, n + 1):
        for c in cuts:
            if i - c >= 0 and dp[i - c] != -1: # Ensure previous state was reachable
                dp[i] = max(dp[i], 1 + dp[i - c])
                
    return dp[n]

# Example usage:
print(f"Max Segments (5, [2, 3]) (Tabulation): {max_segments_tabulation(5, [2, 3])}") # Output: 2
print(f"Max Segments (7, [2, 3, 5]) (Tabulation): {max_segments_tabulation(7, [2, 3, 5])}") # Output: 3
print(f"Max Segments (4, [5]) (Tabulation): {max_segments_tabulation(4, [5])}") # Output: -1
print(f"Max Segments (10, [3, 5]) (Tabulation): {max_segments_tabulation(10, [3, 5])}") # Output: 4
```

**Time Complexity:** $O(n \\times k)$ - Outer loop runs $n$ times, inner loop runs $k$ times.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

**Intuitive Insight:** Not applicable for this problem in a simple way, as the dependencies `dp[i - c]` can be arbitrary previous values depending on `c`. You'd need to keep track of all `dp[i-c]` for all `c` in `cuts`. If `cuts` are small and fixed (like 1, 2, 3 for climbing stairs), it's possible. For arbitrary `cuts`, `O(n)` space is generally required.

-----

### 8\. nth Catalan Number

**Problem Statement:**
Given an integer $n$, return the $n$-th Catalan number, denoted as $C\_n$. The Catalan numbers form a sequence of natural numbers that occur in various counting problems in combinatorics.
$C\_0 = 1$
$C\_{n+1} = \\sum\_{i=0}^{n} C\_i C\_{n-i}$ for $n \\ge 0$

**Example:**
Input: $n = 3$
Output: $5$
$C\_0 = 1$
$C\_1 = C\_0 C\_0 = 1 \\times 1 = 1$
$C\_2 = C\_0 C\_1 + C\_1 C\_0 = 1 \\times 1 + 1 \\times 1 = 2$
$C\_3 = C\_0 C\_2 + C\_1 C\_1 + C\_2 C\_0 = 1 \\times 2 + 1 \\times 1 + 2 \\times 1 = 2 + 1 + 2 = 5$

-----

#### Recursion (Brute Force)

**Intuitive Insight:** The definition itself is recursive. To find $C\_n$, we sum products of earlier Catalan numbers. This will lead to many redundant calculations.

**Code:**

```python
def catalan_recursive(n):
    if n == 0:
        return 1
    
    result = 0
    for i in range(n):
        result += catalan_recursive(i) * catalan_recursive(n - 1 - i)
    return result

# Example usage:
print(f"Catalan(3) (Recursive): {catalan_recursive(3)}") # Output: 5
print(f"Catalan(0) (Recursive): {catalan_recursive(0)}") # Output: 1
print(f"Catalan(4) (Recursive): {catalan_recursive(4)}") # Output: 14
```

**Time Complexity:** Exponential. $T(n) = \\sum\_{i=0}^{n-1} T(i)T(n-1-i)$, which is roughly $O(4^n/n^{1.5})$.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store the computed Catalan numbers to avoid recalculating them.

**Code:**

```python
def catalan_memoization(n, memo={}):
    if n == 0:
        return 1
    
    if n in memo:
        return memo[n]
    
    result = 0
    for i in range(n):
        result += catalan_memoization(i, memo) * catalan_memoization(n - 1 - i, memo)
    
    memo[n] = result
    return result

# Example usage:
print(f"Catalan(3) (Memoization): {catalan_memoization(3)}") # Output: 5
print(f"Catalan(5) (Memoization): {catalan_memoization(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Each $C\_k$ is computed once, and each computation involves a loop of size $k$. So, $\\sum\_{k=0}^{n} O(k) = O(n^2)$.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build the `dp` array from $C\_0$ up to $C\_n$.

**Code:**

```python
def catalan_tabulation(n):
    dp = [0] * (n + 1)
    dp[0] = 1 # C_0 = 1
    
    for i in range(1, n + 1): # Calculate dp[i]
        for j in range(i): # Iterate for C_j * C_{i-1-j}
            dp[i] += dp[j] * dp[i - 1 - j]
            
    return dp[n]

# Example usage:
print(f"Catalan(3) (Tabulation): {catalan_tabulation(3)}") # Output: 5
print(f"Catalan(5) (Tabulation): {catalan_tabulation(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Outer loop runs $n$ times, inner loop runs up to $n$ times.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

Not applicable in a simple way as the calculation of $C\_n$ depends on all $C\_0, \\dots, C\_{n-1}$.

-----

#### Alternatives (Mathematical Formula)

**Intuitive Insight:** There's a direct combinatorial formula for the $n$-th Catalan number.
$C\_n = \\frac{1}{n+1} \\binom{2n}{n} = \\frac{(2n)\!}{(n+1)\! n\!}$

**Code:**

```python
import math

def nCr(n_val, r_val):
    # Calculates n choose r
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    return res

def catalan_formula(n):
    if n < 0:
        return 0
    # C_n = (1 / (n + 1)) * nCr(2 * n, n)
    return nCr(2 * n, n) // (n + 1)

# Example usage:
print(f"Catalan(3) (Formula): {catalan_formula(3)}") # Output: 5
print(f"Catalan(5) (Formula): {catalan_formula(5)}") # Output: 42
```

**Time Complexity:** $O(n)$ - Due to the binomial coefficient calculation, which involves a loop of $n$ iterations.
**Space Complexity:** $O(1)$

-----

### 9\. Count Unique BSTs

**Problem Statement:**
Given an integer $n$, return the number of structurally unique Binary Search Trees (BSTs) that can be formed using $n$ nodes with values from 1 to $n$.

**Example:**
Input: $n = 3$
Output: $5$
Explanation: For $n=3$, the 5 unique BSTs are:

1.  1-null-2-null-3
2.  1-null-3-2-null
3.  2-1-3
4.  3-1-null-null-2
5.  3-2-null-1-null

-----

**Intuitive Insight:** This problem is a classic application of Catalan numbers. If we choose `i` as the root of the BST, then `i-1` nodes are available for the left subtree (values 1 to `i-1`), and `n-i` nodes are available for the right subtree (values `i+1` to `n`). The number of unique BSTs with `n` nodes is the sum of (number of BSTs for left subtree) \* (number of BSTs for right subtree) over all possible choices for the root.
Let $G(n)$ be the number of unique BSTs for $n$ nodes.
$G(0) = 1$ (empty tree)
$G(n) = \\sum\_{i=1}^{n} G(i-1) \\times G(n-i)$ for $n \\ge 1$
This recurrence relation is identical to the Catalan numbers ($C\_n = \\sum\_{j=0}^{n-1} C\_j C\_{n-1-j}$). If we map $j = i-1$, then $C\_{n-1-j} = C\_{n-1-(i-1)} = C\_{n-i}$. So $G(n) = C\_n$.

-----

#### Recursion (Brute Force)

**Code:**

```python
def num_unique_bsts_recursive(n):
    if n == 0:
        return 1
    
    count = 0
    for i in range(1, n + 1): # i is the root value
        left_subtrees = num_unique_bsts_recursive(i - 1)
        right_subtrees = num_unique_bsts_recursive(n - i)
        count += left_subtrees * right_subtrees
    return count

# Example usage:
print(f"Unique BSTs (3) (Recursive): {num_unique_bsts_recursive(3)}") # Output: 5
print(f"Unique BSTs (4) (Recursive): {num_unique_bsts_recursive(4)}") # Output: 14
```

**Time Complexity:** Exponential, same as Catalan numbers.
**Space Complexity:** $O(n)$ - Recursion stack.

-----

#### Memoization (Top-Down)

**Code:**

```python
def num_unique_bsts_memoization(n, memo={}):
    if n == 0:
        return 1
    
    if n in memo:
        return memo[n]
    
    count = 0
    for i in range(1, n + 1):
        left_subtrees = num_unique_bsts_memoization(i - 1, memo)
        right_subtrees = num_unique_bsts_memoization(n - i, memo)
        count += left_subtrees * right_subtrees
        
    memo[n] = count
    return count

# Example usage:
print(f"Unique BSTs (3) (Memoization): {num_unique_bsts_memoization(3)}") # Output: 5
print(f"Unique BSTs (5) (Memoization): {num_unique_bsts_memoization(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Same as Catalan numbers.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Code:**

```python
def num_unique_bsts_tabulation(n):
    dp = [0] * (n + 1)
    dp[0] = 1 # For 0 nodes, there is 1 unique BST (empty tree)
    
    for i in range(1, n + 1): # Calculate dp[i] for i nodes
        for j in range(i): # j nodes for left subtree, (i-1-j) for right subtree
            dp[i] += dp[j] * dp[i - 1 - j]
            
    return dp[n]

# Example usage:
print(f"Unique BSTs (3) (Tabulation): {num_unique_bsts_tabulation(3)}") # Output: 5
print(f"Unique BSTs (5) (Tabulation): {num_unique_bsts_tabulation(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Same as Catalan numbers.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

Not applicable for the same reasons as Catalan numbers.

-----

#### Alternatives (Mathematical Formula - Catalan Number)

Since it's exactly the $n$-th Catalan number, the direct formula can be used.

**Code:**

```python
import math

def nCr(n_val, r_val):
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    return res

def num_unique_bsts_formula(n):
    if n < 0:
        return 0
    return nCr(2 * n, n) // (n + 1)

# Example usage:
print(f"Unique BSTs (3) (Formula): {num_unique_bsts_formula(3)}") # Output: 5
print(f"Unique BSTs (5) (Formula): {num_unique_bsts_formula(5)}") # Output: 42
```

**Time Complexity:** $O(n)$
**Space Complexity:** $O(1)$

-----

### 10\. Count Valid Parenthesis

**Problem Statement:**
Given an integer $n$, return the number of well-formed parenthesis strings of length $2n$. A well-formed parenthesis string means that for every opening parenthesis, there is a corresponding closing parenthesis, and the parentheses are properly nested.

**Example:**
Input: $n = 3$ (length 6)
Output: $5$
Explanation:
"((()))"
"(()())"
"(())()"
"()(())"
"()()()"

-----

**Intuitive Insight:** This is another classic problem whose solution is given by the Catalan numbers. Consider a valid parenthesis sequence. The first character must be '('. This '(' must be matched by some ')'. Suppose it matches the $k$-th character. Then the prefix up to $k$ must be a valid parenthesis sequence, and the remaining characters must also form a valid parenthesis sequence.
If we map '(' to an "up-step" and ')' to a "down-step" on a grid, a valid sequence is a path from (0,0) to (2n,0) that never goes below the x-axis. This corresponds to the properties counted by Catalan numbers.

Let $f(n)$ be the number of valid parenthesis strings of length $2n$.
To form a valid string of length $2n$, we must start with '(' and end with ')'. The content inside the first '(' and last ')' forms a valid parenthesis string of length $2(n-1)$.
Alternatively, we can iterate through the position where the first '(' is closed. If the first '(' is matched by the $i$-th ')', then the substring between them (from index 1 to $i-1$) must be a valid parenthesis sequence, and the substring after the $i$-th ')' (from index $i+1$ to $2n$) must also be a valid parenthesis sequence.
Let $i$ be the number of pairs in the first segment. Then $i$ can range from $0$ to $n-1$.
The first pair encloses $i$ pairs, leaving $n-1-i$ pairs for the rest.
$f(n) = \\sum\_{i=0}^{n-1} f(i) \\times f(n-1-i)$
This is exactly the recurrence for Catalan numbers where $f(0) = 1$ (empty string). So $f(n) = C\_n$.

-----

#### Recursion (Brute Force)

**Code:**

```python
def count_valid_parenthesis_recursive(n):
    if n == 0:
        return 1
    
    count = 0
    for i in range(n): # Iterate through i pairs inside the first outer pair
        count += count_valid_parenthesis_recursive(i) * count_valid_parenthesis_recursive(n - 1 - i)
    return count

# Example usage:
print(f"Valid Parenthesis (3) (Recursive): {count_valid_parenthesis_recursive(3)}") # Output: 5
print(f"Valid Parenthesis (4) (Recursive): {count_valid_parenthesis_recursive(4)}") # Output: 14
```

**Time Complexity:** Exponential, same as Catalan numbers.
**Space Complexity:** $O(n)$ - Recursion stack.

-----

#### Memoization (Top-Down)

**Code:**

```python
def count_valid_parenthesis_memoization(n, memo={}):
    if n == 0:
        return 1
    
    if n in memo:
        return memo[n]
    
    count = 0
    for i in range(n):
        count += count_valid_parenthesis_memoization(i, memo) * count_valid_parenthesis_memoization(n - 1 - i, memo)
        
    memo[n] = count
    return count

# Example usage:
print(f"Valid Parenthesis (3) (Memoization): {count_valid_parenthesis_memoization(3)}") # Output: 5
print(f"Valid Parenthesis (5) (Memoization): {count_valid_parenthesis_memoization(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Same as Catalan numbers.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Code:**

```python
def count_valid_parenthesis_tabulation(n):
    dp = [0] * (n + 1)
    dp[0] = 1 # For 0 pairs, there is 1 empty string ""
    
    for i in range(1, n + 1): # Calculate dp[i] (for i pairs)
        for j in range(i): # Iterate for j pairs in the first part, (i-1-j) in the second
            dp[i] += dp[j] * dp[i - 1 - j]
            
    return dp[n]

# Example usage:
print(f"Valid Parenthesis (3) (Tabulation): {count_valid_parenthesis_tabulation(3)}") # Output: 5
print(f"Valid Parenthesis (5) (Tabulation): {count_valid_parenthesis_tabulation(5)}") # Output: 42
```

**Time Complexity:** $O(n^2)$ - Same as Catalan numbers.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

Not applicable for the same reasons as Catalan numbers.

-----

#### Alternatives (Mathematical Formula - Catalan Number)

**Code:**

```python
import math

def nCr(n_val, r_val):
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    return res

def count_valid_parenthesis_formula(n):
    if n < 0:
        return 0
    return nCr(2 * n, n) // (n + 1)

# Example usage:
print(f"Valid Parenthesis (3) (Formula): {count_valid_parenthesis_formula(3)}") # Output: 5
print(f"Valid Parenthesis (5) (Formula): {count_valid_parenthesis_formula(5)}") # Output: 42
```

**Time Complexity:** $O(n)$
**Space Complexity:** $O(1)$

-----

### 11\. Ways to Triangulate a Polygon

**Problem Statement:**
Given $n$ points forming a convex polygon, find the number of ways to triangulate the polygon. A triangulation of a convex polygon is a decomposition of the polygon into non-overlapping triangles by a set of non-intersecting diagonals.

**Example:**
Input: $n = 4$ (a quadrilateral)
Output: $2$
Explanation: A square can be triangulated in two ways by drawing one diagonal.

-----

**Intuitive Insight:** This is another problem whose solution is given by the Catalan numbers.
Consider a convex polygon with $n$ vertices. Fix one edge, say $(V\_0, V\_{n-1})$. Any diagonal chosen to form the first triangle must use this edge and one other vertex, say $V\_k$. This forms a triangle $(V\_0, V\_k, V\_{n-1})$. This triangle divides the polygon into two smaller polygons: one with vertices $V\_0, \\dots, V\_k$ and another with vertices $V\_k, \\dots, V\_{n-1}$.
The number of vertices in the first polygon is $k+1$, and in the second is $n-k$.
The number of ways to triangulate a polygon with $m$ vertices is $C\_{m-2}$ (for $m \\ge 2$).
So for $n$ vertices, the problem reduces to $C\_{n-2}$.
Let $f(n)$ be the number of triangulations for an $n$-gon.
Base cases:
$f(2) = 1$ (A 2-gon is just a line segment, considered 1 way, for formula consistency $C\_0$)
$f(3) = 1$ (A triangle, 1 way, $C\_1$)
For $n \\ge 3$: Pick an edge $(P\_0, P\_{n-1})$. Consider all possible third vertices $P\_k$ for a triangle $(P\_0, P\_k, P\_{n-1})$. This splits the original $n$-gon into two sub-polygons (one with $k+1$ vertices and another with $n-k$ vertices, including $P\_0$ and $P\_{n-1}$ for overlap).
Sum over $k$ from $1$ to $n-2$: $f(k+1) \\times f(n-k)$.
This gives a recurrence relation which is equivalent to Catalan numbers shifted.
Number of triangulations for an $n$-gon is $C\_{n-2}$.

-----

#### Recursion (Brute Force)

**Code:**

```python
def ways_to_triangulate_recursive(n):
    # Base cases: A polygon with 2 vertices (a line) has 1 way (C0)
    # A polygon with 3 vertices (a triangle) has 1 way (C1)
    if n <= 2:
        return 1
    
    # We are calculating C_{n-2}.
    # The recurrence for Catalan numbers (C_k) is sum(C_i * C_{k-1-i})
    # Here k = n-2.
    # So we need sum(C_i * C_{n-2-1-i}) = sum(C_i * C_{n-3-i})
    # Which corresponds to splitting a polygon of (n-1) vertices.
    # The direct mapping for n-gon is C_{n-2}.
    # Let's write the recursive function for C_k directly then call for C_{n-2}.
    
    def catalan_recursive_helper(k):
        if k == 0:
            return 1
        
        result = 0
        for i in range(k):
            result += catalan_recursive_helper(i) * catalan_recursive_helper(k - 1 - i)
        return result

    if n < 3: # For n=1, n=2, it doesn't really make sense geometrically for "polygon".
              # But if we consider C_0 = 1 for 2 vertices and C_1 = 1 for 3 vertices,
              # it aligns with the formula C_{n-2}.
        return 1
    
    return catalan_recursive_helper(n - 2)

# Example usage:
print(f"Ways to Triangulate Polygon (4) (Recursive): {ways_to_triangulate_recursive(4)}") # Output: 2 (C_2)
print(f"Ways to Triangulate Polygon (5) (Recursive): {ways_to_triangulate_recursive(5)}") # Output: 5 (C_3)
```

**Time Complexity:** Exponential, same as Catalan numbers.
**Space Complexity:** $O(n)$ - Recursion stack.

-----

#### Memoization (Top-Down)

**Code:**

```python
def ways_to_triangulate_memoization(n):
    if n < 3:
        return 1
    
    memo = {}
    def catalan_memoization_helper(k):
        if k == 0:
            return 1
        if k in memo:
            return memo[k]
        
        result = 0
        for i in range(k):
            result += catalan_memoization_helper(i) * catalan_memoization_helper(k - 1 - i)
        
        memo[k] = result
        return result
        
    return catalan_memoization_helper(n - 2)

# Example usage:
print(f"Ways to Triangulate Polygon (4) (Memoization): {ways_to_triangulate_memoization(4)}") # Output: 2
print(f"Ways to Triangulate Polygon (5) (Memoization): {ways_to_triangulate_memoization(5)}") # Output: 5
```

**Time Complexity:** $O(n^2)$ - Same as Catalan numbers.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Code:**

```python
def ways_to_triangulate_tabulation(n):
    if n < 3:
        return 1
        
    # We need to calculate C_{n-2}
    k_val = n - 2
    
    dp = [0] * (k_val + 1)
    dp[0] = 1 # C_0
    
    for i in range(1, k_val + 1): # Calculate C_i
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
            
    return dp[k_val]

# Example usage:
print(f"Ways to Triangulate Polygon (4) (Tabulation): {ways_to_triangulate_tabulation(4)}") # Output: 2
print(f"Ways to Triangulate Polygon (5) (Tabulation): {ways_to_triangulate_tabulation(5)}") # Output: 5
```

**Time Complexity:** $O(n^2)$ - Since we are calculating $C\_{n-2}$, the loop runs $n-2$ times, each taking $O(n)$ time.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

Not applicable for the same reasons as Catalan numbers.

-----

#### Alternatives (Mathematical Formula - Catalan Number)

**Code:**

```python
import math

def nCr(n_val, r_val):
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    return res

def ways_to_triangulate_formula(n):
    if n < 3: # A polygon needs at least 3 vertices to be a triangle, 
              # but for consistency with C_0=1 for n=2, we return 1.
        return 1
    # Number of triangulations for an n-gon is C_{n-2}
    return nCr(2 * (n - 2), n - 2) // ((n - 2) + 1)

# Example usage:
print(f"Ways to Triangulate Polygon (4) (Formula): {ways_to_triangulate_formula(4)}") # Output: 2
print(f"Ways to Triangulate Polygon (5) (Formula): {ways_to_triangulate_formula(5)}") # Output: 5
```

**Time Complexity:** $O(n)$
**Space Complexity:** $O(1)$

-----

### 12\. Min Sum in a Triangle

**Problem Statement:**
Given a triangle array, find the minimum path sum from top to bottom. Each step you may move to an adjacent number on the row below. That is, if you are on index $i$ on the current row, you may move to either index $i$ or index $i+1$ on the next row.

**Example:**
Input:

```
triangle = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

Output: $11$
Explanation: The minimum path sum from top to bottom is 11 (i.e., $2 + 3 + 5 + 1 = 11$).

-----

#### Recursion (Brute Force)

**Intuitive Insight:** From a given position `(row, col)`, we can move to `(row+1, col)` or `(row+1, col+1)`. We want to find the minimum sum, so we recursively explore both paths and choose the minimum.

**Code:**

```python
def min_sum_triangle_recursive(triangle):
    rows = len(triangle)

    def solve(row, col):
        # Base case: if we are at the last row, return the value at that position
        if row == rows - 1:
            return triangle[row][col]
        
        # Recursive step: current value + min(path from (row+1, col), path from (row+1, col+1))
        current_val = triangle[row][col]
        
        path1 = solve(row + 1, col)
        path2 = solve(row + 1, col + 1)
        
        return current_val + min(path1, path2)

    return solve(0, 0) # Start from the top (0, 0)

# Example usage:
triangle1 = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
print(f"Min Sum in Triangle (Recursive): {min_sum_triangle_recursive(triangle1)}") # Output: 11

triangle2 = [[-10]]
print(f"Min Sum in Triangle (Recursive): {min_sum_triangle_recursive(triangle2)}") # Output: -10
```

**Time Complexity:** $O(2^n)$ where $n$ is the number of rows. Each path branches into two.
**Space Complexity:** $O(n)$ - Recursion stack depth (max `n` calls).

-----

#### Memoization (Top-Down)

**Intuitive Insight:** We observe overlapping subproblems. Store the minimum sum to reach `(row, col)`.

**Code:**

```python
def min_sum_triangle_memoization(triangle):
    rows = len(triangle)
    memo = {} # Dictionary to store results: (row, col) -> min_sum

    def solve(row, col):
        if row == rows - 1:
            return triangle[row][col]
        
        if (row, col) in memo:
            return memo[(row, col)]
            
        current_val = triangle[row][col]
        
        path1 = solve(row + 1, col)
        path2 = solve(row + 1, col + 1)
        
        memo[(row, col)] = current_val + min(path1, path2)
        return memo[(row, col)]

    return solve(0, 0)

# Example usage:
triangle1 = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
print(f"Min Sum in Triangle (Memoization): {min_sum_triangle_memoization(triangle1)}") # Output: 11
```

**Time Complexity:** $O(n^2)$ where $n$ is the number of rows. There are $O(n^2)$ states `(row, col)`, and each state is computed once.
**Space Complexity:** $O(n^2)$ for memoization table and $O(n)$ for recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** We can build the solution from the bottom up. Start from the second-to-last row. For each element, calculate the minimum path sum by adding its value to the minimum of its two possible children in the row below. Propagate this up to the top.

**Code:**

```python
def min_sum_triangle_tabulation(triangle):
    rows = len(triangle)
    
    # Create a copy of the triangle to store DP values, or use the triangle itself.
    # Using a copy is safer if the original triangle needs to be preserved.
    # We can also modify in-place if allowed.
    dp = [row[:] for row in triangle] # Deep copy
    
    # Start from the second to last row and go up to the top row (index 0)
    for r in range(rows - 2, -1, -1): # From rows-2 down to 0
        for c in range(len(dp[r])): # Iterate through columns in the current row
            # The current element's minimum path sum is its value +
            # the minimum of the two elements below it in the next row.
            dp[r][c] += min(dp[r + 1][c], dp[r + 1][c + 1])
            
    return dp[0][0] # The minimum sum will be at the top of the modified triangle

# Example usage:
triangle1 = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
print(f"Min Sum in Triangle (Tabulation): {min_sum_triangle_tabulation(triangle1)}") # Output: 11
```

**Time Complexity:** $O(n^2)$ - Two nested loops, where $n$ is the number of rows.
**Space Complexity:** $O(n^2)$ - For the `dp` table (copy of the triangle).

-----

#### Optimized Space

**Intuitive Insight:** Notice that to compute values for row `r`, we only need values from row `r+1`. This means we can optimize space to $O(n)$ by only storing the values of the previous row.

**Code:**

```python
def min_sum_triangle_optimized_space(triangle):
    rows = len(triangle)
    
    # Start with the last row (it is its own min path sum)
    # This `dp` array will represent the row we are currently processing.
    # Initialize `dp` with the last row of the triangle.
    dp = triangle[rows - 1] 
    
    # Iterate from the second to last row upwards
    for r in range(rows - 2, -1, -1):
        # Create a new current row dp array (or modify in place if careful)
        # For each element in the current row `r`, calculate its min path sum
        # based on the `dp` array (which holds sums for row `r+1`).
        for c in range(len(triangle[r])):
            dp[c] = triangle[r][c] + min(dp[c], dp[c + 1])
            
    return dp[0] # The minimum path sum from the top

# Example usage:
triangle1 = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
print(f"Min Sum in Triangle (Optimized Space): {min_sum_triangle_optimized_space(triangle1)}") # Output: 11

triangle3 = [[-1], [2, 3], [1, -1, -3]]
print(f"Min Sum in Triangle (Optimized Space): {min_sum_triangle_optimized_space(triangle3)}") # Output: -1 + 3 + (-3) = -1. (-1 + 2 + (-1) = 0). The path -1 -> 3 -> -3 is better.
```

**Time Complexity:** $O(n^2)$ - Two nested loops.
**Space Complexity:** $O(n)$ - For the `dp` array (size of the last row).

-----

### 13\. Minimum Perfect Squares

**Problem Statement:**
Given an integer $n$, return the least number of perfect square numbers that sum to $n$. A perfect square is an integer that is the square of an integer; for example, 1, 4, 9, and 16 are perfect squares.

**Example:**
Input: $n = 12$
Output: $3$
Explanation: $12 = 4 + 4 + 4$.

Input: $n = 13$
Output: $2$
Explanation: $13 = 4 + 9$.

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To find the minimum number of perfect squares that sum to $n$, we can try subtracting every possible perfect square less than or equal to $n$. For each perfect square `s*s`, if we use `s*s`, then we need to find the minimum perfect squares for `n - s*s`. We take the minimum among all such possibilities.

**Code:**

```python
import math

def num_squares_recursive(n):
    # Base case: if n is 0, we need 0 perfect squares
    if n == 0:
        return 0
    # Base case: if n is negative, this path is invalid, return a very large number
    if n < 0:
        return float('inf')
    
    min_count = float('inf')
    
    # Iterate through all possible perfect squares less than or equal to n
    for i in range(1, int(math.sqrt(n)) + 1):
        square = i * i
        min_count = min(min_count, 1 + num_squares_recursive(n - square))
        
    return min_count

# Example usage:
print(f"Min Perfect Squares (12) (Recursive): {num_squares_recursive(12)}") # Output: 3
print(f"Min Perfect Squares (13) (Recursive): {num_squares_recursive(13)}") # Output: 2
# print(f"Min Perfect Squares (4) (Recursive): {num_squares_recursive(4)}") # Output: 1
```

**Time Complexity:** Exponential. Roughly $O(n^{\\sqrt{n}})$ in worst case, as it explores many redundant subproblems.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store the minimum number of perfect squares for each `i` from 1 to $n$.

**Code:**

```python
import math

def num_squares_memoization(n, memo={}):
    if n == 0:
        return 0
    if n < 0:
        return float('inf')
    
    if n in memo:
        return memo[n]
    
    min_count = float('inf')
    for i in range(1, int(math.sqrt(n)) + 1):
        square = i * i
        min_count = min(min_count, 1 + num_squares_memoization(n - square, memo))
            
    memo[n] = min_count
    return min_count

# Example usage:
print(f"Min Perfect Squares (12) (Memoization): {num_squares_memoization(12)}") # Output: 3
print(f"Min Perfect Squares (13) (Memoization): {num_squares_memoization(13)}") # Output: 2
print(f"Min Perfect Squares (4) (Memoization): {num_squares_memoization(4)}") # Output: 1
```

**Time Complexity:** $O(n \\sqrt{n})$. Each `memo[i]` is computed once, and each computation involves a loop of size $\\sqrt{i}$.
**Space Complexity:** $O(n)$ - Memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build a `dp` array where `dp[i]` stores the least number of perfect squares that sum to `i`.

**Code:**

```python
import math

def num_squares_tabulation(n):
    # dp[i] will store the least number of perfect square numbers that sum to i.
    # Initialize with a large value (or n+1, since max squares for n is n, e.g., 1+1+...+1)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0 # 0 perfect squares sum to 0
    
    # Precompute perfect squares up to n
    perfect_squares = []
    for i in range(1, int(math.sqrt(n)) + 1):
        perfect_squares.append(i * i)
        
    for i in range(1, n + 1):
        for square in perfect_squares:
            if i - square >= 0:
                dp[i] = min(dp[i], 1 + dp[i - square])
                
    return dp[n]

# Example usage:
print(f"Min Perfect Squares (12) (Tabulation): {num_squares_tabulation(12)}") # Output: 3
print(f"Min Perfect Squares (13) (Tabulation): {num_squares_tabulation(13)}") # Output: 2
print(f"Min Perfect Squares (4) (Tabulation): {num_squares_tabulation(4)}") # Output: 1
```

**Time Complexity:** $O(n \\sqrt{n})$ - Outer loop runs $n$ times, inner loop runs $\\sqrt{n}$ times.
**Space Complexity:** $O(n)$ - For the `dp` array.

-----

#### Optimized Space

Not applicable for this problem. The calculation for `dp[i]` depends on `dp[i - square]` where `square` can be arbitrary, so we need access to all previous `dp` values.

-----

#### Alternatives (Lagrange's Four-Square Theorem, BFS)

**Lagrange's Four-Square Theorem:**
This theorem states that every natural number can be represented as the sum of four integer squares. So the answer will always be 1, 2, 3, or 4.

**BFS Approach:**
We can model this as a shortest path problem on a graph. Each number from 0 to $n$ is a node. An edge exists from $u$ to $v$ if $v = u + s^2$ for some perfect square $s^2$. We want to find the shortest path from $0$ to $n$. Or, more directly, from $n$ to $0$.

Start with $n$. At each step, subtract a perfect square. The "level" in BFS represents the number of perfect squares used. The first time we reach 0, that level is the minimum count.

**Code (BFS):**

```python
import collections
import math

def num_squares_bfs(n):
    if n <= 0:
        return 0

    q = collections.deque([(n, 0)]) # (remaining_value, num_squares_used)
    visited = {n} # Keep track of visited remaining_values

    while q:
        current_n, num_squares = q.popleft()

        # Generate next states by subtracting perfect squares
        for i in range(1, int(math.sqrt(current_n)) + 1):
            square = i * i
            next_n = current_n - square
            
            if next_n == 0:
                return num_squares + 1
            
            if next_n > 0 and next_n not in visited:
                visited.add(next_n)
                q.append((next_n, num_squares + 1))
                
    return -1 # Should not happen for positive n due to Lagrange's theorem

# Example usage:
print(f"Min Perfect Squares (12) (BFS): {num_squares_bfs(12)}") # Output: 3
print(f"Min Perfect Squares (13) (BFS): {num_squares_bfs(13)}") # Output: 2
```

**Time Complexity:** $O(N \\cdot \\sqrt{N})$ in the worst case, as each number from $N$ down to $0$ might be added to the queue, and for each, we try $\\sqrt{N}$ perfect squares. However, due to Lagrange's theorem, the depth of the BFS is at most 4, making it very efficient in practice.
**Space Complexity:** $O(N)$ for the queue and visited set.

-----

### 14\. Ways to Partition a Set

**Problem Statement:**
Given a set of $n$ elements, find the number of ways to partition it into non-empty subsets. This is known as the Bell number $B\_n$.
$B\_0 = 1$ (empty set has one partition: the empty partition)
$B\_n = \\sum\_{k=0}^{n-1} \\binom{n-1}{k} B\_k$ for $n \\ge 1$

**Example:**
Input: $n = 3$ (Set {1, 2, 3})
Output: $5$
Explanation:

1.  {{1}, {2}, {3}}
2.  {{1, 2}, {3}}
3.  {{1, 3}, {2}}
4.  {{2, 3}, {1}}
5.  {{1, 2, 3}}

-----

#### Recursion (Brute Force)

**Intuitive Insight:** To find $B\_n$, we pick an element (say, element $n$). This element can be in a subset by itself, or it can be grouped with one or more other elements.
Consider element $n$.

1.  It forms a singleton set: ${n}$. The remaining $n-1$ elements can be partitioned in $B\_{n-1}$ ways.
2.  It is grouped with $k$ other elements (from the remaining $n-1$ elements). There are $\\binom{n-1}{k}$ ways to choose these $k$ elements. The remaining $n-1-k$ elements can be partitioned in $B\_{n-1-k}$ ways.
    This leads to the recurrence: $B\_n = \\sum\_{k=0}^{n-1} \\binom{n-1}{k} B\_k$.

**Code:**

```python
import math

def nCr(n_val, r_val):
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    return res

def bell_recursive(n):
    if n == 0:
        return 1
    
    result = 0
    # Sum over k, where k is the number of elements *not* with the nth element
    # Recurrence: B_n = sum_{k=0 to n-1} C(n-1, k) * B_k
    for k in range(n): # k is the size of the remaining set to partition, after the group containing element 'n' is formed.
                       # This means k is the index for B_k.
                       # And we choose (n-1-k) elements to be with element 'n'.
        result += nCr(n - 1, k) * bell_recursive(k)
    return result

# Example usage:
print(f"Ways to Partition Set (3) (Recursive): {bell_recursive(3)}") # Output: 5
print(f"Ways to Partition Set (4) (Recursive): {bell_recursive(4)}") # Output: 15
```

**Time Complexity:** Highly exponential due to repeated binomial coefficient calculations and subproblems.
**Space Complexity:** $O(n)$ - Recursion stack.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store computed Bell numbers. Precompute binomial coefficients or memoize them too.

**Code:**

```python
import math

# Memoization for nCr (optional, but good for performance)
nCr_memo = {}
def nCr_memoized(n_val, r_val):
    if r_val < 0 or r_val > n_val:
        return 0
    if r_val == 0 or r_val == n_val:
        return 1
    if r_val > n_val // 2:
        r_val = n_val - r_val
    
    if (n_val, r_val) in nCr_memo:
        return nCr_memo[(n_val, r_val)]
    
    res = 1
    for i in range(r_val):
        res = res * (n_val - i) // (i + 1)
    nCr_memo[(n_val, r_val)] = res
    return res

bell_memo = {}
def bell_memoization(n):
    if n == 0:
        return 1
    
    if n in bell_memo:
        return bell_memo[n]
    
    result = 0
    for k in range(n):
        result += nCr_memoized(n - 1, k) * bell_memoization(k)
    
    bell_memo[n] = result
    return result

# Example usage:
print(f"Ways to Partition Set (3) (Memoization): {bell_memoization(3)}") # Output: 5
print(f"Ways to Partition Set (4) (Memoization): {bell_memoization(4)}") # Output: 15
print(f"Ways to Partition Set (5) (Memoization): {bell_memoization(5)}") # Output: 52
```

**Time Complexity:** $O(n^2)$ - Each $B\_i$ is computed once. To compute $B\_i$, we sum $i$ terms. Each term involves an $nCr$ calculation which is $O(i)$. Total: $\\sum\_{i=1}^{n} (O(i) \\times O(i))$ if $nCr$ is not memoized, or $\\sum\_{i=1}^{n} O(i)$ if $nCr$ is $O(1)$ lookup. So it's $O(n^2)$ if $nCr$ is precomputed/memoized efficiently.
**Space Complexity:** $O(n^2)$ for `nCr_memo` table and $O(n)$ for `bell_memo` and recursion stack. ($O(n^2)$ is dominant).

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build a `dp` array for Bell numbers. We can also precompute Pascal's Triangle for binomial coefficients or calculate them on the fly.

**Code:**

```python
def bell_tabulation(n):
    bell = [0] * (n + 1)
    bell[0] = 1 # B_0 = 1
    
    # Precompute Pascal's Triangle for binomial coefficients (nCr)
    # pascal[i][j] = C(i, j)
    pascal = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        pascal[i][0] = 1
        for j in range(1, i + 1):
            pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j]
            
    for i in range(1, n + 1): # Calculate B_i
        current_bell = 0
        for k in range(i): # Sum C(i-1, k) * B_k
            current_bell += pascal[i - 1][k] * bell[k]
        bell[i] = current_bell
            
    return bell[n]

# Example usage:
print(f"Ways to Partition Set (3) (Tabulation): {bell_tabulation(3)}") # Output: 5
print(f"Ways to Partition Set (4) (Tabulation): {bell_tabulation(4)}") # Output: 15
print(f"Ways to Partition Set (5) (Tabulation): {bell_tabulation(5)}") # Output: 52
```

**Time Complexity:** $O(n^2)$ for Pascal's triangle generation + $O(n^2)$ for filling `bell` array. Total: $O(n^2)$.
**Space Complexity:** $O(n^2)$ for Pascal's triangle table + $O(n)$ for `bell` array. Total: $O(n^2)$.

-----

#### Optimized Space

The space complexity for Bell numbers can be optimized to $O(n)$ using a different recurrence relation based on Stirling numbers of the second kind, or by optimizing Pascal's Triangle to just store the current row.

A different recurrence for Bell numbers:
$B\_{n+1} = \\sum\_{k=0}^n \\binom{n}{k} B\_k$ (similar to the one used)
This means $B\_n$ depends on all previous $B\_k$. So $O(N)$ space for $B$ array is generally needed.

**Alternative (Stirling Numbers of the Second Kind):**
$B\_n = \\sum\_{k=0}^{n} S(n, k)$, where $S(n, k)$ is the Stirling number of the second kind (number of ways to partition a set of $n$ elements into $k$ non-empty subsets).
$S(n, k) = S(n-1, k-1) + k \\times S(n-1, k)$
Base cases: $S(n, 0) = 0$ for $n \\ge 1$, $S(n, n) = 1$, $S(0, 0) = 1$.

```python
def bell_optimized_space_stirling(n):
    if n == 0:
        return 1
    
    # dp[j] will store S(i, j)
    # We only need the previous row (i-1) to calculate the current row (i)
    # The `current_row_stirling` will hold S(i, k)
    # The `prev_row_stirling` will hold S(i-1, k)
    
    # Initialize S(0,0) = 1
    prev_row_stirling = [0] * (n + 1)
    prev_row_stirling[0] = 1 
    
    for i in range(1, n + 1): # i is the number of elements
        current_row_stirling = [0] * (n + 1)
        # S(i, 0) = 0 for i >= 1
        for k in range(1, i + 1): # k is the number of subsets
            current_row_stirling[k] = prev_row_stirling[k - 1] + k * prev_row_stirling[k]
        prev_row_stirling = current_row_stirling # Update for next iteration
        
    return sum(prev_row_stirling) # Sum S(n, k) for k from 0 to n

# Example usage:
print(f"Ways to Partition Set (3) (Optimized Space - Stirling): {bell_optimized_space_stirling(3)}") # Output: 5
print(f"Ways to Partition Set (4) (Optimized Space - Stirling): {bell_optimized_space_stirling(4)}") # Output: 15
print(f"Ways to Partition Set (5) (Optimized Space - Stirling): {bell_optimized_space_stirling(5)}") # Output: 52
```

**Time Complexity:** $O(n^2)$ - Two nested loops.
**Space Complexity:** $O(n)$ - Two lists of size $n$.

-----

### 15\. Binomial Coefficient $\\binom{n}{k}$

**Problem Statement:**
Given two non-negative integers $n$ and $k$, where $0 \\le k \\le n$, return the binomial coefficient $\\binom{n}{k}$, which represents the number of ways to choose $k$ items from a set of $n$ distinct items.

**Example:**
Input: $n = 5, k = 2$
Output: $10$
Explanation: $\\binom{5}{2} = \\frac{5\!}{2\!(5-2)\!} = \\frac{5 \\times 4}{2 \\times 1} = 10$.

-----

#### Recursion (Brute Force)

**Intuitive Insight:** Pascal's identity: $\\binom{n}{k} = \\binom{n-1}{k-1} + \\binom{n-1}{k}$.
Base cases: $\\binom{n}{0} = 1$, $\\binom{n}{n} = 1$, $\\binom{n}{k} = 0$ if $k \< 0$ or $k \> n$.

**Code:**

```python
def binomial_coefficient_recursive(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    return binomial_coefficient_recursive(n - 1, k - 1) + binomial_coefficient_recursive(n - 1, k)

# Example usage:
print(f"Binomial Coefficient (5, 2) (Recursive): {binomial_coefficient_recursive(5, 2)}") # Output: 10
print(f"Binomial Coefficient (4, 0) (Recursive): {binomial_coefficient_recursive(4, 0)}") # Output: 1
print(f"Binomial Coefficient (4, 4) (Recursive): {binomial_coefficient_recursive(4, 4)}") # Output: 1
```

**Time Complexity:** Exponential, $O(2^n)$ in the worst case (e.g., $k \\approx n/2$). It resembles Fibonacci.
**Space Complexity:** $O(n)$ - Recursion stack depth.

-----

#### Memoization (Top-Down)

**Intuitive Insight:** Store the computed binomial coefficients in a 2D memoization table.

**Code:**

```python
def binomial_coefficient_memoization(n, k, memo={}):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    if (n, k) in memo:
        return memo[(n, k)]
        
    memo[(n, k)] = binomial_coefficient_memoization(n - 1, k - 1, memo) + \
                    binomial_coefficient_memoization(n - 1, k, memo)
    return memo[(n, k)]

# Example usage:
print(f"Binomial Coefficient (5, 2) (Memoization): {binomial_coefficient_memoization(5, 2)}") # Output: 10
print(f"Binomial Coefficient (10, 5) (Memoization): {binomial_coefficient_memoization(10, 5)}") # Output: 252
```

**Time Complexity:** $O(n \\times k)$ - Each state `(i, j)` is computed once.
**Space Complexity:** $O(n \\times k)$ - For the memoization table and recursion stack.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Build Pascal's Triangle row by row. `dp[i][j]` will store $\\binom{i}{j}$.

**Code:**

```python
def binomial_coefficient_tabulation(n, k):
    # dp[i][j] stores C(i, j)
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    
    # Base cases
    for i in range(n + 1):
        dp[i][0] = 1 # C(i, 0) = 1
    
    # Fill the table using Pascal's identity
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            if j <= i: # C(i, j) is only defined for j <= i
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                
    return dp[n][k]

# Example usage:
print(f"Binomial Coefficient (5, 2) (Tabulation): {binomial_coefficient_tabulation(5, 2)}") # Output: 10
print(f"Binomial Coefficient (10, 5) (Tabulation): {binomial_coefficient_tabulation(10, 5)}") # Output: 252
```

**Time Complexity:** $O(n \\times k)$ - Two nested loops.
**Space Complexity:** $O(n \\times k)$ - For the `dp` table.

-----

#### Optimized Space

**Intuitive Insight:** To compute row $i$, we only need row $i-1$. We can optimize the 2D DP table to a 1D array.
When computing `dp[j]` for row `i`, it depends on `dp[j-1]` (from previous row `i-1`) and `dp[j]` (from previous row `i-1`).
However, when updating in-place from left to right, `dp[j-1]` would be the *current* row's `dp[j-1]`, not the previous row's. So we need to compute from right to left or use two rows.
To optimize to $O(k)$ space: `dp[j]` for current row depends on `dp[j]` (prev row) and `dp[j-1]` (prev row).
If we compute from right to left `j` from $k$ down to $1$:
`dp[j] = dp[j] (old_value, which is dp[i-1][j]) + dp[j-1] (old_value, which is dp[i-1][j-1])`

**Code:**

```python
def binomial_coefficient_optimized_space(n, k):
    # Handle edge cases
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # C(n, k) = C(n, n-k) (Symmetry) - compute for smaller k
    if k > n // 2:
        k = n - k
        
    dp = [0] * (k + 1)
    dp[0] = 1 # C(i, 0) is always 1
    
    for i in range(1, n + 1): # Iterate through rows up to n
        # Iterate from right to left to use values from the previous row correctly
        for j in range(min(i, k), 0, -1):
            dp[j] = dp[j] + dp[j - 1]
            
    return dp[k]

# Example usage:
print(f"Binomial Coefficient (5, 2) (Optimized Space): {binomial_coefficient_optimized_space(5, 2)}") # Output: 10
print(f"Binomial Coefficient (10, 5) (Optimized Space): {binomial_coefficient_optimized_space(10, 5)}") # Output: 252
```

**Time Complexity:** $O(n \\times k)$
**Space Complexity:** $O(k)$ - For the `dp` array.

-----

#### Alternatives (Direct Formula)

**Intuitive Insight:** The direct mathematical definition using factorials.
$\\binom{n}{k} = \\frac{n\!}{k\!(n-k)\!}$

**Code:**

```python
import math

def binomial_coefficient_formula(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # C(n, k) = C(n, n-k)
    if k > n // 2:
        k = n - k
        
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

# Example usage:
print(f"Binomial Coefficient (5, 2) (Formula): {binomial_coefficient_formula(5, 2)}") # Output: 10
print(f"Binomial Coefficient (10, 5) (Formula): {binomial_coefficient_formula(10, 5)}") # Output: 252
```

**Time Complexity:** $O(k)$ - Loop runs $k$ times.
**Space Complexity:** $O(1)$

-----

### 16\. Pascal's Triangle

**Problem Statement:**
Given an integer `numRows`, return the first `numRows` of Pascal's triangle. In Pascal's triangle, each number is the sum of the two numbers directly above it.

**Example:**
Input: `numRows = 5`
Output:

```
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

-----

#### Recursion (Brute Force)

**Intuitive Insight:** Each element `triangle[i][j]` is $\\binom{i}{j}$. We can use the recursive definition of binomial coefficients.

**Code:**

```python
def pascals_triangle_recursive(numRows):
    if numRows == 0:
        return []
    
    result = []
    
    # Helper for binomial coefficient (can be memoized for efficiency)
    memo_nCr = {}
    def nCr_helper(n_val, r_val):
        if r_val < 0 or r_val > n_val:
            return 0
        if r_val == 0 or r_val == n_val:
            return 1
        if (n_val, r_val) in memo_nCr:
            return memo_nCr[(n_val, r_val)]
        
        res = nCr_helper(n_val - 1, r_val - 1) + nCr_helper(n_val - 1, r_val)
        memo_nCr[(n_val, r_val)] = res
        return res

    for i in range(numRows):
        current_row = []
        for j in range(i + 1):
            current_row.append(nCr_helper(i, j))
        result.append(current_row)
        
    return result

# Example usage:
print(f"Pascal's Triangle (5) (Recursive): {pascals_triangle_recursive(5)}")
```

**Time Complexity:** $O(numRows^2 \\times numRows)$ = $O(numRows^3)$ if `nCr_helper` is exponential and not memoized. With memoized `nCr_helper`, it's $O(numRows^2)$.
**Space Complexity:** $O(numRows^2)$ for the result list and memoization table.

-----

#### Memoization (Top-Down)

This is effectively covered by the above recursive solution using memoization for `nCr`.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Each row can be built from the previous row. The first and last elements of each row are 1. Other elements are the sum of the two elements directly above it.

**Code:**

```python
def pascals_triangle_tabulation(numRows):
    if numRows == 0:
        return []
    
    triangle = []
    
    # First row
    triangle.append([1])
    
    for i in range(1, numRows):
        prev_row = triangle[i - 1]
        current_row = [1] # First element is always 1
        
        for j in range(1, i): # Elements in between
            current_row.append(prev_row[j - 1] + prev_row[j])
            
        current_row.append(1) # Last element is always 1
        triangle.append(current_row)
        
    return triangle

# Example usage:
print(f"Pascal's Triangle (5) (Tabulation): {pascals_triangle_tabulation(5)}")
```

**Time Complexity:** $O(numRows^2)$ - Two nested loops. The outer loop runs `numRows` times, and the inner loop runs up to `numRows` times.
**Space Complexity:** $O(numRows^2)$ - To store the entire triangle.

-----

#### Optimized Space

Not applicable for storing the entire triangle. If the question was to return only the $N$-th row, then optimized space is possible (see next problem).

-----

### 17\. Nth Row of Pascal Triangle

**Problem Statement:**
Given a non-negative integer `rowIndex`, return the `rowIndex`-th row of the Pascal's triangle. (Row index starts from 0).

**Example:**
Input: `rowIndex = 3`
Output: `[1, 3, 3, 1]`

-----

#### Recursion (Brute Force)

**Intuitive Insight:** Same as `binomial_coefficient_recursive`, but we call it for all `j` from `0` to `rowIndex`.

**Code:**

```python
def get_row_recursive(rowIndex):
    if rowIndex < 0:
        return []
    
    row = []
    
    # Helper for binomial coefficient (can be memoized)
    memo_nCr = {}
    def nCr_helper(n_val, r_val):
        if r_val < 0 or r_val > n_val:
            return 0
        if r_val == 0 or r_val == n_val:
            return 1
        if (n_val, r_val) in memo_nCr:
            return memo_nCr[(n_val, r_val)]
        
        res = nCr_helper(n_val - 1, r_val - 1) + nCr_helper(n_val - 1, r_val)
        memo_nCr[(n_val, r_val)] = res
        return res

    for j in range(rowIndex + 1):
        row.append(nCr_helper(rowIndex, j))
        
    return row

# Example usage:
print(f"Nth Row of Pascal Triangle (3) (Recursive): {get_row_recursive(3)}") # Output: [1, 3, 3, 1]
```

**Time Complexity:** $O(rowIndex^2)$ if `nCr_helper` is memoized, otherwise exponential.
**Space Complexity:** $O(rowIndex^2)$ for `memo_nCr` and $O(rowIndex)$ for recursion stack.

-----

#### Memoization (Top-Down)

This is implicitly covered when using memoized `nCr_helper`.

-----

#### Tabulation (Bottom-Up)

**Intuitive Insight:** Similar to generating the full Pascal's triangle, but we only store the current row being built.

**Code:**

```python
def get_row_tabulation(rowIndex):
    if rowIndex < 0:
        return []
    
    # Initialize with the first row
    row = [1]
    
    for i in range(1, rowIndex + 1): # Iterate for rows from 1 up to rowIndex
        new_row = [1] # First element of new row is 1
        for j in range(1, i): # Compute inner elements based on previous row (which is 'row')
            new_row.append(row[j - 1] + row[j])
        new_row.append(1) # Last element of new row is 1
        row = new_row # Update 'row' to be the newly computed row
        
    return row

# Example usage:
print(f"Nth Row of Pascal Triangle (3) (Tabulation): {get_row_tabulation(3)}") # Output: [1, 3, 3, 1]
print(f"Nth Row of Pascal Triangle (0) (Tabulation): {get_row_tabulation(0)}") # Output: [1]
```

**Time Complexity:** $O(rowIndex^2)$ - Outer loop runs `rowIndex` times, inner loop runs up to `rowIndex` times.
**Space Complexity:** $O(rowIndex)$ - To store the current row.

-----

#### Optimized Space

**Intuitive Insight:** We can compute the row in-place. The current element `row[j]` depends on `row[j]` (from previous iteration/row) and `row[j-1]` (from previous iteration/row). To correctly use the previous values, iterate from right to left.

**Code:**

```python
def get_row_optimized_space(rowIndex):
    if rowIndex < 0:
        return []
        
    row = [0] * (rowIndex + 1)
    row[0] = 1 # C(rowIndex, 0) is always 1
    
    for i in range(1, rowIndex + 1): # i represents the current row number we are conceptually building
        # Iterate from right to left to ensure we use values from the previous 'row' correctly
        for j in range(i, 0, -1): # j goes from i down to 1
            row[j] = row[j] + row[j - 1]
            
    return row

# Example usage:
print(f"Nth Row of Pascal Triangle (3) (Optimized Space): {get_row_optimized_space(3)}") # Output: [1, 3, 3, 1]
print(f"Nth Row of Pascal Triangle (4) (Optimized Space): {get_row_optimized_space(4)}") # Output: [1, 4, 6, 4, 1]
```

**Time Complexity:** $O(rowIndex^2)$ - Two nested loops.
**Space Complexity:** $O(rowIndex)$ - For the single `row` array.

-----

#### Alternatives (Direct Formula)

**Intuitive Insight:** Each element in the `rowIndex`-th row is a binomial coefficient $\\binom{rowIndex}{j}$. We can calculate these directly using the formula $\\binom{n}{k} = \\frac{n\!}{k\!(n-k)\!}$ or iteratively.

**Code:**

```python
def get_row_formula(rowIndex):
    if rowIndex < 0:
        return []
        
    row = [0] * (rowIndex + 1)
    row[0] = 1 # C(rowIndex, 0) = 1
    
    # Calculate C(n, k) using C(n, k) = C(n, k-1) * (n-k+1) / k
    # C(n, k) = C(n, k-1) * (n - (k-1)) / k
    # For example, C(5, 2) = C(5, 1) * (5-1)/2 = 5 * 4/2 = 10
    
    for j in range(1, rowIndex + 1):
        row[j] = row[j - 1] * (rowIndex - j + 1) // j
        
    return row

# Example usage:
print(f"Nth Row of Pascal Triangle (3) (Formula): {get_row_formula(3)}") # Output: [1, 3, 3, 1]
print(f"Nth Row of Pascal Triangle (4) (Formula): {get_row_formula(4)}") # Output: [1, 4, 6, 4, 1]
```

**Time Complexity:** $O(rowIndex)$ - A single loop.
**Space Complexity:** $O(rowIndex)$ - To store the row.

-----
