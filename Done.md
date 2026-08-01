# Markdown Lists Formatting Guide

## Unordered Lists

### Concepts
- Item 1
- Item 2
- Item 3


## Ordered Lists

### Basic Ordered List
1. First item
2. Second item
3. Third item

### Ordered List Starting from Specific Number
5. Fifth item
6. Sixth item
7. Seventh item

## Nested Sublists

### Unordered Sublists
- ML Algos
  - Sub-item 1.1
  - Sub-item 1.2
    - Sub-sub-item 1.2.1
    - Sub-sub-item 1.2.2
- Main item 2
  - Sub-item 2.1
  - Sub-item 2.2

### Ordered Sublists
1. Main item 1
   1. Sub-item 1.1
   2. Sub-item 1.2
      1. Sub-sub-item 1.2.1
      2. Sub-sub-item 1.2.2
2. Main item 2
   1. Sub-item 2.1
   2. Sub-item 2.2

### Mixed Sublists (Unordered inside Ordered)
1. First main item
   - Sub-item with bullet
   - Another sub-item
    1. Numbered sub-sub-item
    2. Another numbered sub-sub-item
2. Second main item
   - Bullet sub-item
   - Another bullet sub-item

### Mixed Sublists (Ordered inside Unordered)
- Main bullet item
  1. Numbered sub-item
  2. Another numbered sub-item
    - Bullet sub-sub-item
    - Another bullet sub-sub-item
- Another main bullet item

## Task Lists (Checkboxes)
- [ ] Unchecked task
- [x] Completed task
  - [ ] Sub-task 1
  - [x] Sub-task 2

## Multi-level Deep Nesting
- Level 1
  - Level 2
    - Level 3
      - Level 4
        - Level 5 (very deep nesting)

1. Level 1
   1. Level 2
      1. Level 3
         1. Level 4

## Important Formatting Rules

### Indentation Matters
- Use **2-4 spaces** or **one tab** for each nesting level
- Inconsistent indentation may break the list rendering

### Spacing
- Always include a space after the bullet or number
- Leave a blank line before and after lists for better readability
- Use blank lines between list items if they contain multiple paragraphs

### Multi-paragraph List Items
- First paragraph of item 1

  Second paragraph (indented to align with the first paragraph)

- Second item

### Numbered List Continuation
1. First item
2. Second item
  
   Additional paragraph for item 2

3. Third item

### Best Practices
- **Be consistent** with your chosen formatting style
- Use **ordered lists** for sequential or numbered items
- Use **unordered lists** for items without a specific order
- Limit nesting to **3-4 levels** for better readability
- Use **checkboxes** for task tracking

### Rendering Preview

Here's how the basic list formats render:

- Bullet item
  - Indented sub-item
    - Further indented item

1. Numbered item
   1. Numbered sub-item
   2. Another sub-item

### Common Issues to Avoid
- Mixing different bullet styles in the same list
- Using inconsistent indentation
- Forgetting spaces after list markers
- Nesting too deeply (beyond 5 levels)
