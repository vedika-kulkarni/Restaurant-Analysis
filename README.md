# 🍕 Restaurant Combo Analysis — Apriori Algorithm
## DSP Project | Market Basket Analysis

---

## 📁 Project Structure

```
apriori_project/
├── restaurant_orders.csv     ← Real-world dataset (100 orders, 11 items)
├── apriori_analysis.py       ← Full Apriori implementation (from scratch)
├── association_rules.csv     ← Generated rules with support/confidence/lift
├── frequent_itemsets.csv     ← All frequent itemsets
├── apriori_analysis.png      ← 9-panel visualization dashboard
└── README.md                 ← This file
```

---

## 🧠 What is Apriori?

Apriori is a classic **association rule mining** algorithm used to find frequent item combinations in transactional data.

### Key Concepts

| Term | Formula | Meaning |
|------|---------|---------|
| **Support** | P(A ∪ B) | How often A and B appear together |
| **Confidence** | P(B\|A) = P(A∪B)/P(A) | If A is bought, how likely is B? |
| **Lift** | Confidence / P(B) | Is the rule better than random chance? |

- **Lift > 1** → positive correlation (buy together more than expected)
- **Lift = 1** → independent
- **Lift < 1** → negative correlation

---

## ⚙️ Algorithm Steps

```
Step 1: Generate Candidate 1-itemsets (C1)
Step 2: Prune to Frequent 1-itemsets (L1) using min_support
Step 3: Join L1 to generate C2
Step 4: Prune using Apriori property (all subsets must be frequent)
Step 5: Generate L2 → repeat until no new frequent sets
Step 6: Generate association rules from all frequent itemsets
Step 7: Filter by min_confidence
```

**Apriori Property (Anti-monotone):**
> If an itemset is infrequent, ALL its supersets are also infrequent.

This dramatically **reduces the search space**.

---

## 📊 Dataset

- **100 restaurant orders** with 11 menu items
- Items: Burger, Fries, Coke, Pizza, Garlic Bread, Pasta, Sandwich, Milkshake, Pepsi, Lemonade, Ice Cream

### Parameters Used
```python
MIN_SUPPORT    = 0.15   # Item appears in ≥ 15% of orders
MIN_CONFIDENCE = 0.50   # Rule fires with ≥ 50% confidence
```

---

## 🔍 Key Results

### Top Association Rules

| Rule | Confidence | Lift |
|------|-----------|------|
| Pizza → Garlic Bread | 82% | 2.21 |
| Garlic Bread → Pizza | 73% | 2.21 |
| Fries → Burger | 61% | 1.78 |
| Burger → Fries | 59% | 1.78 |
| Burger → Coke | 56% | 1.60 |

### Meal Combo Recommendations

| Combo | Items | Why? |
|-------|-------|------|
| 🍕 Combo 1 | Pizza + Garlic Bread | Lift: 2.21 — strongest pair |
| 🍔 Combo 2 | Burger + Fries | Lift: 1.78 — classic combo |
| 🥤 Combo 3 | Burger + Fries + Coke | Classic value meal |

---

## 🚀 How to Run

```bash
# Make sure you have pandas, numpy, matplotlib, seaborn
pip install pandas numpy matplotlib seaborn

# Run the analysis
python3 apriori_analysis.py
```

No external ML libraries needed — **Apriori is implemented from scratch!**

---

## 💡 Business Use Cases

1. **Meal Combos** — Bundle Pizza + Garlic Bread as a "Pizza Combo"
2. **Upselling** — When customer orders Burger, suggest Fries
3. **Menu Optimization** — Remove/promote items based on frequency
4. **Cross-promotions** — Display Garlic Bread next to Pizza section

---

## 📈 Visualizations Generated

1. Item Frequency Bar Chart
2. Support Distribution Histogram
3. Confidence vs Lift Scatter Plot
4. Top 10 Rules by Lift
5. Item Co-occurrence Heatmap
6. Bubble Chart (Support × Confidence × Lift)
7. Itemset Size Distribution Pie Chart
8. Top Multi-Item Combos
9. Recommended Meal Combos Table

---

*DSP Project — Market Basket Analysis using Apriori Algorithm*
