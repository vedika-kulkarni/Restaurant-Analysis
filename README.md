# Restaurant Combo Analysis using Apriori Algorithm

A data mining and market basket analysis project implementing the Apriori Algorithm from scratch to discover frequent item combinations, association rules, and customer purchasing patterns from restaurant order transactions.

The project analyzes transactional ordering behavior to generate actionable business insights for combo recommendations, upselling strategies, and menu optimization.

---
## Live Demo
https://restaurant-analysis-five.vercel.app/

# Problem Statement

Restaurants and food businesses often struggle to identify:
- Frequently purchased item combinations
- Effective combo meal opportunities
- Customer buying behavior patterns
- Cross-selling and upselling opportunities

This project applies association rule mining techniques to uncover meaningful relationships between ordered menu items using the Apriori algorithm.

---

# Key Features

## Market Basket Analysis
- Frequent itemset generation
- Association rule mining
- Customer purchase pattern discovery
- Transactional data analysis

---

## Business Intelligence Insights
- Combo meal recommendations
- Upselling opportunities
- Menu optimization analysis
- Product affinity discovery

---

## Data Visualization Dashboard
- Item frequency analysis
- Support/confidence distribution
- Lift-based association insights
- Co-occurrence heatmaps
- Rule visualization charts

---

# Project Architecture

```text
apriori_project/
│
├── restaurant_orders.csv
├── apriori_analysis.py
├── association_rules.csv
├── frequent_itemsets.csv
├── apriori_analysis.png
└── README.md
```

---

# Tech Stack

## Programming & Analysis
- Python
- Pandas
- NumPy

## Visualization
- Matplotlib
- Seaborn

## Concepts
- Apriori Algorithm
- Association Rule Mining
- Market Basket Analysis
- Data Mining
- Business Intelligence

---

# Algorithm Workflow

```text
Step 1 → Generate Candidate Itemsets
Step 2 → Calculate Support
Step 3 → Prune Infrequent Itemsets
Step 4 → Generate Frequent Itemsets
Step 5 → Create Association Rules
Step 6 → Evaluate Confidence & Lift
Step 7 → Generate Business Insights
```

---

# Key Metrics

| Metric | Purpose |
|---|---|
| Support | Frequency of item occurrence |
| Confidence | Probability of purchasing related items |
| Lift | Strength of association between items |

---

# Dataset Information

- 100 restaurant orders
- 11 menu items analyzed

### Items Included
- Burger
- Fries
- Coke
- Pizza
- Garlic Bread
- Pasta
- Sandwich
- Milkshake
- Pepsi
- Lemonade
- Ice Cream

---

# Configuration Parameters

```python
MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.50
```

---

# Sample Insights

| Association Rule | Confidence | Lift |
|---|---|---|
| Pizza → Garlic Bread | 82% | 2.21 |
| Fries → Burger | 61% | 1.78 |
| Burger → Coke | 56% | 1.60 |

---

# Business Recommendations

## Combo Optimization
- Pizza + Garlic Bread combo recommendation
- Burger + Fries promotional bundling

---

## Upselling Strategies
- Recommend fries alongside burgers
- Suggest beverages during combo purchases

---

## Menu Intelligence
- Identify high-frequency products
- Optimize menu placement and promotions

---

# Visualizations Generated

- Item Frequency Analysis
- Support Distribution Histogram
- Confidence vs Lift Analysis
- Top Association Rules
- Item Co-occurrence Heatmap
- Bubble Analysis Dashboard
- Multi-item Combo Analysis

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/apriori_project.git
cd apriori_project
```

---

## 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

---

## 3. Run Analysis

```bash
python apriori_analysis.py
```


---

# Learning Outcomes

- Association rule mining implementation
- Apriori algorithm development
- Market basket analysis concepts
- Data visualization techniques
- Business intelligence analytics
- Customer behavior analysis

---

# Future Improvements

- Real-time recommendation engine
- Interactive analytics dashboard
- Cloud deployment
- Larger-scale datasets
- API integration
- AI-driven recommendation optimization

---

