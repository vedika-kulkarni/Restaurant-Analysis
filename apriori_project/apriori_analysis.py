"""
=============================================================
 DSP Project: Restaurant Combo Analysis using Apriori Algorithm
 Author      : Student DSP Project
 Description : Apriori algorithm implemented from scratch
               (no mlxtend) on real-world restaurant order data
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
#  APRIORI ALGORITHM CORE FUNCTIONS
# ─────────────────────────────────────────────────

def get_support(itemset, transactions):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def apriori(transactions, min_support=0.15, min_confidence=0.5):
    """Full Apriori algorithm implementation."""
    n = len(transactions)

    # --- C1: Candidate 1-itemsets ---
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] += 1

    # --- L1: Frequent 1-itemsets ---
    L1 = {k: v/n for k, v in item_counts.items() if v/n >= min_support}
    all_frequent = dict(L1)

    k = 2
    Lk_prev = L1

    while Lk_prev:
        # Generate candidates by joining
        candidates = set()
        prev_list = list(Lk_prev.keys())
        for i in range(len(prev_list)):
            for j in range(i+1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k:
                    candidates.add(union)

        # Prune: all (k-1) subsets must be frequent
        pruned = set()
        for cand in candidates:
            all_sub_frequent = all(
                frozenset(sub) in Lk_prev
                for sub in combinations(cand, k-1)
            )
            if all_sub_frequent:
                pruned.add(cand)

        # Count support
        Lk = {}
        for cand in pruned:
            sup = sum(1 for t in transactions if cand.issubset(t)) / n
            if sup >= min_support:
                Lk[cand] = sup

        all_frequent.update(Lk)
        Lk_prev = Lk
        k += 1

    # --- Generate Association Rules ---
    rules = []
    for itemset, support in all_frequent.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                ant_support = all_frequent.get(antecedent, get_support(antecedent, transactions))
                if ant_support == 0:
                    continue
                confidence = support / ant_support
                if confidence >= min_confidence:
                    lift = confidence / all_frequent.get(consequent, get_support(consequent, transactions))
                    rules.append({
                        "antecedent": ", ".join(sorted(antecedent)),
                        "consequent": ", ".join(sorted(consequent)),
                        "support"   : round(support, 4),
                        "confidence": round(confidence, 4),
                        "lift"      : round(lift, 4)
                    })
    return all_frequent, pd.DataFrame(rules)

# ─────────────────────────────────────────────────
#  MAIN EXECUTION BLOCK
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    # ── STEP 1: Load Dataset ─────────────────────
    print("=" * 60)
    print("  DSP PROJECT: RESTAURANT COMBO ANALYSIS (APRIORI)")
    print("=" * 60)

    try:
        df = pd.read_csv("restaurant_orders.csv")
    except FileNotFoundError:
        print("[ERROR] restaurant_orders.csv not found!")
        exit(1)

    transactions = [set(row.split(",")) for row in df["items"]]

    print(f"\n[INFO] Total Orders Loaded : {len(transactions)}")
    all_items = sorted(set(item for t in transactions for item in t))
    print(f"[INFO] Unique Menu Items   : {len(all_items)}")
    print(f"[INFO] Items               : {all_items}\n")

    # ── STEP 3: Run Apriori ──────────────────────
    MIN_SUPPORT    = 0.15
    MIN_CONFIDENCE = 0.5

    print("[RUNNING] Apriori Algorithm...")
    frequent_itemsets, rules_df = apriori(transactions, MIN_SUPPORT, MIN_CONFIDENCE)

    print(f"\n[RESULT] Frequent Itemsets Found : {len(frequent_itemsets)}")
    print(f"[RESULT] Association Rules Found : {len(rules_df)}\n")

    # ── STEP 4: Results Display ──────────────────
    print("-" * 60)
    print(" FREQUENT ITEMSETS (Support >= 15%)")
    print("-" * 60)
    itemset_rows = []
    for k, v in sorted(frequent_itemsets.items(), key=lambda x: -x[1]):
        row = {"Itemset": ", ".join(sorted(k)), "Support": f"{v*100:.1f}%"}
        itemset_rows.append(row)
        print(f"  {row['Itemset']:<35} Support: {row['Support']}")

    print("\n" + "-" * 60)
    print(" TOP ASSOCIATION RULES (Confidence >= 50%)")
    print("-" * 60)
    if not rules_df.empty:
        rules_df_sorted = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
        for _, row in rules_df_sorted.head(15).iterrows():
            print(f"  {row['antecedent']:<25} ->  {row['consequent']:<20}  "
                  f"Sup:{row['support']:.2f}  Conf:{row['confidence']:.2f}  Lift:{row['lift']:.2f}")

        # ── STEP 6: Save CSVs ──────────────────────
        rules_df_sorted.to_csv("association_rules.csv", index=False)
        pd.DataFrame(itemset_rows).to_csv("frequent_itemsets.csv", index=False)
        print("\n[SAVED] association_rules.csv")
        print("[SAVED] frequent_itemsets.csv")
    else:
        print(" [!] No association rules found with current parameters.")
        rules_df_sorted = pd.DataFrame()

    # ── STEP 7: Visualizations ───────────────────
    if not rules_df.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        COLORS = ["#FF6B35","#F7C59F","#EFEFD0","#004E89","#1A936F",
                "#88D498","#C6DABF","#FFD166","#EF476F","#06D6A0"]

        fig = plt.figure(figsize=(20, 22))
        fig.patch.set_facecolor("#0F0F1A")
        fig.suptitle("Restaurant Combo Analysis - Apriori Algorithm",
                    fontsize=22, fontweight='bold', color='white', y=0.98)

        # ── Chart 1: Item Frequency Bar ─────────────
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.set_facecolor("#1A1A2E")
        item_freq = defaultdict(int)
        for t in transactions:
            for item in t:
                item_freq[item] += 1
        sorted_items = sorted(item_freq.items(), key=lambda x: -x[1])
        items_names, items_counts = zip(*sorted_items)
        bars = ax1.barh(items_names, items_counts, color=COLORS[:len(items_names)])
        ax1.set_title("Item Frequency", color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.set_xlabel("Count", color='white')
        for bar, count in zip(bars, items_counts):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', color='white', fontsize=8)
        for sp in ax1.spines.values(): sp.set_color('#555')

        # ── Chart 2: Support Distribution ───────────
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.set_facecolor("#1A1A2E")
        supports = [v for k, v in frequent_itemsets.items() if len(k) > 1]
        if supports:
            ax2.hist(supports, bins=12, color="#FF6B35", edgecolor="#0F0F1A", alpha=0.85)
            ax2.axvline(0.15, color='yellow', linestyle='--', linewidth=1.5, label='Min Support=0.15')
            ax2.set_title("Support Distribution", color='white', fontweight='bold')
            ax2.tick_params(colors='white')
            ax2.set_xlabel("Support", color='white')
            ax2.legend(facecolor='#1A1A2E', labelcolor='white', fontsize=8)
        for sp in ax2.spines.values(): sp.set_color('#555')

        # ── Chart 3: Confidence vs Lift Scatter ─────
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.set_facecolor("#1A1A2E")
        sc = ax3.scatter(rules_df_sorted['confidence'], rules_df_sorted['lift'],
                        c=rules_df_sorted['support'], cmap='plasma',
                        s=80, alpha=0.85, edgecolors='white', linewidths=0.3)
        plt.colorbar(sc, ax=ax3, label='Support').ax.yaxis.label.set_color('white')
        ax3.set_title("Confidence vs Lift", color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        for sp in ax3.spines.values(): sp.set_color('#555')

        # ── Chart 5: Heatmap ────────────────────────
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.set_facecolor("#1A1A2E")
        top_items = [item for item, _ in sorted_items[:8]]
        matrix = pd.DataFrame(0, index=top_items, columns=top_items)
        for t in transactions:
            t_items = [i for i in t if i in top_items]
            for i, j in combinations(t_items, 2):
                matrix.loc[i, j] += 1
                matrix.loc[j, i] += 1
        sns.heatmap(matrix, ax=ax5, cmap='YlOrRd', annot=True, fmt='d',
                    linewidths=0.5, linecolor='#0F0F1A')
        ax5.set_title("Item Co-occurrence Matrix", color='white', fontweight='bold')
        ax5.tick_params(colors='white', labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("apriori_analysis.png", dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print("[SAVED] apriori_analysis.png")

    # ── STEP 8: Report ───────────────────────────
    print("\n" + "=" * 60)
    print("  MEAL COMBO RECOMMENDATIONS")
    print("=" * 60)
    print("\n* Based on Apriori Analysis, suggest these combos:\n")

    if not rules_df.empty:
        high_lift = rules_df_sorted[rules_df_sorted['lift'] > 1.5].head(8)
        for i, (_, r) in enumerate(high_lift.iterrows(), 1):
            print(f"  Combo {i}: {r['antecedent']} + {r['consequent']}")
            print(f"           Confidence: {r['confidence']:.0%}  |  Lift: {r['lift']:.2f}\n")
    
    print("=" * 60)
    print("  ANALYSIS COMPLETE - All files saved!")
    print("=" * 60)
