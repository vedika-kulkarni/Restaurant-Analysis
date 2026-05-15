import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apriori_analysis import apriori
from itertools import combinations
import base64

# ─────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Combo Analytics",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0F0F1A;
        color: #FFFFFF;
    }
    .stMetric {
        background-color: #1A1A2E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .combo-card {
        background-color: #1A1A2E;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        margin-bottom: 15px;
    }
    .combo-title {
        color: #FF6B35;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .combo-stats {
        color: #88D498;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────
#  DATA LOADING (CACHED)
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_orders.csv")
    transactions = [set(row.split(",")) for row in df["items"]]
    all_items = sorted(set(item for t in transactions for item in t))
    return df, transactions, all_items

df, transactions, all_items = load_data()

# ─────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3655/3655682.png", width=80)
    st.title("APRIORI SETTINGS")
    st.markdown("Adjust parameters to find patterns in order data.")
    
    min_sup = st.slider("Min Support", 0.05, 0.50, 0.15, 0.01, help="Min percentage of orders containing the items.")
    min_conf = st.slider("Min Confidence", 0.10, 1.00, 0.50, 0.05, help="Min probability of item B given item A.")
    
    st.divider()
    st.info(f"📊 Dataset: {len(transactions)} Orders\n🍎 Unique Items: {len(all_items)}")

# ─────────────────────────────────────────────────
#  ALGORITHM RUN
# ─────────────────────────────────────────────────
with st.spinner("Analyzing patterns..."):
    frequent_itemsets, rules_df = apriori(transactions, min_sup, min_conf)

# ─────────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────────
st.title("🍕 Restaurant Combo Analytics")
st.markdown("### Discovering Menu Item Associations using the Apriori Algorithm")

# Top Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Frequent Itemsets", len(frequent_itemsets))
m2.metric("Association Rules", len(rules_df))
m3.metric("Avg Order Size", round(df['items'].apply(lambda x: len(x.split(","))).mean(), 1))

st.divider()

# Layout: Rules Table and Visuals
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Association Rules")
    if not rules_df.empty:
        # Style and Display Table
        styled_rules = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
        st.dataframe(styled_rules, 
                     column_config={
                         "support": st.column_config.NumberColumn(format="%.3f"),
                         "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                         "lift": st.column_config.NumberColumn(format="%.2f"),
                     },
                     use_container_width=True,
                     hide_index=True)
    else:
        st.warning("No rules found. Try lowering Min Support or Confidence.")

with col_right:
    st.subheader("📈 Visualization")
    
    if not rules_df.empty:
        # Plot Confidence vs Lift
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0F0F1A")
        ax.set_facecolor("#1A1A2E")
        
        sc = ax.scatter(rules_df['confidence'], rules_df['lift'], 
                         c=rules_df['support'], cmap='plasma', s=100, alpha=0.8)
        plt.colorbar(sc, label='Support')
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Lift")
        ax.set_title("Rule Confidence vs Lift (Size/Color=Support)")
        ax.grid(alpha=0.2)
        
        st.pyplot(fig)
    else:
        st.info("Visualizations will appear once rules are found.")

# ─────────────────────────────────────────────────
#  BOTTOM SECTION: RECOMMENDATIONS
# ─────────────────────────────────────────────────
st.divider()
st.subheader("💡 Recommended Meal Combos")

if not rules_df.empty:
    high_lift = rules_df.sort_values("lift", ascending=False).head(6)
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(high_lift.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="combo-card">
                <div class="combo-title">Combo {idx+1}: {row['antecedent']} + {row['consequent']}</div>
                <div class="combo-stats">
                    Confidence: <b>{row['confidence']:.0%}</b> | Lift: <b>{row['lift']:.2f}</b><br>
                    Support: {row['support']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.write("Reduce filters to see combo recommendations.")

# ─────────────────────────────────────────────────
#  CO-OCCURRENCE HEATMAP
# ─────────────────────────────────────────────────
st.divider()
st.subheader("🔥 Item Co-occurrence Matrix")
top_items = all_items[:10]  # Limit to top 10 for clarity
matrix = pd.DataFrame(0, index=top_items, columns=top_items)
for t in transactions:
    t_items = [i for i in t if i in top_items]
    for i, j in combinations(t_items, 2):
        matrix.loc[i, j] += 1
        matrix.loc[j, i] += 1

fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
fig_hm.patch.set_facecolor("#0F0F1A")
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax_hm, 
            cbar_kws={'label': 'Number of Co-purchases'})
ax_hm.set_title("How often items are bought together", color="white")
st.pyplot(fig_hm)

st.markdown("---")
st.caption("DSP Project — Market Basket Analysis Dashbaord | Built with Streamlit")
