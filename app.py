import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# ==============================================================================
# PAGE CONFIGURATION & EDITORIAL BRANDING
# ==============================================================================
st.set_page_config(
    page_title="ZARA | Enterprise Catalog Intelligence & Predictive Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Zara's Signature Editorial Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Clean white editorial canvas */
    .stApp {
        background-color: #FFFFFF;
        color: #111111;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Editorial header with stark hairline rule */
    .zara-header {
        border-bottom: 1px solid #111111;
        padding-bottom: 22px;
        margin-bottom: 30px;
    }
    .zara-pretitle {
        font-size: 11px;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: #767676;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .zara-title {
        font-family: 'Cinzel', 'Times New Roman', serif;
        font-size: 32px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #000000;
        margin: 0 0 8px 0;
    }
    .zara-description {
        font-size: 13px;
        color: #555555;
        letter-spacing: 0.02em;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards: Minimalist, sharp, unrounded */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 0px;
        padding: 20px 16px;
        text-align: left;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #000000;
    }
    .metric-sub {
        font-size: 10px;
        color: #767676;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-val {
        font-family: 'Cinzel', 'Times New Roman', serif;
        font-size: 28px;
        font-weight: 500;
        color: #000000;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .metric-tag {
        font-size: 11px;
        color: #222222;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    
    /* Result card for inference & valuation */
    .zara-result-card {
        background-color: #FAFAFA;
        border: 1px solid #111111;
        border-radius: 0px;
        padding: 24px;
        margin-bottom: 18px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #EEEEEE;
    }
    
    /* Button & link overrides */
    .stButton > button, div.stDownloadButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 0px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        font-size: 11px !important;
        border: 1px solid #000000 !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #262626 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA INGESTION & PIPELINE CACHING
# ==============================================================================
@st.cache_data
def load_data():
    local_path = "fast_fashion_zara_dataset.csv"
    try:
        df = pd.read_csv(local_path)
    except Exception:
        url = "https://drive.google.com/uc?export=download&id=17ke231Lv8I2I9ddZ85LQJ5qLGrNXBZ6e"
        df = pd.read_csv(url)
    
    df = df.dropna(subset=['Price_INR', 'Gender', 'Category']).copy()
    return df

df = load_data()

# Model features
features_cat = ['Category']
features_meta = ['Price_INR', 'Details_Length', 'Num_Images']
features_fabrics = ['Fabric_Cotton', 'Fabric_Leather', 'Fabric_Linen', 'Fabric_Denim', 'Fabric_Wool_Knit']
features_all_num = features_meta + features_fabrics

full_prep = ColumnTransformer([
    ('num', StandardScaler(), features_all_num),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
], sparse_threshold=0)

@st.cache_resource
def train_production_models(df):
    y = (df['Gender'] == 'Women').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=42, stratify=y)
    
    # 1. Random Forest Classifier
    rf_clf = Pipeline([
        ('prep', full_prep),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    rf_clf.fit(X_train, y_train)
    
    # 2. HistGradientBoosting Classifier
    hgb_clf = Pipeline([
        ('prep', full_prep),
        ('clf', HistGradientBoostingClassifier(random_state=42))
    ])
    hgb_clf.fit(X_train, y_train)
    
    # 3. Logistic Regression Classifier
    lr_clf = Pipeline([
        ('prep', full_prep),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_clf.fit(X_train, y_train)
    
    # 4. Pricing Regressor (Explainable Random Forest Regressor)
    reg_num = ['Details_Length', 'Num_Images'] + features_fabrics
    reg_prep = ColumnTransformer([
        ('num', StandardScaler(), reg_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ], sparse_threshold=0)
    
    rf_reg = Pipeline([
        ('prep', reg_prep),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
    ])
    rf_reg.fit(df[['Category'] + reg_num], df['Price_INR'])
    
    # Feature importances mapping
    cat_names = list(rf_reg.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(features_cat))
    reg_feat_names = reg_num + cat_names
    pricing_importances = pd.Series(rf_reg.named_steps['rf'].feature_importances_, index=reg_feat_names).sort_values(ascending=False)
    
    return {
        'rf_clf': rf_clf,
        'hgb_clf': hgb_clf,
        'lr_clf': lr_clf,
        'rf_reg': rf_reg,
        'pricing_importances': pricing_importances,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

models_bundle = train_production_models(df)

# ==============================================================================
# SIDEBAR NAVIGATION (EDITORIAL MINIMALISM)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Zara_Logo.svg", width=150)
    
    st.markdown("""
    <div style="font-size: 10px; letter-spacing: 0.22em; text-transform: uppercase; color: #888888; margin-top: 10px; font-weight: 600;">
        COLLECTION INTELLIGENCE
    </div>
    <div style="font-size: 12px; color: #555555; line-height: 1.45; margin: 4px 0 16px 0;">
        Demographic classification and retail valuation across 4,179 commercial garments.
    </div>
    <hr style="border: none; border-top: 1px solid #E5E5E5; margin: 12px 0 18px 0;">
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        [
            "Collection Overview",
            "Demographic Classification",
            "Garment Valuation Model",
            "Model Sensitivity & Benchmarks",
            "Crossover Diagnostics & Catalog Audit"
        ]
    )
    
    st.markdown("""
    <hr style="border: none; border-top: 1px solid #E5E5E5; margin: 28px 0 14px 0;">
    <div style="font-size: 10px; letter-spacing: 0.20em; text-transform: uppercase; color: #888888; margin-bottom: 8px; font-weight: 600;">
        PLATFORM RESOURCES
    </div>
    """, unsafe_allow_html=True)
    
    st.link_button("GitHub Repository", "https://github.com/deepakraolive/zara-apparel-classification")
    st.link_button("Google Colab Environment", "https://colab.research.google.com/drive/1z_l8rwBcJZMH4ZyJ8qdzBP2pFifq3Dvp?usp=sharing")

# ==============================================================================
# PAGE 1: COLLECTION OVERVIEW
# ==============================================================================
if page == "Collection Overview":
    st.markdown("""
    <div class="zara-header">
        <div class="zara-pretitle">ZARA INDIA / COMMERCIAL INTELLIGENCE</div>
        <h1 class="zara-title">COLLECTION OVERVIEW</h1>
        <p class="zara-description">
            Empirical demographic distribution, catalog volume metrics, and department pricing structures across 4,179 commercial garments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 5 Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-sub">Catalog Volume</div><div class="metric-val">4,179</div><span class="metric-tag">35 Departments</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-sub">Majority Prior</div><div class="metric-val">64.71%</div><span class="metric-tag">Women\'s Segment</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-sub">Peak Accuracy</div><div class="metric-val">92.50%</div><span class="metric-tag">+27.79% Net Lift</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-sub">5-Fold Stratified CV</div><div class="metric-val">91.17%</div><span class="metric-tag">±0.75% Std (0.9788 AUC)</span></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><div class="metric-sub">Stability Spread</div><div class="metric-val">&lt; 1.83%</div><span class="metric-tag">6 Partition Schemes</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1 Visualizations (Monochrome & Warm Neutral)
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        st.subheader("Figure 1A: Retail Pricing Density by Demographic Segment")
        fig_price = px.histogram(
            df, x="Price_INR", color="Gender", barmode="overlay", nbins=40,
            color_discrete_map={"Men": "#111111", "Women": "#8C827A"},
            opacity=0.7, marginal="box",
            labels={"Price_INR": "Retail Price in INR (₹)", "Gender": "Segment"}
        )
        fig_price.update_layout(
            xaxis_range=[0, 15000], height=380, template="plotly_white",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#111111")
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
    with vcol2:
        st.subheader("Figure 1B: Top 10 Fast-Fashion Apparel Departments")
        top_cats = df['Category'].value_counts().head(10).reset_index()
        top_cats.columns = ['Category', 'Product_Count']
        fig_cats = px.bar(
            top_cats, x="Product_Count", y="Category", orientation='h',
            color="Product_Count", color_continuous_scale="Greys",
            labels={"Product_Count": "Number of SKUs", "Category": "Department"}
        )
        fig_cats.update_layout(
            yaxis=dict(autorange="reversed"), height=380, template="plotly_white",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#111111")
        )
        st.plotly_chart(fig_cats, use_container_width=True)
        
    # Demographic & Pricing Summary Table
    st.markdown("### Commercial Parametric Summary")
    p_summary = df.groupby('Gender')['Price_INR'].describe()[['count', 'mean', 'std', 'min', '50%', 'max']].round(2)
    p_summary.columns = ['Silhouettes', 'Mean (₹)', 'Std Dev (₹)', 'Min (₹)', 'Median (₹)', 'Max (₹)']
    st.dataframe(p_summary, use_container_width=True)

# ==============================================================================
# PAGE 2: DEMOGRAPHIC CLASSIFICATION
# ==============================================================================
elif page == "Demographic Classification":
    st.markdown("""
    <div class="zara-header">
        <div class="zara-pretitle">ZARA MERCHANDISING / INFERENCE ENGINE</div>
        <h1 class="zara-title">DEMOGRAPHIC CLASSIFICATION</h1>
        <p class="zara-description">
            Algorithmic demographic segmentation based on structural garment parameters and engineered textile compositions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    icol1, icol2 = st.columns([1, 1.2])
    
    with icol1:
        st.markdown("#### Configure Garment Attributes")
        
        dept_options = sorted(df['Category'].unique().tolist())
        selected_cat = st.selectbox("Department Category:", dept_options, index=dept_options.index("JACKETS") if "JACKETS" in dept_options else 0)
        
        selected_price = st.slider("Retail Sale Price in INR (₹):", min_value=390, max_value=18990, value=3990, step=100)
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            desc_len = st.number_input("Description Length (Chars):", min_value=10, max_value=1200, value=220, step=10)
        with m_col2:
            num_images = st.slider("Catalog Image Count:", min_value=1, max_value=12, value=4, step=1)
            
        st.markdown("##### Textile Fiber Composition:")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            c_cotton = st.checkbox("100% Cotton Fiber", value=True)
            c_linen = st.checkbox("Linen Fabric", value=False)
            c_denim = st.checkbox("Denim Material", value=False)
        with tcol2:
            c_leather = st.checkbox("Genuine / Faux Leather", value=False)
            c_wool = st.checkbox("Wool / Knitwear", value=False)
            
        selected_model_name = st.selectbox("Classification Architecture:", ["Random Forest (Production)", "HistGradientBoosting", "Logistic Regression"])
        
    with icol2:
        st.markdown("#### Real-Time Model Inference & Predictions")
        
        input_data = pd.DataFrame([{
            'Category': selected_cat,
            'Price_INR': float(selected_price),
            'Details_Length': int(desc_len),
            'Num_Images': int(num_images),
            'Fabric_Cotton': int(c_cotton),
            'Fabric_Leather': int(c_leather),
            'Fabric_Linen': int(c_linen),
            'Fabric_Denim': int(c_denim),
            'Fabric_Wool_Knit': int(c_wool)
        }])
        
        # Select active estimator
        estimator = models_bundle['rf_clf'] if selected_model_name.startswith("Random") else (
            models_bundle['hgb_clf'] if selected_model_name.startswith("Hist") else models_bundle['lr_clf']
        )
        
        pred_label = estimator.predict(input_data)[0]
        pred_probs = estimator.predict_proba(input_data)[0]
        
        pred_gender = "Women's Collection" if pred_label == 1 else "Men's Collection"
        women_prob = pred_probs[1] * 100
        men_prob = pred_probs[0] * 100
        
        # Display Result Card in Zara High-End Box
        st.markdown(f"""
        <div class="zara-result-card">
            <div class="metric-sub">CLASSIFIED DEMOGRAPHIC SEGMENT</div>
            <div style="font-family: 'Cinzel', 'Times New Roman', serif; font-size: 32px; letter-spacing: 0.06em; text-transform: uppercase; color: #000000; margin: 4px 0 8px 0;">{pred_gender}</div>
            <div style="font-size: 13px; color: #555555;">Model Confidence: <strong>{max(women_prob, men_prob):.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Minimalist Editorial Probability Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=women_prob,
            title={'text': "Women's Collection Likelihood (%)", 'font': {'size': 13, 'color': '#111111', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#111111'},
                'bar': {'color': "#111111"},
                'bgcolor': "#FFFFFF",
                'borderwidth': 1,
                'bordercolor': "#E5E5E5",
                'steps': [
                    {'range': [0, 50], 'color': '#FAFAFA'},
                    {'range': [50, 100], 'color': '#EFEFEF'}
                ],
                'threshold': {
                    'line': {'color': "#000000", 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=260, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        **Inference Diagnostic Breakdown:**
        * Silhouette classified into **{pred_gender}** based on `{selected_cat}` department priors and a price tier of **₹{selected_price:,}**.
        * Fabric indicators (Leather: `{c_leather}`, Linen: `{c_linen}`, Denim: `{c_denim}`) actively shift decision boundaries relative to catalog distributions.
        """)

# ==============================================================================
# PAGE 3: GARMENT VALUATION MODEL
# ==============================================================================
elif page == "Garment Valuation Model":
    st.markdown("""
    <div class="zara-header">
        <div class="zara-pretitle">ZARA PRICING ARCHITECTURE / VALUATION</div>
        <h1 class="zara-title">GARMENT VALUATION MODEL</h1>
        <p class="zara-description">
            Predictive retail valuation and sensitivity analysis across textile determinants and construction complexity ($R^2 = 0.54$).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns([1, 1.2])
    
    with pcol1:
        st.markdown("#### Garment Specification Inputs")
        dept_options = sorted(df['Category'].unique().tolist())
        r_cat = st.selectbox("Apparel Department:", dept_options, index=dept_options.index("BLAZERS") if "BLAZERS" in dept_options else 0)
        
        r_desc_len = st.slider("Product Description Verbosity (Chars):", 20, 1000, 320, 20)
        r_images = st.slider("Number of Catalog Photos:", 1, 10, 5)
        
        st.markdown("##### Material Composition Surcharges:")
        r_leather = st.checkbox("Includes Genuine / Faux Leather", value=False)
        r_linen = st.checkbox("100% Linen Composition", value=False)
        r_cotton = st.checkbox("High-Grade Cotton", value=True)
        r_denim = st.checkbox("Specialty Denim Finish", value=False)
        r_wool = st.checkbox("Merino Wool / Knitwear", value=False)
        
        reg_input = pd.DataFrame([{
            'Category': r_cat,
            'Details_Length': int(r_desc_len),
            'Num_Images': int(r_images),
            'Fabric_Cotton': int(r_cotton),
            'Fabric_Leather': int(r_leather),
            'Fabric_Linen': int(r_linen),
            'Fabric_Denim': int(r_denim),
            'Fabric_Wool_Knit': int(r_wool)
        }])
        
        predicted_price = models_bundle['rf_reg'].predict(reg_input)[0]
        
        st.markdown(f"""
        <div class="zara-result-card" style="text-align: center;">
            <div class="metric-sub">ESTIMATED RETAIL PRICE (ZARA INDIA)</div>
            <div style="font-family: 'Cinzel', 'Times New Roman', serif; font-size: 38px; font-weight: 500; color: #000000; margin: 6px 0;">₹{predicted_price:,.2f}</div>
            <div style="font-size: 12px; color: #767676; letter-spacing: 0.04em;">RECOMMENDED RANGE: ₹{max(390, predicted_price*0.85):,.0f} &ndash; ₹{predicted_price*1.15:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with pcol2:
        st.markdown("#### Primary Drivers of Garment Valuation")
        top_drivers = models_bundle['pricing_importances'].head(10).reset_index()
        top_drivers.columns = ['Feature', 'Importance']
        top_drivers['Importance_Pct'] = top_drivers['Importance'] * 100
        
        fig_drivers = px.bar(
            top_drivers, x="Importance_Pct", y="Feature", orientation='h',
            color="Importance_Pct", color_continuous_scale="Greys",
            labels={"Importance_Pct": "Relative Pricing Impact (%)", "Feature": "Engineered Attribute"}
        )
        fig_drivers.update_layout(
            yaxis=dict(autorange="reversed"), height=420, template="plotly_white",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#111111")
        )
        st.plotly_chart(fig_drivers, use_container_width=True)
        
        st.markdown("""
        **Merchandising & Commercial Findings:**
        1. **Product Description Length (35.40% importance):** Technical specification length proxies intricate stitching, multi-piece construction, and tailoring complexity.
        2. **Leather Textile Presence (11.51% premium):** Genuine and vegan leather constitutes the single highest material surcharge across Zara's catalog.
        3. **Structured Outerwear:** Tailored blazers and jackets command the highest margin tiers across both demographic sections.
        """)

# ==============================================================================
# PAGE 4: MODEL SENSITIVITY & BENCHMARKS
# ==============================================================================
elif page == "Model Sensitivity & Benchmarks":
    st.markdown("""
    <div class="zara-header">
        <div class="zara-pretitle">STATISTICAL VALIDATION / BENCHMARKS</div>
        <h1 class="zara-title">MODEL SENSITIVITY & STABILITY</h1>
        <p class="zara-description">
            Empirical stability across varying catalog partition schemes, ablation progression, and 5-fold cross-validation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Multi-Split Data-Decay Curves", "4-Level Ablation Progression", "5-Fold Stratified Cross-Validation"])
    
    with tab1:
        st.subheader("Performance Across 6 Train:Test Partition Ratios")
        
        decay_data = pd.DataFrame([
            {'Split': '85:15', 'Model': 'Random Forest', 'Accuracy': 0.9250, 'F1-Score': 0.9416},
            {'Split': '85:15', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9250, 'F1-Score': 0.9429},
            {'Split': '85:15', 'Model': 'Logistic Regression', 'Accuracy': 0.8884, 'F1-Score': 0.9105},
            {'Split': '85:15', 'Model': 'Support Vector Machine', 'Accuracy': 0.9059, 'F1-Score': 0.9245},
            {'Split': '85:15', 'Model': 'Decision Tree', 'Accuracy': 0.8070, 'F1-Score': 0.8254},
            
            {'Split': '80:20', 'Model': 'Random Forest', 'Accuracy': 0.9211, 'F1-Score': 0.9384},
            {'Split': '80:20', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9234, 'F1-Score': 0.9417},
            {'Split': '80:20', 'Model': 'Logistic Regression', 'Accuracy': 0.8959, 'F1-Score': 0.9164},
            {'Split': '80:20', 'Model': 'Support Vector Machine', 'Accuracy': 0.9031, 'F1-Score': 0.9219},
            {'Split': '80:20', 'Model': 'Decision Tree', 'Accuracy': 0.8002, 'F1-Score': 0.8191},
            
            {'Split': '75:25', 'Model': 'Random Forest', 'Accuracy': 0.9158, 'F1-Score': 0.9344},
            {'Split': '75:25', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9196, 'F1-Score': 0.9379},
            {'Split': '75:25', 'Model': 'Logistic Regression', 'Accuracy': 0.8842, 'F1-Score': 0.9061},
            {'Split': '75:25', 'Model': 'Support Vector Machine', 'Accuracy': 0.8986, 'F1-Score': 0.9182},
            {'Split': '75:25', 'Model': 'Decision Tree', 'Accuracy': 0.7952, 'F1-Score': 0.8136},
            
            {'Split': '70:30', 'Model': 'Random Forest', 'Accuracy': 0.9163, 'F1-Score': 0.9349},
            {'Split': '70:30', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9179, 'F1-Score': 0.9366},
            {'Split': '70:30', 'Model': 'Logistic Regression', 'Accuracy': 0.8876, 'F1-Score': 0.9088},
            {'Split': '70:30', 'Model': 'Support Vector Machine', 'Accuracy': 0.8931, 'F1-Score': 0.9135},
            {'Split': '70:30', 'Model': 'Decision Tree', 'Accuracy': 0.8038, 'F1-Score': 0.8232},
            
            {'Split': '65:35', 'Model': 'Random Forest', 'Accuracy': 0.9159, 'F1-Score': 0.9347},
            {'Split': '65:35', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9125, 'F1-Score': 0.9324},
            {'Split': '65:35', 'Model': 'Logistic Regression', 'Accuracy': 0.8818, 'F1-Score': 0.9048},
            {'Split': '65:35', 'Model': 'Support Vector Machine', 'Accuracy': 0.8872, 'F1-Score': 0.9082},
            {'Split': '65:35', 'Model': 'Decision Tree', 'Accuracy': 0.8038, 'F1-Score': 0.8229},
            
            {'Split': '60:40', 'Model': 'Random Forest', 'Accuracy': 0.9187, 'F1-Score': 0.9367},
            {'Split': '60:40', 'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9067, 'F1-Score': 0.9285},
            {'Split': '60:40', 'Model': 'Logistic Regression', 'Accuracy': 0.8810, 'F1-Score': 0.9039},
            {'Split': '60:40', 'Model': 'Support Vector Machine', 'Accuracy': 0.8834, 'F1-Score': 0.9048},
            {'Split': '60:40', 'Model': 'Decision Tree', 'Accuracy': 0.8002, 'F1-Score': 0.8195}
        ])
        
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            fig_decay = px.line(
                decay_data, x="Split", y="Accuracy", color="Model", markers=True,
                color_discrete_sequence=['#111111', '#555555', '#888888', '#B5B5B5', '#8C827A']
            )
            fig_decay.update_layout(
                yaxis_range=[0.78, 0.95], height=380, template="plotly_white",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", size=12, color="#111111")
            )
            st.plotly_chart(fig_decay, use_container_width=True)
            
        with dcol2:
            spread_data = pd.DataFrame([
                {'Model': 'Random Forest', 'Spread (%)': 0.92},
                {'Model': 'Decision Tree', 'Spread (%)': 1.18},
                {'Model': 'Logistic Regression', 'Spread (%)': 1.49},
                {'Model': 'K-Nearest Neighbors', 'Spread (%)': 1.83},
                {'Model': 'Support Vector Machine', 'Spread (%)': 2.25}
            ])
            fig_spread = px.bar(
                spread_data, x="Model", y="Spread (%)", color="Spread (%)",
                color_continuous_scale="Greys", text_auto=".2f"
            )
            fig_spread.update_layout(
                height=380, template="plotly_white",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", size=12, color="#111111")
            )
            st.plotly_chart(fig_spread, use_container_width=True)
            
    with tab2:
        st.subheader("4-Level Feature Ablation Progression")
        ablation_df = pd.DataFrame([
            {'Layer': 'Level 1: Majority Baseline', 'Accuracy (%)': 64.71, 'Macro-F1': 0.3929, 'Lift': '+0.00%'},
            {'Layer': 'Level 2: Category Only', 'Accuracy (%)': 83.73, 'Macro-F1': 0.8341, 'Lift': '+19.02%'},
            {'Layer': 'Level 3: Category + Metadata', 'Accuracy (%)': 88.28, 'Macro-F1': 0.8762, 'Lift': '+23.56%'},
            {'Layer': 'Level 4: Full Suite (+ Fabrics)', 'Accuracy (%)': 89.23, 'Macro-F1': 0.8857, 'Lift': '+24.52%'}
        ])
        
        fig_abl = px.bar(
            ablation_df, x="Layer", y="Accuracy (%)", color="Accuracy (%)",
            color_continuous_scale="Greys", text_auto=".2f"
        )
        fig_abl.add_hline(y=64.71, line_dash="dash", line_color="#111111", annotation_text="Baseline (64.71%)")
        fig_abl.update_layout(
            yaxis_range=[50, 100], height=400, template="plotly_white",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", size=12, color="#111111")
        )
        st.plotly_chart(fig_abl, use_container_width=True)
        st.dataframe(ablation_df, use_container_width=True)
        
    with tab3:
        st.subheader("5-Fold Stratified Cross-Validation Benchmark (Full N = 4,179)")
        cv_df = pd.DataFrame([
            {'Algorithm': 'HistGradientBoosting', 'Accuracy (Mean ± Std)': '91.17% ± 0.75%', 'Macro-F1': '0.9029 ± 0.0084', 'ROC-AUC': '0.9788 ± 0.0017'},
            {'Algorithm': 'Random Forest', 'Accuracy (Mean ± Std)': '90.69% ± 0.72%', 'Macro-F1': '0.8988 ± 0.0078', 'ROC-AUC': '0.9737 ± 0.0034'},
            {'Algorithm': 'K-Nearest Neighbors', 'Accuracy (Mean ± Std)': '90.24% ± 0.35%', 'Macro-F1': '0.8926 ± 0.0041', 'ROC-AUC': '0.9584 ± 0.0055'},
            {'Algorithm': 'Support Vector Machine', 'Accuracy (Mean ± Std)': '89.26% ± 1.14%', 'Macro-F1': '0.8866 ± 0.0125', 'ROC-AUC': '0.9641 ± 0.0044'},
            {'Algorithm': 'Logistic Regression', 'Accuracy (Mean ± Std)': '88.59% ± 0.70%', 'Macro-F1': '0.8785 ± 0.0079', 'ROC-AUC': '0.9729 ± 0.0030'},
            {'Algorithm': 'Decision Tree', 'Accuracy (Mean ± Std)': '80.16% ± 0.83%', 'Macro-F1': '0.7996 ± 0.0091', 'ROC-AUC': '0.8695 ± 0.0132'}
        ])
        st.dataframe(cv_df, use_container_width=True)

# ==============================================================================
# PAGE 5: CROSSOVER ERROR ANALYSIS & CATALOG EXPLORER
# ==============================================================================
elif page == "Crossover Diagnostics & Catalog Audit":
    st.markdown("""
    <div class="zara-header">
        <div class="zara-pretitle">CATALOG AUDIT / FAILURE MODE ANALYSIS</div>
        <h1 class="zara-title">CROSSOVER DIAGNOSTICS & SKU AUDIT</h1>
        <p class="zara-description">
            Quantifying classification boundaries and crossover ambiguity in unisex apparel departments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Error Summary by Department
    st.subheader("Department Categories with Highest Classification Ambiguity")
    err_cat_df = pd.DataFrame([
        {'Category': 'SHOES', 'Test Items Tested': 87, 'Misclassifications': 35, 'Error Rate (%)': '40.23%'},
        {'Category': 'SHIRTS', 'Test Items Tested': 71, 'Misclassifications': 31, 'Error Rate (%)': '43.66%'},
        {'Category': 'JACKETS', 'Test Items Tested': 24, 'Misclassifications': 9, 'Error Rate (%)': '37.50%'},
        {'Category': 'BLAZERS', 'Test Items Tested': 18, 'Misclassifications': 5, 'Error Rate (%)': '27.78%'},
        {'Category': 'BEAUTY', 'Test Items Tested': 6, 'Misclassifications': 1, 'Error Rate (%)': '16.67%'},
        {'Category': 'PERFUMES', 'Test Items Tested': 10, 'Misclassifications': 1, 'Error Rate (%)': '10.00%'}
    ])
    st.dataframe(err_cat_df, use_container_width=True)
    
    st.markdown("""
    **Commercial Diagnostic Takeaways:**
    * **Footwear & Basic Shirts:** Account for over **75% of misclassifications**. Unisex white sneakers, leather loafers, and oversized button-down shirts carry identical pricing and fabric mixtures across Zara Men and Women departments.
    * **100% Precision Departments:** Zero errors occurred on strictly gendered silhouettes such as `SKIRTS`, `DRESSES`, and `WAISTCOATS`.
    """)
    
    st.markdown("---")
    st.subheader("Interactive Catalog SKU Explorer")
    
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        f_gender = st.multiselect("Filter by Demographic:", ["Men", "Women"], default=["Men", "Women"])
    with fcol2:
        f_cat = st.multiselect("Filter by Category:", sorted(df['Category'].unique()), default=["JACKETS", "BLAZERS", "SHOES"])
    with fcol3:
        f_max_price = st.slider("Filter Max Price (₹):", 490, 18990, 8000, 500)
        
    filtered_df = df[
        (df['Gender'].isin(f_gender)) &
        (df['Category'].isin(f_cat)) &
        (df['Price_INR'] <= f_max_price)
    ]
    
    st.markdown(f"**Showing {len(filtered_df):,} matching products:**")
    st.dataframe(
        filtered_df[['Product_Name', 'Category', 'Gender', 'Price_INR', 'Details_Length', 'Num_Images', 'Price_Tier']],
        use_container_width=True
    )
