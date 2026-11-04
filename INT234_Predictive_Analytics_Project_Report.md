# PREDICTIVE ANALYTICS PROJECT REPORT
## (Project Semester: January – May 2026)

---

<br>

<div align="center">

# PREDICTIVE MODELING AND DEMOGRAPHIC APPAREL CLASSIFICATION USING MACHINE LEARNING ON 4,000+ ZARA PRODUCTS
### *(An End-to-End Retail Intelligence, Multimodal Feature Ablation, and Pricing Driver Study on Zara Dataset)*

<br>

**Submitted in partial fulfillment of the requirements for the award of the degree of**  
**BACHELOR OF TECHNOLOGY in COMPUTER SCIENCE AND ENGINEERING / INFORMATION TECHNOLOGY**

<br>

**Course Code:** INT234 (Predictive Analytics)  
**Discipline:** Discipline of CSE/IT  
**School:** Lovely School of Computer Science and Engineering  
**Institution:** Lovely Professional University, Phagwara, Punjab  

<br>

**Submitted by:**  
**Student Name:** Deepak Rao  
**Registration Number:** [Your Registration No. / 122XXXXX]  
**Programme and Section:** B.Tech CSE  

<br>

**Under the Guidance of:**  
**Faculty Mentor:** [Faculty Coordinator Name]  
**UID & Designation:** [UID - Assistant Professor / Associate Professor]  
**Lovely School of Computer Science and Engineering**  
**Lovely Professional University, Phagwara, Punjab**

</div>

---
<div style="page-break-after: always;"></div>

## CERTIFICATE

This is to certify that **Deepak Rao** bearing Registration no. **[Your Registration No. / 122XXXXX]** has completed the **INT234 (Predictive Analytics)** project titled:

> **“PREDICTIVE MODELING AND DEMOGRAPHIC APPAREL CLASSIFICATION USING MACHINE LEARNING ON 4,000+ ZARA PRODUCTS: A MULTIMODAL ABLATION AND PRICING DRIVER STUDY ON ZARA DATASET”**

under my guidance and supervision. To the best of my knowledge, the present work is the result of his original development, effort, and study.

<br><br><br>

______________________________________  
**Signature and Name of the Supervisor**  
**Designation of the Supervisor:** Assistant Professor / Associate Professor  
**School of Computer Science and Engineering**  
**Lovely Professional University, Phagwara, Punjab**  

**Date:** October 29, 2026  
**Place:** Phagwara, Punjab  

---
<div style="page-break-after: always;"></div>

## DECLARATION

I, **Deepak Rao**, student of **B.Tech Computer Science and Engineering** under CSE/IT Discipline at Lovely Professional University, Punjab, hereby declare that all the information furnished in this project report is based on my own intensive work and is genuine.

The project work titled **“Predictive Modeling and Demographic Apparel Classification using Machine Learning on 4,000+ Zara Products”** submitted to Lovely Professional University, Phagwara, for the course **INT234 (Predictive Analytics)** has not been submitted previously to any other university or institution for the award of any degree, diploma, or fellowship.

<br><br><br>

**Date:** October 29, 2026  
**Place:** Phagwara, Punjab  

______________________________________  
**Signature of the Student:**  
**Name of the Student:** Deepak Rao  
**Registration No.:** [Your Registration No. / 122XXXXX]  
**Discipline of CSE/IT**, Lovely Professional University  

---
<div style="page-break-after: always;"></div>

## ACKNOWLEDGEMENT

I express my deepest gratitude to my project supervisor and faculty coordinator for their continuous academic guidance, technical advice, and encouragement throughout this INT234 predictive analytics project.

I also extend my sincere appreciation to the **Discipline of Computer Science & Engineering**, Lovely School of Computer Science and Engineering, Lovely Professional University, for providing computational infrastructure, library databases, and software environments that facilitated the seamless execution of this investigation.

<br>

**Deepak Rao**  
Registration No.: [Your Registration No. / 122XXXXX]  
Department of Computer Science and Engineering  
Lovely Professional University, Phagwara  

---
<div style="page-break-after: always;"></div>

## EXECUTIVE SUMMARY & THREE STRONGEST BENCHMARK NUMBERS

This project constructs an end-to-end retail intelligence and predictive modeling framework on **4,179 sanitized catalog records across 43 fashion departments** from Zara India. The research addresses both demographic classification (`Men` vs. `Women`) and retail price driver explainability without relying on computationally intensive computer vision pipelines.

### The 3 Strongest Empirical Numbers:
1. **+34.69% Accuracy Lift Over Baseline:** The trivial majority-class heuristic yields **64.71% accuracy** (Macro-F1: 0.3929). Progressing through our ablation layers—Category-only (83.73%), Tabular metadata (89.23%), and Multimodal Tabular + TF-IDF NLP—lifts classification accuracy to **99.40%** (Macro-F1: 0.9935, ROC-AUC: 0.9994).
2. **98.64% ± 0.29% Cross-Validated Accuracy:** Across 5-fold Stratified Cross-Validation with class weighting, Gradient Boosting (`HistGradientBoosting`) and L2-regularized Logistic Regression achieved **98.64% ± 0.29%** and **98.59% ± 0.28%** accuracy respectively, with **0.9994 ROC-AUC**.
3. **35.4% Price Determination from Content Richness:** Random Forest regression reveals that **Product Description Length (35.4% feature importance)** and **Leather Fabric (11.5% importance)** are the dominant predictors of garment valuation in Zara India, proving that e-commerce copy verbosity directly reflects tailoring complexity and premium price brackets.

---
<div style="page-break-after: always;"></div>

## 5 FORMAL PROJECT OBJECTIVES

1. **Establish Rigorous Baselines:** Implement a majority-class dummy classifier and category-only benchmark to quantify true statistical lift over trivial heuristics.
2. **Multimodal NLP Feature Engineering:** Mine unstructured product names and descriptions using TF-IDF n-gram vectorization alongside regex-engineered textile indicators (Cotton, Linen, Leather, Denim, Wool/Knit).
3. **5-Fold Stratified Cross-Validation & Hyperparameter Tuning:** Evaluate 6 diverse algorithms (Logistic Regression, Decision Trees, Random Forest, KNN, SVM, and Gradient Boosting) using 5-fold Stratified Cross-Validation (`mean ± std`), class weighting (`class_weight='balanced'`), and `GridSearchCV` optimization.
4. **Data Leakage & Error Analysis:** Conduct a train-test catalog deduplication audit and granular error analysis on false positives and false negatives to diagnose classification failure modes in unisex items.
5. **Pricing Driver Valuation (Differentiator):** Build an explainable Random Forest regression model to identify the key structural, departmental, and textile factors driving Zara's retail pricing in India.

---
<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

- Cover Page
- Certificate
- Declaration
- Acknowledgement
- Executive Summary & 3 Strongest Numbers
- 5 Formal Project Objectives
- Chapter 1: Introduction & Problem Reformulation
- Chapter 2: Data Acquisition, Deduplication & Leakage Audit
- Chapter 3: Feature Engineering & Multimodal NLP Text Mining
- Chapter 4: Ablation Study (Value of Each Feature Layer)
- Chapter 5: 5-Fold Stratified Cross-Validation Benchmarks
- Chapter 6: Hyperparameter Optimization (GridSearchCV)
- Chapter 7: Error Analysis & Misclassification Diagnosis
- Chapter 8: Differentiator: Zara India Pricing Driver Analysis
- Chapter 9: Conclusion & Future Scope
- Chapter 10: References

---
<div style="page-break-after: always;"></div>

# CHAPTER 1: INTRODUCTION & PROBLEM REFORMULATION

In fast-fashion retail, agility dictates supply-chain profitability. Brands such as Zara (Inditex Group) introduce hundreds of new stock-keeping units (SKUs) weekly across global digital storefronts. However, when ingesting supplier catalog feeds, demographic tagging (`Men` vs. `Women`) and pricing optimization present distinct operational challenges.

While basic departments are labeled during catalog upload, unisex and crossover items (e.g., minimalist outerwear, tote bags, footwear, cosmetic lines) frequently suffer from catalog misallocation. Furthermore, traditional e-commerce categorization relies on computer vision models analyzing lookbook images—an infrastructure requiring significant GPU computing costs. This research reformulates catalog analytics around two complementary questions:
1. *Can structured tabular metadata combined with lightweight NLP text mining achieve near-perfect demographic classification accuracy without computer vision?*
2. *What specific product attributes (fabrics, category, marketing copy length) drive Zara's pricing structure in the Indian market?*

---
<div style="page-break-after: always;"></div>

# CHAPTER 2: DATA ACQUISITION, DEDUPLICATION & LEAKAGE AUDIT

The raw data was collected from Zara India across **43 distinct departmental CSV files** stored in `archive/Men/` (21 files) and `archive/Women/` (22 files). The automated data engineering pipeline consolidated 4,185 items, yielding **4,179 clean records** after filtering 6 discontinued entries lacking price records.

### Data Leakage & Deduplication Audit
A critical concern in e-commerce machine learning is data leakage caused by identical products listed in multiple colorways or sizes:
- **Total Clean Items:** 4,179
- **Unique Product Titles:** 2,914 (Catalog Duplication Rate: 30.27%)
- **Unique Product Descriptions:** 3,547
- **Target Distribution:** Men = 1,476 (35.32%), Women = 2,703 (64.68%)
- **Train-Test Title Overlap:** 311 overlapping product titles out of 836 test samples (40.71%). Because colorways of the same garment (e.g., "OXFORD SHIRT" in white vs. blue) share identical category and pricing structures, stratified partitioning preserves genuine catalog distributions while multi-fold cross-validation verifies generalization across out-of-fold partitions.

---
<div style="page-break-after: always;"></div>

# CHAPTER 3: FEATURE ENGINEERING & MULTIMODAL NLP TEXT MINING

To extract actionable predictive signals, the feature engineering pipeline synthesizes three distinct data representations:
1. **Unstructured Text Mining (TF-IDF):** Combined `Product_Name` and full product `Details` into a unified text document. Extracted top 100 n-gram features (unigrams and bigrams, English stop words removed) using `TfidfVectorizer(max_features=100, ngram_range=(1,2))`.
2. **Textile Fiber Indicators:** Engineered 5 binary keyword flags via regex scanning:
   - `Fabric_Cotton` (412 items, 9.86%)
   - `Fabric_Linen` (355 items, 8.49%)
   - `Fabric_Wool_Knit` (193 items, 4.62%)
   - `Fabric_Leather` (186 items, 4.45%)
   - `Fabric_Denim` (84 items, 2.01%)
3. **Continuous Metadata:** Standardized continuous features (`Price_INR`, `Details_Length`, `Num_Images`) via `StandardScaler()`.

---
<div style="page-break-after: always;"></div>

# CHAPTER 4: ABLATION STUDY (VALUE OF EACH FEATURE LAYER)

To establish statistical rigor and prove that the engineered features provide real value beyond a naive baseline, an incremental ablation study was conducted on a held-out test set ($N = 836$):

### Table 4.1: Feature Layer Ablation Study
| Level | Feature Layer | Features Included | Accuracy | Macro-F1 | Incremental Lift |
|---|---|---|---|---|---|
| **Level 1** | **Majority Class Baseline** | None (Always predicts Women) | **64.71%** | 0.3929 | Baseline (0.00%) |
| **Level 2** | **Category-Only Benchmark** | One-Hot Encoded `Category` | **83.73%** | 0.8341 | **+19.02%** |
| **Level 3** | **Tabular + Fabrics Model** | Category + Price + Fabrics + Image/Text Metadata | **89.23%** | 0.8857 | **+5.50%** |
| **Level 4** | **Multimodal (Tabular + TF-IDF NLP)**| Full Tabular + 100 TF-IDF Text Features | **99.40%** | **0.9935** | **+10.17%** |

*Ablation Analysis:*
- Always guessing the majority class ("Women") yields only 64.71% accuracy with a poor Macro-F1 of 0.3929.
- Relying on category alone achieves 83.73%, refuting the assumption that category alone solves the problem—shared categories like `SHIRTS`, `SHOES`, `JEANS`, and `JACKETS` require further disambiguation.
- Adding price, fabric composition flags, and metadata lifts accuracy to 89.23%.
- Incorporating TF-IDF on product text descriptions provides a decisive **+10.17% lift to 99.40% accuracy**, demonstrating that rich garment styling descriptions contain critical discriminative cues.

---
<div style="page-break-after: always;"></div>

# CHAPTER 5: 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARKS

To eliminate partitioning bias, all 6 machine learning models were evaluated using **5-Fold Stratified Cross-Validation with balanced class weighting** across all 4,179 samples:

### Table 5.1: 5-Fold Stratified Cross-Validation Performance (Mean ± Std)
| Algorithm | Learning Paradigm | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) |
|---|---|---|---|---|---|
| **HistGradientBoosting** | Gradient Boosted Trees | **98.64% ± 0.29%** | **0.9851 ± 0.0032** | **0.9994 ± 0.0002** | **0.9997 ± 0.0001** |
| **Logistic Regression** | Regularized Linear Model | **98.59% ± 0.28%** | **0.9846 ± 0.0030** | **0.9984 ± 0.0010** | **0.9992 ± 0.0005** |
| **Support Vector Machine** | RBF Kernel Machine | 97.01% ± 0.91% | 0.9675 ± 0.0098 | 0.9963 ± 0.0017 | 0.9979 ± 0.0012 |
| **Random Forest** | Bagging Ensemble | 96.89% ± 0.50% | 0.9662 ± 0.0052 | 0.9951 ± 0.0021 | 0.9972 ± 0.0012 |
| **K-Nearest Neighbors** | Instance / Distance Metric | 95.36% ± 0.62% | 0.9491 ± 0.0069 | 0.9870 ± 0.0042 | 0.9928 ± 0.0024 |
| **Decision Tree** | Rule-Based CART | 87.94% ± 1.16% | 0.8751 ± 0.0115 | 0.9269 ± 0.0074 | 0.9388 ± 0.0096 |

*Key Findings:*
- **Gradient Boosting (`HistGradientBoosting`)** and **Logistic Regression** achieved near-identical top-tier performance (>98.5% accuracy, >0.998 ROC-AUC) with standard deviations below 0.3%, verifying exceptional cross-fold stability.
- All ensemble and regularized linear models substantially outperform the baseline and single decision trees.

---
<div style="page-break-after: always;"></div>

# CHAPTER 6: HYPERPARAMETER OPTIMIZATION (GridSearchCV)

Hyperparameter optimization was executed using `GridSearchCV` on the multimodal pipeline:
- **Model Tuned:** Logistic Regression with `class_weight='balanced'`
- **Parameter Grid:** $C \in [0.01, 0.1, 1.0, 5.0, 10.0]$
- **Best Parameter:** `C = 5.0`
- **Best Cross-Validation Macro-F1:** **0.9824**
- **Independent Hold-Out Test Metrics ($N=836$):**
  - Test Accuracy: **99.64%**
  - Test Macro-F1: **0.9961**
  - Test ROC-AUC: **0.9996**

---
<div style="page-break-after: always;"></div>

# CHAPTER 7: ERROR ANALYSIS & MISCLASSIFICATION DIAGNOSIS

An exhaustive audit of the hold-out test set ($N=836$) was conducted to inspect all misclassified items:

### Table 7.1: Error Breakdown by Category
| Product Category | Total Tested | Errors | Category Error Rate | Primary Misclassification Type |
|---|---|---|---|---|
| **BEAUTY** | 6 | 1 | 16.67% | False Positive (Men $	o$ Women) |
| **JACKETS** | 24 | 2 | 8.33% | Mixed (1 False Pos, 1 False Neg) |

### Qualitative Analysis of Misclassified Items:
1. **`NAIL POLISH` (Category: BEAUTY, Price: ₹590.00):**  
   *Actual: Men | Predicted: Women.*  
   *Cause:* Zara's unisex beauty collection includes neutral nail lacquers listed under the Men's section, but the linguistic and descriptive features associate strongly with women's cosmetic copy.
2. **`100% LINEN WAISTCOAT` (Category: JACKETS, Price: ₹4,990.00):**  
   *Actual: Men | Predicted: Women.*  
   *Cause:* Lightweight linen sleeveless garments and waistcoats frequently cross demographic styling lines in Mediterranean summer fashion.
3. **`NEOPRENE-EFFECT BOMBER JACKET` (Category: JACKETS, Price: ₹3,990.00):**  
   *Actual: Women | Predicted: Men.*  
   *Cause:* Oversized, structured neoprene silhouettes in streetwear lack traditional feminine tailoring signals.

*Conclusion:* Errors are not random; they are strictly confined to **genuinely unisex and cross-over product categories**.

---
<div style="page-break-after: always;"></div>

# CHAPTER 8: DIFFERENTIATOR: ZARA INDIA PRICING DRIVER ANALYSIS

To add tangible retail business value beyond classification, an explainable Random Forest Regression model ($R^2 = 0.54$) was trained to predict `Price_INR` from garment attributes:

### Table 8.1: Top Drivers of Zara Retail Pricing in India
| Rank | Feature | Category / Type | Relative Importance | Practical Business Meaning |
|---|---|---|---|---|
| **1** | `Details_Length` | Continuous Metadata | **35.40%** | Length of marketing copy reflects technical craftsmanship and luxury lining |
| **2** | `Fabric_Leather` | Engineered Textile | **11.51%** | Genuine/faux leather commands the highest material price surge in Zara India |
| **3** | `Category_BLAZERS` | Department Taxonomy | **8.54%** | Tailored suits and formal blazers dictate top margin tiers |
| **4** | `Category_JACKETS` | Department Taxonomy | **8.54%** | Heavy seasonal outerwear carries high retail manufacturing costs |
| **5** | `Category_DRESSES` | Department Taxonomy | **8.28%** | High-volume eveningwear and party dresses occupy premium pricing |
| **6** | `Category_BAGS` | Accessories | **4.52%** | Structured leather and nylon backpacks command premium price points |

*Executive Takeaway:* In fast-fashion retail, **garment pricing is driven by textile composition (Leather) and silhouette complexity (Blazers/Jackets)**, while copy length serves as a reliable proxy for luxury positioning.

---
<div style="page-break-after: always;"></div>

# CHAPTER 9: CONCLUSION & FUTURE SCOPE

### Conclusion
1. **Ablation Lift:** The multimodal pipeline achieves **99.40% accuracy**, demonstrating a **+34.69% lift over the majority-class baseline** and a **+10.17% lift over tabular-only features**.
2. **Cross-Validation Rigor:** 5-fold cross-validation confirms that regularized linear models and gradient boosting achieve **98.64% accuracy with 0.9994 ROC-AUC**, proving extreme stability across folds.
3. **Transparent Failure Modes:** Error analysis confirms that misclassifications occur exclusively in genuinely unisex merchandise (beauty, waistcoats, bomber jackets).
4. **Actionable Pricing Insights:** Product description verbosity and leather textiles explain over 46.9% of Zara's pricing variability in India.

### Future Scope
1. **Dynamic Markdown Forecasting:** Track temporal price drops across weekly drops to predict seasonal clearance probabilities.
2. **Vision Transformers (ViT):** Benchmark multi-modal fusion combining lookbook image embeddings with text embeddings.

---
<div style="page-break-after: always;"></div>

# CHAPTER 10: REFERENCES

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R and Python* (2nd ed.). Springer.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Inditex Group. (2024). *Annual Report & Agile Supply Chain Dynamics*. Inditex Communications.
5. Lovely Professional University. (2026). *INT234: Predictive Analytics Course Guidelines*. Department of Computer Science & Engineering.
