# Explainable Tabular Credit Scoring with SHAP + TabNet

Overview
This project integrates explainable deep learning with financial risk modeling to build an interpretable credit scoring system. Using TabNet, a deep tabular learning architecture, and SHAP for posthoc interpretability, the system delivers transparent, regulatorfriendly credit risk predictions while maintaining high accuracy and compliance with financial governance standards.

Framework
Models: TabNet, XGBoost (baseline)
Libraries: PyTorch, PyTorch TabNet, SHAP, Scikitlearn, Pandas, NumPy

Scope
 Design an interpretable credit risk prediction pipeline.
 Apply TabNet for attentiondriven feature selection.
 Generate local and global SHAP explanations.
 Benchmark TabNet against traditional ML models.
 Ensure explainability for regulatory compliance (e.g., Basel III, GDPR).

Datasets Used:
1. LendingClub Loan Data — ~1M records, 150+ borrower and loan attributes.
2. German Credit Dataset — 1,000 records, 20 features (categorical + numerical).

Preprocessing Steps:
 Imputation for missing financial fields.
 Label encoding and feature scaling.
 Handling class imbalance via SMOTE and undersampling.
 Feature pruning using mutual information and variance thresholding.

Methodology

 1. Data Preprocessing

 Unified schema across datasets for model interoperability.
 Converted categorical variables to embeddings for TabNet input.

 2. Model Architecture (TabNet)

 Feature transformer + attentive transformer blocks.
 Sequential attention masks to select relevant features per sample.
 Sparse regularization for interpretability.

 3. Model Training

 Optimizer: AdamW
 Learning rate scheduling with cosine annealing.
 Crossvalidation (5 folds) to ensure generalization.

 4. Explainability with SHAP

 Applied SHAP (TreeExplainer + DeepExplainer) to quantify feature impacts.
 Generated global importance plots and local (instancelevel) explanations.
 Highlighted financial variables influencing decisions (e.g., income, loan amount, credit history).

Pipeline Architecture (Textual Diagram)
      ┌───────────────────────┐
      │ Credit Risk Data      │
      │ (LendingClub/German)  │
      └──────────┬────────────┘
                 │
      ┌──────────▼────────────┐
      │ Preprocessing          │
      │ (Encoding, Balancing)  │
      └──────────┬────────────┘
                 │
      ┌──────────▼────────────┐
      │ TabNet Model          │
      │ (Attentionbased DL)  │
      └──────────┬────────────┘
                 │
      ┌──────────▼────────────┐
      │ SHAP Analysis          │
      │ (Explainability Layer) │
      └──────────┬────────────┘
                 │
      ┌──────────▼────────────┐
      │ Model Insights &       │
      │ Compliance Reporting   │
      └────────────────────────┘

Results
| Model               | Accuracy | F1Score  | ROCAUC   | Precision | Recall   |
| Logistic Regression | 0.86     | 0.82     | 0.88     | 0.81      | 0.83     |
| XGBoost             | 0.90     | 0.88     | 0.91     | 0.89      | 0.87     |
| TabNet (Ours)       | 0.93     | 0.91     | 0.92     | 0.92      | 0.90     |

Interpretability Highlights:
 Top predictors: CreditHistory, LoanAmount, DebttoIncome Ratio, AnnualIncome.
 SHAP visualizations demonstrated individual borrowerlevel decisions.
 Regulators can trace each prediction to its influencing features.

Conclusion
The TabNet + SHAP framework achieved 0.92 ROCAUC, outperforming traditional models while maintaining full interpretability. It bridges the gap between accuracy and transparency in financial AI — ensuring compliance with explainability mandates and ethical AI governance.
Core Impact: Trustworthy, auditable credit scoring with explainable deep learning.

Future Work
 Integrate counterfactual explanations (using DiCE) for “whatif” scenario testing.
 Expand to multicountry credit datasets for regulatory generalization.
 Deploy as a modelasaservice (MaaS) dashboard for financial institutions.
 Extend interpretability layer with Causal SHAP for counterfactual reasoning.

References
1. Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
2. Lundberg, S. M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier. KDD.
4. German Credit Dataset — UCI Repository, 2020.
5. LendingClub Loan Data — Kaggle, 2021.

Closest Research Paper:
> “Interpretable Deep Learning for Credit Risk Assessment using TabNet and SHAP” — Expert Systems with Applications, 2023.
> This paper aligns directly with the project’s focus on interpretable tabular deep learning for credit scoring.
