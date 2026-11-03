# MLE_CaseStudy_3
Deliverables

This folder deliberately contains exactly three submission files: this README, `MLE_CaseStudy_3.ipynb`, and `Online_Retail.csv`.

## Data provenance and integrity

`Online_Retail.csv` is the complete, unmodified 541,909-row CSV downloaded on 3 November 2026 from Kaggle's **Online Retail** dataset by Bojan Tunguz: <https://www.kaggle.com/datasets/tunguz/online-retail>. It is a real transactional dataset, not the earlier 42-row sample. Its columns are InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

The notebook loads only the CSV located alongside it. It removes rows without a customer ID, cancelled invoices (`InvoiceNo` beginning `C`), non-positive quantities, and non-positive prices; those rules are explicit so the analysis remains reproducible.

## Run instructions

Place the notebook and CSV in the same directory, then run all cells in Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` installed. t-SNE is intentionally applied to the customer-level RFM table rather than all transaction rows because it is a visualization, not a clustering feature.

## Method and code walkthrough

1. **Imports:** pandas reads and cleans transactions; NumPy supports transforms; scikit-learn supplies scaling, K-Means, DBSCAN, PCA, t-SNE and metrics.
2. **Load and clean:** `pd.read_csv` reads the full file; conversion of `InvoiceDate` makes time arithmetic valid; `filter` retains valid completed purchases.
3. **Revenue:** `TotalPrice = Quantity * UnitPrice` creates invoice-line value.
4. **RFM:** grouping by `CustomerID` calculates Recency (days from the latest transaction plus one day), Frequency (distinct invoices), and Monetary (sum of revenue).
5. **Transform and scale:** `log1p` reduces the long right tails of Frequency and Monetary; `StandardScaler` puts all three inputs on comparable scale.
6. **K-Means:** the notebook evaluates k=2..8 using inertia and silhouette score, then chooses the explicitly recorded selected k=4. Cluster means and sizes are printed.
7. **DBSCAN:** DBSCAN is run on the same scaled RFM features. The notebook reports its cluster count and noise share rather than forcing every customer into a segment.
8. **PCA and t-SNE:** PCA gives an interpretable two-dimensional linear view; t-SNE gives a local-neighbourhood diagnostic. Neither plot is used to claim causal separation.
9. **Personas:** labels are assigned by ranking K-Means cluster profiles, not by arbitrary row order. The recommended actions are: Champions—VIP rewards/cross-sell; Loyal—bundles/subscriptions; Potential—second-purchase offers; At-risk—time-limited win-back offers.

## Interpretation and limitations

Use the notebook's computed cluster profile table as the source of record. K-Means assumes roughly spherical segments and is sensitive to k; DBSCAN is sensitive to `eps`; t-SNE can change with its seed. This is behavioural segmentation, not demographic profiling. Customer IDs are pseudonymous and should be handled under the applicable data policy.
