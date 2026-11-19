# DatasetTest Overview

**Turn Data Into Intelligence.**

DatasetTest Overview is a Streamlit application for interactive dataset exploration,
preparation, statistical analysis, anonymous project workspaces, and
customer-churn classification. The current workspace includes an overview
dashboard, searchable data explorer, data preparation actions, interactive EDA
charts, a statistics lab, model evaluation, and session-scoped project history.

## Current capabilities

- Use the clearly labeled synthetic demo dataset or upload a CSV/XLSX file up to
  50 MB.
- Inspect dataset dimensions, missing cells, duplicates, types, and unique values.
- Search across dataset values and download the active dataset as CSV or Excel,
  including a column profile sheet in the Excel file.
- Impute feature missing values, drop missing/duplicate rows, cap feature
  outliers with IQR, standardize numeric features, and one-hot encode a category.
  Changes affect the active copy; the original upload remains available to reset.
- Build histograms, bar charts, boxplots, violin plots, scatter plots, and line
  charts from the active dataset. The dashboard also compares target shares
  across feature groups and provides candlesticks when four valid OHLC columns
  are available; it never invents market data.
- Calculate descriptive statistics, Pearson/Spearman/Kendall correlations, and
  run Shapiro-Wilk, Welch independent t-test, Mann-Whitney U, one-way ANOVA, and
  chi-square tests on selected columns.
- Train Logistic Regression and Random Forest classification pipelines, compare
  evaluation metrics, and download test-set predictions.
- Use the application without signing in; project workspaces are isolated to the
  current browser session.
- Choose dark, light, or system appearance.
- Manage session-scoped projects, favorites, and recent activities.

The demo data is synthetic and clearly labeled. Metrics and charts are calculated
from the active dataset; there are no placeholder analysis results.

No login is required. Project metadata is stored locally in
`data/statlytics.sqlite3`, while each browser session gets a separate workspace.
The database file is excluded from source control. A project stores its name and
activity history, not uploaded dataset or model files.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

For an uploaded CSV/XLSX, select the target column in the sidebar. Model Studio
currently supports classification targets with at least two classes and at least
two rows per class. Regression, clustering, association rules, and PDF reports
are not implemented yet. Dataset state remains in the current Streamlit session;
projects persist locally in SQLite.
