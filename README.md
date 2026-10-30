# Real Estate Buyer Segmentation

### Machine Learning-Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## 1. Project Overview

This project applies machine learning techniques to segment real estate buyers based on demographic and behavioural characteristics.

The objective is to identify meaningful buyer groups that can help support real estate market analysis and customer profiling.

## 2. Dataset

The dataset contains 2,000 buyer records and includes:

* Client type
* Age
* Gender
* Country and region
* Acquisition purpose
* Loan application status
* Referral channel
* Satisfaction score

The satisfaction score was excluded from model training and used later for cluster profiling.

## 3. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* K-Means Clustering

## 4. Methodology

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Age feature engineering
5. Categorical feature encoding
6. Feature scaling
7. K-Means clustering
8. Cluster evaluation using silhouette score
9. Comparison with Agglomerative Clustering
10. Buyer segment profiling

## 5. Model Results

The selected K-Means candidate used:

* Number of clusters: 2
* Input features: 22
* Number of buyers: 2,000
* Silhouette score: 0.1928

### Cluster Profiles

| Metric                 | Cluster 0 | Cluster 1 |
| ---------------------- | --------: | --------: |
| Buyer count            |     1,030 |       970 |
| Buyer percentage       |     51.5% |     48.5% |
| Average age            |     70.32 |     40.64 |
| Average satisfaction   |      3.03 |      3.03 |
| Home purpose (%)       |     72.23 |     66.08 |
| Investment purpose (%) |     27.77 |     33.92 |

Cluster 0 represents the older buyer group, while Cluster 1 represents the younger buyer group.

Cluster 1 has a higher proportion of investment-purpose buyers in this dataset.

## 6. Important Limitations

* The silhouette score of 0.1928 indicates relatively weak cluster separation.
* Clustering identifies patterns, not guaranteed buyer behaviour.
* The dataset does not contain income, property value, or investment amount.
* The clusters should not be interpreted as measures of buyer wealth or investment capacity.

## 7. Future Improvements

* Develop an interactive Streamlit dashboard.
* Explore additional clustering algorithms.
* Improve cluster quality through feature engineering.
* Expand the analysis with richer real estate market data.

## 8. Project Files

* `RealEstateBuyerSegmentation.ipynb` — Complete machine learning analysis.
* `clients.csv` — Dataset used for the project.

## 9. Author

**Amandeep Singh**

Machine Learning Project | Real Estate Market Intelligence

