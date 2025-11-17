# Identify Customer Segments  
_Unsupervised Learning with PCA + KMeans_

This project applies unsupervised machine learning techniques to identify customer segments for a mail-order company in Germany. Using demographic data provided by Arvato/Bertelsmann, the goal is to understand which population groups are overrepresented (or underrepresented) among the company's customers, supporting improved marketing targeting.

This project is part of the Udacity Data Scientist Nanodegree.

---

## 🚀 Project Overview

The project uses **real-world demographic data** for:
- The **general German population** (AZDIAS dataset)
- **Existing customers** of a mail-order company

Because the data has no labels, the approach is fully **unsupervised**.

### The main workflow:
1. **Preprocessing**
   - Handle missing values (custom unknown codes → NaN)
   - Remove high-missingness features
   - Handle categorical & mixed-type variables
   - Engineer new features (PRAEGENDE_JUGENDJAHRE, CAMEO_INTL_2015)

2. **Feature Transformation**
   - Median imputation  
   - Feature scaling using `StandardScaler`
   - Dimensionality reduction using **PCA (30 components)**

3. **Clustering**
   - KMeans evaluated for K = 2–20
   - **12 clusters** selected based on elbow behavior
   - Customer data transformed using the *same* preprocessing pipeline
   - Cluster distribution compared between population vs customers

4. **Interpretation**
   - Identify overrepresented customer groups  
   - Provide demographic insights for marketing targeting  

---

## 🧠 Key Findings

### ✅ **Overrepresented Customer Clusters**
Customers appear disproportionately in:
- **Cluster 0**
- **Cluster 2**
- **Cluster 8**

These segments tend to represent:
- Urban / suburban regions  
- Higher mobility  
- Stronger consumption and lifestyle signals  
- Higher marketing responsiveness  

### ❌ **Underrepresented Clusters**
- Clusters **3, 5, 9, 10, 11**

These segments often align with:
- Rural or low-density households  
- More traditional / less consumption-oriented behavior  
- Lower digital engagement  
- Weaker interest in mail-order retail  

### 🎯 **Conclusion**
For optimized marketing campaigns, the company should **focus on clusters 0, 2, and 8**, while deprioritizing outreach to underrepresented clusters.

---

## 📁 Repository Structure

├── data/ # (Not included – restricted dataset)
├── src/
│ ├── preprocessing.py # Data cleaning utilities
│ ├── pca_analysis.py # Scaling + PCA functions
│ └── clustering.py # KMeans helpers
├── Identify_Customer_Segments.ipynb # Main Jupyter Notebook
├── Identify_Customer_Segments.html # Exported HTML report
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Usage

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Identify-Customer-Segments.git
cd Identify-Customer-Segments
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Add the datasets
Due to data licensing restrictions, the Arvato datasets cannot be included in this repository.
Download them from Udacity and place them in the data/ folder:

kotlin
Copy code
data/
├── Udacity_AZDIAS_Subset.csv
├── Udacity_CUSTOMERS_Subset.csv
├── AZDIAS_Feature_Summary.csv
└── Data_Dictionary.md
4. Run the notebook
bash
Copy code
jupyter notebook Identify_Customer_Segments.ipynb
🛑 Data Usage Restrictions (IMPORTANT)
These datasets are proprietary and provided exclusively for educational use under the Udacity–Bertelsmann agreement.
You must not upload the data publicly to GitHub or elsewhere.

Your repository should not contain any data files.

🧰 Tools & Libraries
Python 3.8+

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook

📊 Techniques Used
Data Wrangling
Converting coded missing values

Removing high-missingness features

Handling categorical + mixed-type features

Feature engineering for socioeconomic attributes

Unsupervised Learning
StandardScaler

PCA (30 components retained)

KMeans clustering (K = 12)

Evaluation
Explained variance analysis

Elbow method for cluster selection

Cluster frequency comparisons

Interpretation of principal components

📦 Results Summary
This analysis successfully identifies the most valuable demographic segments for a large mail-order retailer. PCA effectively compresses the complex socioeconomic data into meaningful components, and clustering reveals clear distinctions between the general population and actual customers.

The strongest customer segments correspond to:

Urban households

Higher mobility and consumer affinity

Younger–middle-age demographic groups

More modern lifestyle patterns

These insights can support targeted marketing, acquisition strategies, and customer retention efforts.
