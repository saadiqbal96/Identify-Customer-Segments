Udacity Data Science Nanodegree

This repository contains my implementation of the Identify Customer Segments project from Udacity’s Data Science Nanodegree.
The goal of this project is to analyze real-world demographic data provided by Arvato Financial Services and identify customer segments using unsupervised machine learning techniques.

📁 Repository Structure
Identify-Customer-Segments/
├── data/                        # (empty - data is proprietary and cannot be uploaded)
├── notebooks/
│   └── Identify_Customer_Segments.ipynb
├── html/
│   └── Identify_Customer_Segments.html
├── src/
│   ├── preprocessing.py
│   ├── pca_analysis.py
│   ├── clustering.py
│   └── utils.py
├── images/
├── README.md
└── requirements.txt

📌 Project Overview

The project focuses on answering:

Which segments of the German population are most likely to be customers of a mail-order company?

To achieve this, the project uses:

1. Preprocessing

Standardizes missing values

Cleans and re-encodes categorical and mixed-type features

Removes high-missing-value features

Splits rows with extreme missingness

Creates a reusable preprocessing pipeline

2. Feature Transformation

Scales all numerical features

Applies PCA to reduce dimensionality

Analyzes explained variance

Interprets principal components

3. Clustering

Uses KMeans to find demographic clusters in the general population

Determines optimal k via inertia/Elbow analysis

Applies same pipeline to customer data

Compares population vs. customer cluster distributions

Identifies overrepresented customer segments

📊 Technologies Used

Python 3

NumPy

pandas

scikit-learn

matplotlib

seaborn

Google Colab

🚫 About the Data

The datasets used in this project are proprietary and provided under strict agreement with Udacity, AZ Direct, and Arvato.
Therefore, raw data files are not included in this repository.

If you're a Udacity student, you can obtain the files from the course workspace.

▶️ How to Run This Project

Clone this repository:

git clone https://github.com/your-username/Identify-Customer-Segments.git


Upload the Udacity data (if you have access) into the /data folder.

Open the notebook in Google Colab or Jupyter:

Colab: File > Upload notebook

Jupyter: jupyter notebook notebooks/Identify_Customer_Segments.ipynb

Run cells in order.

Export as HTML via:

File > Download as > HTML
