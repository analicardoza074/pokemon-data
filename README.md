Pokémon Legendary Status Classifier 🏆
A machine learning project that predicts whether a Pokémon is Legendary using battle stats and characteristics — trained across four different models including a custom PyTorch neural network.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/analicardoza074/pokemon-data/blob/main/Take2.ipynb)

Goal
Out of 801 Pokémon, only 8.7% are Legendary — making this an imbalanced binary classification problem. The goal was to build a model that could reliably identify Legendary Pokémon based on numerical features like HP, attack, capture rate, and more.

Dataset

Source: Kaggle Pokémon Dataset (loaded via raw GitHub CSV)
Size: 801 Pokémon × 41 features
Target: is_legendary (0 = Regular, 1 = Legendary)

Features Used
CategoryFeaturesBattle Statshp, attack, defense, sp_attack, sp_defense, speedSummarybase_total (sum of all battle stats)Game Mechanicscapture_rate, base_experience, base_egg_steps, base_happinessPhysicalheight_m, weight_kg, generationType Matchupsagainst_* (18 type effectiveness columns)

Models & Results
ModelTest AccuracyNaive Baseline (always Regular)90.9%Logistic Regression99.2%Random Forest99.2%SVM97.5%Neural Network (PyTorch)98.3%Neural Network ROC AUC0.9967

The neural network used inverse-frequency class weighting to handle the imbalanced dataset (only 8.7% Legendary).


Project Highlights

EDA & Visualization — Box plots and distribution charts showing the clear stat gap between Legendary and Regular Pokémon
Data Cleaning — Handled mixed-type capture_rate column and missing values in height/weight
Model Comparison — Trained and evaluated four different classifiers with full classification reports, confusion matrices, and ROC curves
Interactive Quiz — A fun end feature: answer questions about yourself and the neural network predicts which Pokémon you'd be matched with

How to Run

Click the Open in Colab badge above
Run all cells top to bottom (Runtime > Run all)
No API keys or tokens needed — the dataset loads directly from this repo

Tech Stack

Language: Python
ML Libraries: scikit-learn, PyTorch
Data: pandas, NumPy
Visualization: Matplotlib, Seaborn
Environment: Google Colab / Jupyter Notebook


Author
Anali Cardoza — CS Senior @ CSU Channel Islands
