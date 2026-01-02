# diabetes
Project OverviewThis repository contains a machine learning pipeline designed to predict whether a patient has diabetes based on specific diagnostic measurements. This project utilizes the Pima Indians Diabetes Dataset to train a predictive model that can assist in early screening and risk assessment.Shutterstock📊 Dataset FeaturesThe model analyzes the following physiological factors:Pregnancies: Number of times pregnant.Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test).Blood Pressure: Diastolic blood pressure (mm Hg).Skin Thickness: Triceps skin fold thickness (mm).Insulin: 2-hour serum insulin (mu U/ml).BMI: Body mass index (weight in $kg/(height\ in\ m)^2$).Diabetes Pedigree Function: A function that scores likelihood of diabetes based on family history.Age: Age in years.📂 Repository StructurePlaintext├── data/
│   └── diabetes.csv          # Raw dataset
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── train_model.py        # Model training logic
│   └── utils.py              # Data cleaning helpers
├── models/
│   └── diabetes_model_v1.pkl # Saved 83% accuracy model
├── requirements.txt
└── README.md
⚙️ Implementation & PerformanceThe core of this project is a Logistic Regression or Support Vector Machine (SVM) model, optimized to reach a predictive value of 83%.
