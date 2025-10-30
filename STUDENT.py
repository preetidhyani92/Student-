# ===============================================================
# 🎓 STUDENT PERFORMANCE PREDICTION PROJECT
# Author: Preeti Dhyani
# ===============================================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Create Sample Dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 96, 98],
    'Test_Score': [50, 55, 60, 65, 70, 75, 80, 85, 88, 90],
    'Result': ['Fail', 'Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)
print("📊 Dataset:\n", df)

# 3️⃣ Split Data
X = df[['Study_Hours', 'Attendance', 'Test_Score']]
y = df['Result']

# Encode target variable
y = y.map({'Fail': 0, 'Pass': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5️⃣ Predict
y_pred = model.predict(X_test)

# 6️⃣ Evaluate Model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Visualization
plt.scatter(df['Study_Hours'], df['Test_Score'], c=y, cmap='coolwarm')
plt.title("Study Hours vs Test Score (Pass=1, Fail=0)")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.show()

# 8️⃣ Predict for New Student
new_data = pd.DataFrame({'Study_Hours':[7], 'Attendance':[92], 'Test_Score':[83]})
prediction = model.predict(new_data)
result = 'Pass' if prediction[0] == 1 else 'Fail'
print(f"\n📘 Prediction for new student: {result}")
