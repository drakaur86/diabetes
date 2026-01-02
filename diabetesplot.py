import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Plotting Feature Importance
# This helps explain why the model makes certain decisions
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances for Diabetes Prediction', fontsize=14)
plt.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance (Weight)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance.png') # Saves to your repository folder

# 2. Plotting the Confusion Matrix
# This visualizes the True Positives vs False Positives
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Healthy', 'Diabetic'], 
            yticklabels=['Healthy', 'Diabetic'])
plt.ylabel('Actual Status', fontsize=12)
plt.xlabel('Predicted Status', fontsize=12)
plt.title(f'Confusion Matrix (Model Accuracy: {accuracy*100:.1f}%)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
