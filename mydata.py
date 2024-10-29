import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Grumbullimi dhe sistemimi i raw data
# Këtu përdorim një dataset të thjeshtë për klasifikim
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, names=columns)

# 2. Dizajnimi i dataset-it fillestar
print("Dataset fillestar:")
print(data.head())

# 3. Paraprocesimi i të dhënave
# Ndajmë atributet (features) nga klasa që duam të klasifikojmë
X = data.drop('class', axis=1)
y = data['class']

# Ndarja në set trajnimi dhe testimi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizimi i të dhënave
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Testimi i algoritmeve

# Algoritmi 1: Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

# Algoritmi 2: K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 5. Analiza e rezultateve
# Matja e saktësisë për secilin algoritëm
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\nRezultatet për Decision Tree:")
print(f"Saktësia: {accuracy_dt:.2f}")
print(classification_report(y_test, y_pred_dt))

print("\nRezultatet për K-Nearest Neighbors:")
print(f"Saktësia: {accuracy_knn:.2f}")
print(classification_report(y_test, y_pred_knn))

# Matrica e konfuzionit për secilin algoritëm
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, cmap="Blues", fmt='g')
plt.title('Matrica e konfuzionit - Decision Tree')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, cmap="Blues", fmt='g')
plt.title('Matrica e konfuzionit - KNN')

plt.tight_layout()
plt.show()

