import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score

# ----------------------------
# 1️⃣ Load processed dataset
# ----------------------------
processed_path = r'D:\FactoryGuard-AI\Data\Processed\sensor_data_engineered.csv'
df = pd.read_csv(processed_path)

# ----------------------------
# 2️⃣ Choose target column
# ----------------------------
target_column = 'label'  # classification target

# ----------------------------
# 3️⃣ Select numeric features only
# ----------------------------
X = df.select_dtypes(include='number').drop(columns=[target_column])
y = df[target_column]

# ----------------------------
# 4️⃣ Split data into train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5️⃣ Train Random Forest Classifier
# ----------------------------
clf = RandomForestClassifier(
    n_estimators=100,       # number of trees
    random_state=42,
    class_weight='balanced' # handle imbalanced classes
)
clf.fit(X_train, y_train)

# ----------------------------
# 6️⃣ Make predictions
# ----------------------------
y_pred = clf.predict(X_test)

# ----------------------------
# 7️⃣ Evaluate classification metrics
# ----------------------------
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Classification Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
