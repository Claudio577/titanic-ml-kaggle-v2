import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Iniciando script Titanic ML...")

# ===============================
# Caminhos absolutos (SOLUÇÃO FINAL)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

print("Base dir:", BASE_DIR)
print("Train path:", TRAIN_PATH)
print("Test path:", TEST_PATH)

# ===============================
# 1. Carregar dados
# ===============================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Dados carregados com sucesso!")

# ===============================
# 2. Limpeza básica
# ===============================
for df in [train, test]:
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# ===============================
# 3. Feature Engineering
# ===============================
for df in [train, test]:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ===============================
# 4. Encoding
# ===============================
train = pd.get_dummies(train, columns=["Sex", "Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex", "Embarked"], drop_first=True)

# ===============================
# 5. Separar features e target
# ===============================
X = train.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)
y = train["Survived"]

X_test = test.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)

# ===============================
# 6. Treinar modelo
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 7. Avaliação
# ===============================
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Acurácia de validação: {accuracy:.4f}")

# ===============================
# 8. Gerar submissão Kaggle
# ===============================
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")
submission.to_csv(SUBMISSION_PATH, index=False)

print("Arquivo submission.csv gerado com sucesso!")
print("Local:", SUBMISSION_PATH)
print("Script finalizado.")
