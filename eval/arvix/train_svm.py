import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.data_loader import load_data
from src.data.data_splitting import split_data
import src.config as config
from src.core.utils import print_color


# Load and sample data
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), config.DATA_PATH))
df = load_data(DATA_PATH)
df_sampled = df.sample(frac=config.SAMPLE_RATIO, random_state=config.RANDOM_SEED)
train_df, val_df, _ = split_data(df_sampled, config.TEST_SIZE, config.VAL_SIZE)


# MultiLabelBinarizer
all_labels = set(label for labels in df["labels"] for label in labels)
mlb = MultiLabelBinarizer(classes=list(all_labels))
mlb.fit([list(all_labels)])
y_train = mlb.transform(train_df["labels"])
y_val = mlb.transform(val_df["labels"])


# TF-IDF + One-vs-Rest Logistic Regression
X_train = train_df["text"].values
X_val = val_df["text"].values

svm_model = make_pipeline(
    TfidfVectorizer(max_features=5000),
    OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="lbfgs"))
)

svm_model.fit(X_train, y_train)
val_preds = svm_model.predict(X_val)
micro_f1 = f1_score(y_val, val_preds, average="micro")

print_color(f"SVM Micro-F1 on validation: {micro_f1:.4f}", "GREEN")

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
