import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming you have the labeled dataset in a DataFrame
data = {
    "IP Detected": ["27.62.154.249", "27.15.128.195", "27.45.248.135"],
    "Payload Used": ["MS Excel Sheet", "DNS Token", "QR Code"],
    "Class": ["normal", "suspicious", "malicious"],
}

df = pd.DataFrame(data)

# Feature Engineering
# Here, for simplicity, we're just using IP addresses as numerical features
df["IP_numeric"] = df["IP Detected"].apply(
    lambda x: int("".join(filter(str.isdigit, x)))
)

# Encode categorical features (Payload)
df["Payload_numeric"] = LabelEncoder().fit_transform(df["Payload Used"])

# Prepare the feature matrix and target variable
X = df[["IP_numeric", "Payload_numeric"]]
y = LabelEncoder().fit_transform(df["Class"])

# Train a Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# Plot decision boundaries
x_min, x_max = X["IP_numeric"].min() - 1, X["IP_numeric"].max() + 1
y_min, y_max = X["Payload_numeric"].min() - 1, X["Payload_numeric"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    X["IP_numeric"],
    X["Payload_numeric"],
    c=y,
    edgecolors="k",
    marker="o",
    s=100,
    linewidth=1,
)
plt.title("Decision Boundaries of Classifier")
plt.xlabel("IP Address (numeric)")
plt.ylabel("Payload (numeric)")
plt.show()
