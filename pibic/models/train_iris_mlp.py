import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Export weights to C header
weights = model.state_dict()
w1 = weights['fc1.weight'].numpy()
b1 = weights['fc1.bias'].numpy()
w2 = weights['fc2.weight'].numpy()
b2 = weights['fc2.bias'].numpy()

def format_array(arr):
    if len(arr.shape) == 1:
        return "{" + ", ".join([f"{x}f" for x in arr]) + "}"
    else:
        rows = []
        for row in arr:
            rows.append("    {" + ", ".join([f"{x}f" for x in row]) + "}")
        return "{\n" + ",\n".join(rows) + "\n}"

header_content = f"""#ifndef NN_WEIGHTS_IRIS_H
#define NN_WEIGHTS_IRIS_H

#define INPUT_SIZE 4
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 3

const float w1[HIDDEN_SIZE][INPUT_SIZE] = {format_array(w1)};

const float b1[HIDDEN_SIZE] = {format_array(b1)};

const float w2[OUTPUT_SIZE][HIDDEN_SIZE] = {format_array(w2)};

const float b2[OUTPUT_SIZE] = {format_array(b2)};

#endif // NN_WEIGHTS_IRIS_H
"""

header_path = os.path.join("verification", "nn_weights_iris.h")
with open(header_path, "w") as f:
    f.write(header_content)

print(f"Weights exported to {header_path}")

# Pick a test sample to represent the local point for robustness verification
sample_idx = 0
sample_input = X_test[sample_idx]
sample_pred = predicted[sample_idx].item()

print(f"Test Sample 0 input: {sample_input}")
print(f"Test Sample 0 predicted class: {sample_pred}")
