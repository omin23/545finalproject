# ========= 数据准备 ==========
from sklearn.model_selection import train_test_split
import numpy as np
import torch

X = np.load("outputs/mouse_X_cls.npy")
y = np.load("outputs/mouse_y_cls.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.3, random_state=42, stratify=y_test)

print(X_train.shape, X_test.shape, X_valid.shape, y_train.shape, y_test.shape, y_valid.shape)

# Class for classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(embedding_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)  # Project input to hidden dimension
        x = x.unsqueeze(1)  # Ensure shape is (batch_size, seq_length=1, hidden_dim)
        x = self.transformer_encoder(x)  # Transformer encoder expects (seq_length, batch_size, hidden_dim)
        x = x.squeeze(1)  # Remove extra dimension
        logits = self.fc(x)  # Aggregate features along embedding space
        return self.softmax(logits)

# Training function
def train_classifier(X_train, y_train, X_validation, y_validation, X_test, y_test, embedding_dim, num_classes, epochs=10, lr=0.001, batch_size=32):
    model = TransformerClassifier(embedding_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_validation, y_validation)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss / len(val_loader):.4f}')
    
    return model


# Prediction function
def predict(model, X_test):
    with torch.no_grad():
        outputs = model(X_test)
        predicted_labels = torch.argmax(outputs, dim=1)
    return predicted_labels

from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit on all labels and transform them
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_valid_encoded = label_encoder.transform(y_valid)

# Assuming your datasets are in NumPy arrays, convert them to PyTorch tensors
# Convert Pandas Series to NumPy arrays
X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
X_valid_tensor = torch.tensor(np.array(X_valid), dtype=torch.float32)

# Convert labels to NumPy arrays with explicit integer conversion
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
y_valid_tensor = torch.tensor(y_valid_encoded, dtype=torch.long)

# Define parameters
embedding_dim = X.shape[1]  # Based on input shape, i.e., gene number
#num_classes = len(torch.unique(y_train_tensor))  # Automatically infer number of classes
num_classes = len(np.unique(y)) # Because we have a total of 18 different cell types in our input data
epochs = 20  # Adjust based on training needs
lr = 0.001
batch_size = 32

# Train the model
model = train_classifier(
    X_train_tensor, y_train_tensor,
    X_valid_tensor, y_valid_tensor,
    X_test_tensor, y_test_tensor,
    embedding_dim, num_classes, 
    epochs=epochs, lr=lr, batch_size=batch_size
)

predicted_labels = predict(model, X_test_tensor)
#print("Predicted Labels:", predicted_labels)

# Compare true cell types with predicted ones for the test data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_curve, precision_score, recall_score

accuracy =accuracy_score(y_test_tensor, predicted_labels)
print(f"Accuracy: {accuracy}")

report= classification_report(y_test_tensor, predicted_labels)
print(f"Classification report: {report}")

# Calculate F1 score for multi-class classification with 'weighted' average
f1 = f1_score(y_test_tensor, predicted_labels, average='macro')
print(f"F1 Score (Macro): {f1}")

# Calculate F1 score for multi-class classification with 'micro' average
f1_micro = f1_score(y_test_tensor, predicted_labels, average='micro')
print(f"F1 Score (Micro): {f1_micro}")

precision_macro = precision_score(y_test_tensor, predicted_labels, average='macro', labels=range(18))
recall_macro = recall_score(y_test_tensor, predicted_labels, average='macro', labels=range(18))
f1_macro = f1_score(y_test_tensor, predicted_labels, average='macro', labels=range(18))
print(f"Test precision: {precision_macro: .3f}")
print(f"Test recall: {recall_macro: .3f}")
print(f"Test F1 score: {f1_macro: .3f}")

#Compare the true cell types with the predicted ones on validation data
valid_predicted_labels = predict(model, X_valid_tensor)
accuracy =accuracy_score(y_valid_tensor, valid_predicted_labels)
print(f"Accuracy: {accuracy}")

# Calculate F1 score for multi-class classification with 'weighted' average
f1 = f1_score(y_valid_tensor, valid_predicted_labels, average='macro')
print(f"F1 Score (Macro): {f1}")

# Calculate F1 score for multi-class classification with 'micro' average
f1_micro = f1_score(y_valid_tensor, valid_predicted_labels, average='micro')
print(f"F1 Score (Micro): {f1_micro}")

precision_macro = precision_score(y_valid_tensor, valid_predicted_labels, average='macro', labels=range(18))
recall_macro = recall_score(y_valid_tensor, valid_predicted_labels, average='macro', labels=range(18))
f1_macro = f1_score(y_valid_tensor, valid_predicted_labels, average='macro', labels=range(18))
print(f"Validation precision: {precision_macro: .3f}")
print(f"Validation recall: {recall_macro: .3f}")
print(f"Validation F1 score: {f1_macro: .3f}")

# ========= UMAP 可视化 ==========
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

os.makedirs("figures_0420", exist_ok=True)
X_all = np.vstack([X_train, X_test, X_valid])
y_all = np.concatenate([y_train, y_test, y_valid])

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
X_umap = reducer.fit_transform(X_all)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_all, palette="tab10", s=10, legend='full')
plt.title("UMAP of CLS Embeddings (All Mouse Data)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures_0420/umap_mouse.svg", format ="svg")
plt.close()
print("UMAP embedding figure saved to: figures_0420/umap_mouse.svg")




