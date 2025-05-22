# gdsc

Open index.html as the main file
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Enable MLflow autologging for PyTorch
mlflow.pytorch.autolog()

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # output: 26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                  # output: 13x13
            nn.Conv2d(32, 64, kernel_size=3), # output: 11x11
            nn.ReLU(),
            nn.MaxPool2d(2)                   # output: 5x5
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Train and evaluate model inside MLflow run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with mlflow.start_run():
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
        mlflow.log_metric("train_loss", running_loss, step=epoch)

    # Evaluate on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Save a sample image
    plt.imshow(train_dataset[0][0].squeeze(), cmap="gray")
    plt.title(f"True Label: {train_dataset[0][1]}")
    plt.savefig("mnist_sample_pytorch.png")
    mlflow.log_artifact("mnist_sample_pytorch.png")

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "pytorch_mnist_model")