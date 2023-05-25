import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt

code_timer = time.perf_counter()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Define the model architecture
class NumberRecognitionModel(nn.Module):
    def __init__(self):
        super(NumberRecognitionModel, self).__init__()

        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc1(x)


# Set up the training and testing datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create an instance of the model
model = NumberRecognitionModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
# Training loop
for epoch in range(epochs):
    sum_loss = 0
    tic = time.perf_counter()
    print("Loop", epoch, "of", epochs)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
    toc = time.perf_counter()
    print(f"Finished epoch in {toc - tic:0.1f} seconds")
    print(sum_loss/len(train_loader))


# Plot a subset of images with labels and predictions
num_images = 10  # Number of images to plot
fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(6, 6))
model.eval()
with torch.no_grad():
    for i in range(num_images):
        data, target = test_dataset[i]
        data = data.unsqueeze(0).to(device)
        output = model(data)
        _, predicted = output.max(1)

        axes[i].imshow(data.squeeze().cpu(), cmap='gray')
        axes[i].set_title(f"Label: {target}, Predicted: {predicted.item()}")
        axes[i].axis('off')

plt.tight_layout()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "numbers.pth")

code_timer_end = time.perf_counter()
print(f"Code completed after {code_timer_end - code_timer:0.1f} seconds")
