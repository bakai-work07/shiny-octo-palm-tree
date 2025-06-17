import torch
from torch import nn
import matplotlib.pyplot as plt

# Select GPU if available, else fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ground truth parameters for synthetic data
weight = 0.6
bias = 0.2

# Generate input data
start = 0
stop = 1
step = 0.02 # Shape: (N, 1)

x = torch.arange(start, stop, step).unsqueeze(1)
y = weight*x + bias # Linear relation: y = wx + b

# Train-test split (80% train, 20% test)
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Move data to device (GPU or CPU)
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(12, 8))

    # Always bring to CPU for plotting
    train_data = train_data.cpu()
    train_labels = train_labels.cpu()
    test_data = test_data.cpu()
    test_labels = test_labels.cpu()

    plt.scatter(train_data, train_labels, c='r', s=4, label='training data')
    plt.scatter(test_data, test_labels, c='b', s=4, label='test data')
    if predictions is not None:
        predictions = predictions.cpu()
        plt.scatter(test_data, predictions, c='g', s=4, label='predictions')

    plt.legend(loc='upper left')

# Define a simple linear regression model using PyTorch's nn.Module
class LinearRegressionModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # One input feature, one output

    def forward(self, x):
        return self.linear(x)

# Initialize model, loss function, and optimizer
model_2 = LinearRegressionModelV3().to(device)
loss_fn = nn.L1Loss().to(device)
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

# Training loop
epochs = 1000
def train_and_save_model():
    for epoch in range(epochs):
        model_2.train() # Set model to training mode
        # Forward pass
        y_pred = model_2(x_train)
        # Calculate loss between predictions and true labels
        loss = loss_fn(y_pred, y_train)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Evaluation mode (disable gradient calculation)
        model_2.eval()
        with torch.inference_mode():
            test_pred = model_2(x_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float32))
        # Print loss every 20 epochs
        if epoch % 20 == 0:
            print(f'epoch: {epoch}, train loss: {test_loss}, test loss: {test_loss}')

    # Save model weights to file
    torch.save(model_2.state_dict(), 'LinearRegressionModelV2.pth')
    print("Saved model")

# Load saved model and make predictions on test data
def load_and_predict():
    model_2 = LinearRegressionModelV3().to(device)
    model_2.load_state_dict(torch.load('LinearRegressionModelV2.pth'))
    model_2.eval()
    with torch.inference_mode():
        preds = model_2(x_test.to(device))
    plot_predictions(x_train, y_train, x_test, y_test, predictions=preds)
    plt.show()

# Main execution
if __name__ == "__main__":
    train_and_save_model()
    load_and_predict()
