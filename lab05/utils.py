from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset


# plot fit for trends

def log_model(x, a, b, c):
    return a + b * np.log(x + c)

def plot_log_fit(running_losses):
    with torch.no_grad():
        epochs = list(range(len(running_losses)))
        # Prepare x and y data, shift x data to ensure positivity 
        x_data = np.array(epochs) + 1  # +1 to avoid log(0) which is undefined
        y_data = np.array(running_losses)
        # Fit the model
        params, params_covariance = curve_fit(log_model, x_data, y_data, p0=[1, 1, 1])
        # Generate x values for the curve
        x_fit = np.linspace(1, len(running_losses), 400)
        y_fit = log_model(x_fit, *params)
        # Plotting the data
        plt.figure(figsize=(10, 5))
        plt.scatter(epochs, running_losses, label='Individual Losses', alpha=0.6)  # Alpha for better visibility of the trend
        plt.plot(x_fit - 1, y_fit, label='Logarithmic Fit', color='red')  # Plot the fitted curve
        # Add labels, title, and grid
        plt.title('Training Loss with Logarithmic Fit')
        plt.xlabel('Batch Index (x1000)')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.legend()  # Show legend
        plt.show()

def plot_moving_average(running_losses):
    with torch.no_grad():
        plt.figure(figsize=(10, 5))
        # Data points
        epochs = list(range(len(running_losses)))
        plt.scatter(epochs, running_losses, label='Individual Losses')
    
        # Trend - Using a simple moving average
        window_size = 3 # Adjust window size according to your preference
        cumulative_sum = np.cumsum(np.insert(running_losses, 0, 0)) 
        moving_averages = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        plt.plot(epochs[window_size - 1:], moving_averages, color='red', label='Moving Average Trend')
    
        # Add labels and title
        plt.title('Training Loss per 1000 Batches and Trend')
        plt.xlabel('Batch Index (x1000)')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.legend()  # Show legend to identify scatter and line plot
    
        plt.show()

# fit poly expression
def plot_poly_fit(running_losses, deg):
    with torch.no_grad():   
        plt.figure(figsize=(10, 5))
        
        # Data points
        epochs = list(range(len(running_losses)))
        plt.scatter(epochs, running_losses, label='Individual Losses', alpha=0.6)  # Alpha for better visibility of the trend
    
        # Fit a third-degree (cubic) polynomial
        coeffs = np.polyfit(epochs, running_losses, deg)
        poly = np.poly1d(coeffs)
        
        # Generate x-values for plotting the polynomial line
        x_poly = np.linspace(0, len(running_losses)-1, num=len(running_losses))
        y_poly = poly(x_poly)
        
        plt.plot(x_poly, y_poly, color='red', label='Cubic Polynomial Trend')
    
        # Add labels, title, and grid
        plt.title('Training Loss per 1000 Batches with Cubic Polynomial Trend')
        plt.xlabel('Batch Index (x1000)')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.legend() 
        plt.show()


## Confusion matrix and predictions

def get_all_predictions(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.detach())  # Detach to avoid tracking gradient
    return torch.cat(all_preds, dim=0).to('cpu')

def print_confusion_matrix(model, testset, name_of_classes):
    # no shuflfe testset so using a loader appropriately, i could use testset directly but loader is more memory efficient
    loader = torch.utils.data.DataLoader(testset,
                                         batch_size = 4,
                                         shuffle = False,
                                         num_workers = 2)
    
    # Consistent device usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = get_all_predictions(model, loader)
    predicted_labels = predictions.argmax(dim=1)
    
    # Collect true labels and ensure they are on the same device as predictions
    true_labels = torch.cat([y for _, y in loader], dim=0).to(device)
    
    # Calculate confusion matrix using numpy as it is not dependent on the device
    cm = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
    
    # Visualization using matplotlib and seaborn
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='viridis', 
                xticklabels=name_of_classes, yticklabels=name_of_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
## filter dataset

class FilteredCIFAR10(Dataset):
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        self.target_classes = set(target_classes)
        
        # Filter indices and store filtered targets
        self.indices = []
        self.targets = []
        
        for i, (_, target) in enumerate(dataset):
            if target in self.target_classes:
                self.indices.append(i)
                self.targets.append(target)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

