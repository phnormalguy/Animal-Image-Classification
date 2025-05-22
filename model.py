# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

# --- Model Architecture Definition ---
class NeuralNet(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    It consists of three convolutional layers followed by max pooling,
    and then two fully connected layers.
    """
    def __init__(self):
        super().__init__()
        # First convolutional layer: 3 input channels (RGB image), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Third convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Corrected input channels to 32

        # Max pooling layer with 2x2 kernel and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer 1: Input features calculated based on output of conv/pool layers
        # For a 256x256 input, after three 2x2 max pools, the spatial dimensions become 256/2/2/2 = 32x32
        # So, the input features to fc1 will be 64 (channels from conv3) * 32 * 32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        # Fully connected layer 2: 128 input features, 3 output features (for 3 classes: cat, dog, snake)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        Args:
            x (torch.Tensor): The input tensor (image batch).
        Returns:
            torch.Tensor: The output logits from the network.
        """
        # Apply conv1, ReLU activation, and then max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2, ReLU activation, and then max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply conv3, ReLU activation, and then max pooling
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        # -1 infers the batch size, 64 * 32 * 32 is the total number of features
        x = x.view(-1, 64 * 32 * 32)
        # Apply fc1 and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply fc2 (output layer, no activation here as CrossEntropyLoss expects logits)
        x = self.fc2(x)
        return x

# --- Function to Load the Model ---
def load_model(model_path: str):
    """
    Loads the trained PyTorch model from a .pth file.

    Args:
        model_path (str): The file path to the saved model's state_dict.

    Returns:
        NeuralNet: An instance of the NeuralNet model with loaded weights.
    """
    # Instantiate the model architecture
    model = NeuralNet()

    # Load the state_dict from the specified path
    # map_location='cpu' ensures the model loads correctly even if it was trained on a GPU
    # and you are deploying on a CPU-only environment.
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set the model to evaluation mode. This is crucial for inference as it
    # disables dropout and batch normalization updates, ensuring consistent predictions.
    model.eval()
    return model

# --- Preprocessing Function ---
def preprocess_input(image_bytes: bytes):
    """
    Preprocesses raw image bytes into a PyTorch tensor suitable for model inference.

    Args:
        image_bytes (bytes): The raw bytes of an image file.

    Returns:
        torch.Tensor: A preprocessed image tensor, ready for the model.
                      It will have a batch dimension (e.g., [1, C, H, W]).
    """
    # Define the transformations for inference.
    # These should match the transformations used during training, especially Resize and Normalize.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
        transforms.ToTensor(),          # Convert the PIL Image to a PyTorch Tensor (scales to [0, 1])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize pixel values
    ])

    # Open the image using PIL from the bytes stream
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure image is in RGB format

    # Apply the transformations and add a batch dimension (unsqueeze(0))
    # The model expects input in the format [batch_size, channels, height, width]
    return transform(image).unsqueeze(0)

# --- Postprocessing Function ---
def postprocess_output(model_output: torch.Tensor):
    """
    Postprocesses the raw output (logits) from the model into a human-readable prediction.

    Args:
        model_output (torch.Tensor): The raw output tensor (logits) from the neural network.

    Returns:
        dict: A dictionary containing the predicted class name.
    """
    # Define the class names. These must match the order of your training labels.
    class_names = ['cat', 'dog', 'snake'] # Based on your notebook's class_names definition

    # Get the predicted class index by finding the maximum logit along dimension 1 (classes)
    _, predicted_class_idx = torch.max(model_output.data, 1)

    # Convert the tensor index to a Python integer and use it to get the class name
    predicted_class_name = class_names[predicted_class_idx.item()]

    return {"predicted_class": predicted_class_name}

