# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import os
from contextlib import asynccontextmanager # Import asynccontextmanager

# Import your model loading and preprocessing functions from model.py
from model import load_model, preprocess_input, postprocess_output, NeuralNet

# Define the path to your saved model
MODEL_PATH = "animal_classifier_aug_30e_m095.pth" # Adjust if your path is different

# Global variable to store the loaded model
# This ensures the model is loaded only once when the app starts
model = None

@asynccontextmanager
async def lifespan_events(app: FastAPI):
    """
    Context manager for managing application startup and shutdown events.
    The model loading logic is moved here to ensure it runs only once
    when the FastAPI application starts.
    """
    global model
    print("Application startup: Loading model...")
    try:
        if not os.path.exists(MODEL_PATH):
            # If the model file is not found, raise an error and exit
            # In a real deployment, you might want to log this extensively
            # and ensure the file is present before starting.
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
        # Load the model using the function from model.py
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model during startup: {e}")
        # Re-raise the exception to prevent the application from starting
        # if the model cannot be loaded, as it's a critical dependency.
        raise RuntimeError(f"Application failed to start due to model loading error: {e}")

    # Yield control to the application. Code after 'yield' runs on shutdown.
    yield
    print("Application shutdown: Cleaning up resources (if any)...")
    # You can add any cleanup logic here if needed, e.g., closing database connections
    # For a simple model serving, there might not be much to clean up.


# Initialize the FastAPI application with the lifespan event handler
app = FastAPI(lifespan=lifespan_events)


# Define request and response models using Pydantic for data validation
# Adjust these based on your model's input and output
class PredictionRequest(BaseModel):
    # For image prediction, we expect a file, so this Pydantic model isn't directly used
    # for the image upload endpoint. It's kept here as an example for other types of inputs.
    # If your model takes structured data, define its fields here.
    # Example:
    # feature1: float
    # feature2: float
    pass # No fields needed for file upload example

class PredictionResponse(BaseModel):
    # The response structure matches what postprocess_output returns
    predicted_class: str


@app.get("/")
async def read_root():
    """
    Root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the ML Model API! Use /predict_image for predictions."}

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint for image classification.
    Expects an image file as input and returns the predicted class.
    """
    if model is None:
        # This should ideally not happen if lifespan_events works correctly,
        # but it's a good safeguard.
        raise HTTPException(status_code=500, detail="Model not loaded. Server might be starting up or encountered an error.")

    try:
        # Read the contents of the uploaded image file
        contents = await file.read()
        
        # Preprocess the image bytes into a tensor suitable for the model
        processed_image = preprocess_input(contents)

        # Perform inference with the model
        # torch.no_grad() is used to disable gradient calculations, saving memory and speeding up inference
        with torch.no_grad():
            output = model(processed_image)

        # Postprocess the model's raw output (logits) into a human-readable prediction
        response_data = postprocess_output(output)
        
        # Return the structured response
        return PredictionResponse(**response_data)
    except Exception as e:
        # Catch any exceptions during prediction and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {e}")

