# inference.py
import json
import pickle
import os
import numpy as np
from tensorflow import keras
import pandas as pd

def model_fn(model_dir):
    """
    Load model artifacts when endpoint starts
    Called once at container startup
    """
    print("=" * 50)
    print("Loading model artifacts...")
    print("=" * 50)
    # Import here to avoid issues

    # Load Keras model
    model_path = os.path.join(model_dir, "model.h5")
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Model loaded successfully")

    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.pickle")
    print(f"Loading tokenizer from: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")

    # Load categories
    cat_path = os.path.join(model_dir, "catList.csv")
    print(f"Loading categories from: {cat_path}")
    categories = pd.read_csv(cat_path, header=None)
    print(f"✓ Categories loaded: {len(categories)} categories")

    print("=" * 50)
    print("Model initialization complete!")
    print("=" * 50)

    return {"model": model, "tokenizer": tokenizer, "categories": categories}


def input_fn(request_body, content_type="application/json"):
    """
    Parse incoming request
    """
    print(f"Received content type: {content_type}")

    if content_type == "application/json":
        data = json.loads(request_body)
        print(f"Parsed JSON: {data}")

        # Support multiple input formats
        if isinstance(data, dict):
            if "instances" in data:
                return data["instances"]
            elif "inputs" in data:




                return data["inputs"]
            elif "text" in data:
                return [data["text"]]
        elif isinstance(data, list):
            return data
        elif isinstance(data, str):
            return [data]

        return [str(data)]
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """
    Make predictions on input data
    """
    # Import here
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print(f"Making predictions for {len(input_data)} input(s)")

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    categories = model_dict["categories"]

    # Tokenize text
    print("Tokenizing input...")
    sequences = tokenizer.texts_to_sequences(input_data)

    # Pad sequences
    max_length = 25  # Change this if your model uses different length
    padded = pad_sequences(sequences, maxlen=max_length, padding="post")
    print(f"Padded to shape: {padded.shape}")

    # Predict
    print("Running model inference...")
    predictions = model.predict(padded, verbose=0)
    print(f"Predictions shape: {predictions.shape}")

    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Build results
    results = []
    for i, pred_class in enumerate(predicted_classes):
        category_name = categories.iloc[pred_class, 0]

        result = {
            "input": input_data[i],
            "predicted_class_index": int(pred_class),
            "category": str(category_name),
            "confidence": float(confidences[i]),
            "all_probabilities": predictions[i].tolist(),
        }
        results.append(result)
        print(f"Result {i+1}: {category_name} (confidence: {confidences[i]:.3f})")

    return results


def output_fn(prediction, accept="application/json"):
    """
    Format output for response
    """
    print(f"Formatting output as: {accept}")

    if accept == "application/json":
        return json.dumps(prediction), accept

    raise ValueError(f"Unsupported accept type: {accept}")