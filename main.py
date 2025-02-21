import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image

# Load pre-trained Xception model for deepfake detection
model = timm.create_model('xception', pretrained=True, num_classes=1)  # Ensure output has 1 neuron
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception model input size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to match training data
])

def preprocess_frame(frame):
    """Preprocess a video frame for model input"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def detect_deepfake(video_path):
    """Detect deepfake in a video by processing frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    deepfake_probabilities = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame for efficiency
            input_tensor = preprocess_frame(frame)

            with torch.no_grad():
                output = model(input_tensor)  # Output is a tensor of shape [1, 1]
                fake_score = torch.sigmoid(output).squeeze().item()  # Convert to scalar

                deepfake_probabilities.append(fake_score)

            print(f"Frame {frame_count}: Deepfake Probability = {fake_score:.4f}")

    cap.release()
    cv2.destroyAllWindows()

    # Compute average deepfake probability
    avg_fake_score = np.mean(deepfake_probabilities) if deepfake_probabilities else 0
    print(f"\nOverall Deepfake Probability: {avg_fake_score:.4f}")
    if avg_fake_score > 0.5:  # Threshold for deepfake detection
        print("Deepfake detected!")
    else:
        print("No deepfake detected.")

# Run detection on a video file
detect_deepfake("videoplayback3.mp4")
