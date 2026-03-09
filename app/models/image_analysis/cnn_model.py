"""
SmartCertify ML — CNN Tampering Detection
ResNet-18 based certificate image tampering detector with Grad-CAM.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR
from app.utils.model_io import save_pytorch_model, load_pytorch_model
from app.models.image_analysis.preprocess import (
    preprocess_image, load_image_from_base64, image_to_base64,
)

logger = logging.getLogger(__name__)

_model = None
_device = None


class TamperingDetectorCNN(nn.Module):
    """ResNet-18 based tampering detector."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=False)

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        # Store activations for Grad-CAM
        self._activations = None
        self._gradients = None

    def _activation_hook(self, module, input, output):
        self._activations = output.detach()

    def _gradient_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def register_hooks(self):
        """Register hooks for Grad-CAM."""
        self.backbone.layer4.register_forward_hook(self._activation_hook)
        self.backbone.layer4.register_full_backward_hook(self._gradient_hook)

    def forward(self, x):
        return self.backbone(x)


def _get_model() -> TamperingDetectorCNN:
    """Load or create the CNN model."""
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = TamperingDetectorCNN()

        # Try to load saved weights
        model_path = MODEL_DIR / "cnn_tampering.pt"
        if model_path.exists():
            _model = load_pytorch_model(_model, "cnn_tampering.pt", device=_device)
        else:
            logger.warning("No trained CNN model found, using pretrained ResNet-18 backbone")
            _model.to(_device)

        _model.eval()
        _model.register_hooks()

    return _model


def generate_gradcam(model: TamperingDetectorCNN, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """Generate Grad-CAM heatmap."""
    model.eval()
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    model.zero_grad()

    output[0, target_class].backward()

    gradients = model._gradients
    activations = model._activations

    if gradients is None or activations is None:
        return np.zeros((224, 224), dtype=np.float32)

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize and resize
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to input size
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((224, 224), PILImage.Resampling.LANCZOS)
    cam = np.array(cam_img, dtype=np.float32) / 255.0

    return cam


def analyze_image(
    image_input: Any,
    certificate_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze a certificate image for tampering.

    Args:
        image_input: Base64 string or PIL Image
        certificate_id: ID for tracking

    Returns:
        Dictionary with tampering analysis results
    """
    model = _get_model()
    device = _device or torch.device("cpu")

    # Load image
    if isinstance(image_input, str):
        image = load_image_from_base64(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        return {"error": "Invalid image input"}

    if image is None:
        return {"error": "Failed to load image"}

    # Preprocess
    img_array = preprocess_image(image)
    input_tensor = torch.FloatTensor(img_array).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    is_tampered = bool(probabilities[1] > probabilities[0])
    tamper_probability = float(probabilities[1])
    confidence = float(max(probabilities))

    # Generate Grad-CAM
    gradcam_base64 = None
    try:
        cam = generate_gradcam(model, input_tensor.clone().requires_grad_(True), target_class=1)

        # Create heatmap overlay
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        heatmap = cm.jet(cam)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap)

        # Overlay on original image
        original_resized = image.resize((224, 224))
        overlay = Image.blend(original_resized, heatmap_img, alpha=0.4)
        gradcam_base64 = image_to_base64(overlay)

    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    return {
        "certificate_id": certificate_id,
        "is_tampered": is_tampered,
        "tamper_probability": round(tamper_probability, 4),
        "confidence": round(confidence, 4),
        "gradcam_heatmap_base64": gradcam_base64,
    }


def train_cnn(epochs: int = 20):
    """Train the CNN on synthetic data."""
    from app.models.image_analysis.preprocess import generate_synthetic_tampered_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TamperingDetectorCNN().to(device)

    logger.info("Generating synthetic training data...")
    samples = generate_synthetic_tampered_images(n_samples=200)

    # Build training data
    X_list, y_list = [], []
    for s in samples:
        X_list.append(s["authentic"])
        y_list.append(0)
        X_list.append(s["tampered"])
        y_list.append(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total if total > 0 else 0
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    save_pytorch_model(model, "cnn_tampering.pt", metadata={"val_acc": val_acc})
    logger.info(f"CNN model saved. Final val accuracy: {val_acc:.4f}")
    return model
