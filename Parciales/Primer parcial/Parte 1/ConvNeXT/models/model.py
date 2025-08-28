import torch
import torch.nn as nn
from torchvision import models
from config.config import MODEL_CONFIG
import os

class DigitRecognizer(nn.Module):
    def __init__(self, num_classes=10):  # 10 clases para dígitos EMNIST
        super(DigitRecognizer, self).__init__()
        
        # Usar ConvNeXT como backbone
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        
        # Reemplazar la capa final para clasificación de dígitos
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
        
        # Congelar capas iniciales para fine-tuning más eficiente
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Congelar capas iniciales para transfer learning más efectivo"""
        # Congelar las primeras capas convolucionales
        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Descongelar todas las capas"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path=None, num_classes=10):
    """Carga el modelo preentrenado"""
    model = DigitRecognizer(num_classes=num_classes)
    
    if model_path and os.path.exists(model_path):
        try:
            # Cargar pesos guardados
            model.load_state_dict(torch.load(model_path, map_location=MODEL_CONFIG['device']))
            print(f"✓ Modelo cargado desde {model_path}")
        except Exception as e:
            print(f"⚠ Error cargando modelo: {e}")
            print("✓ Inicializando modelo desde cero")
    else:
        print("✓ Inicializando modelo desde cero")
    
    model.to(MODEL_CONFIG['device'])
    model.eval()
    return model

def predict_digit(model, image_tensor):
    """Predice el dígito de una imagen"""
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(MODEL_CONFIG['device']))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted].item()
    
    return predicted, confidence