import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config.config import ZONES, MODEL_CONFIG

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def crop_zone(self, image, zone_coords):
        """Recorta una zona específica de la imagen"""
        xmin, ymin, xmax, ymax = map(int, zone_coords)
        return image.crop((xmin, ymin, xmax, ymax))
    
    def preprocess_digit(self, image):
        """Preprocesa la imagen del dígito para el modelo"""
        # Convertir a OpenCV para procesamiento
        cv_image = np.array(image.convert('L'))
        
        # Aplicar threshold
        _, thresh = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Encontrar contornos y obtener ROI
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande (presumiblemente el dígito)
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Recortar el dígito
            digit_roi = thresh[y:y+h, x:x+w]
            
            # Redimensionar manteniendo relación de aspecto
            if w > 0 and h > 0:
                size = max(w, h)
                padded = np.zeros((size, size), dtype=np.uint8)
                
                x_offset = (size - w) // 2
                y_offset = (size - h) // 2
                padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
                
                # Redimensionar a tamaño del modelo
                resized = cv2.resize(padded, MODEL_CONFIG['input_size'])
                return Image.fromarray(resized)
        
        return image
    
    def extract_digits_from_zones(self, image_path, zone_groups):
        """Extrae dígitos de grupos de zonas (3 zonas por número)"""
        image = Image.open(image_path)
        results = {}
        
        for group_name, zones in zone_groups.items():
            digits = []
            for zone_key in zones:
                zone_img = self.crop_zone(image, ZONES[zone_key])
                processed_img = self.preprocess_digit(zone_img)
                digits.append(processed_img)
            
            results[group_name] = digits
        
        return results