import argparse
import json
import matplotlib.pyplot as plt
from classes.image_processor import ImageProcessor
from classes.metrics import ModelEvaluator
from database.database import ActaDatabase
from models.model import load_model, predict_digit
from classes.utils import process_acta
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def demonstrate_model_effectiveness():
    """Demostración completa de la efectividad del modelo"""
    print("=" * 60)
    print("DEMOSTRACIÓN DE EFECTIVIDAD DEL MODELO")
    print("=" * 60)
    
    # 1. Cargar el mejor modelo
    print("\n1. CARGANDO MODELO ENTRENADO")
    model = load_model('models/best_model.pth')
    
    # 2. Evaluar en dataset de prueba
    print("\n2. EVALUACIÓN EN DATASET DE PRUEBA (MNIST)")
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    evaluator = ModelEvaluator(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    metrics = evaluator.evaluate_model(test_loader)
    
    print(f"✓ Precisión: {metrics['accuracy']:.4f}")
    print(f"✓ Precisión ponderada: {metrics['precision']:.4f}")
    print(f"✓ Sensibilidad: {metrics['recall']:.4f}")
    print(f"✓ Puntuación F1: {metrics['f1_score']:.4f}")
    
    # 3. Mostrar matriz de confusión
    print("\n3. MATRIZ DE CONFUSIÓN")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'])
    
    # 4. Procesar actas de ejemplo
    print("\n4. PROCESAMIENTO DE ACTAS ELECTORALES")
    db = ActaDatabase()
    image_processor = ImageProcessor()
    
    # Procesar algunas actas de ejemplo
    actas_folder = 'actas'
    import os
    if os.path.exists(actas_folder):
        image_files = [f for f in os.listdir(actas_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files[:3]:  # Procesar solo 3 para demo
            image_path = os.path.join(actas_folder, image_file)
            print(f"\nProcesando: {image_file}")
            
            try:
                results, confidences = process_acta(image_path, model, image_processor, db)
                print(f"✓ Mesa {results.get('mesa', 'N/A')} procesada")
                print(f"  Confianza promedio: {sum(confidences.values())/len(confidences):.3f}")
                
            except Exception as e:
                print(f"✗ Error procesando {image_file}: {e}")
    
    # 5. Mostrar resultados en base de datos
    print("\n5. RESULTADOS EN BASE DE DATOS")
    actas = db.get_all_actas()
    if actas:
        print(f"✓ Total de actas procesadas: {len(actas)}")
        for acta in actas[:3]:  # Mostrar solo 3
            print(f"  Mesa {acta['numero_mesa']}: "
                  f"Votos válidos: {acta['votos_validos']}")
    else:
        print("✗ No hay actas procesadas")
    
    db.close()
    
    # 6. Explicación del modelo ConvNeXT
    print("\n6. EXPLICACIÓN DEL MODELO ConvNeXT")
    print("✓ Arquitectura moderna basada en Vision Transformers")
    print("✓ Bloques convolucionales invertidos para mejor captura de features")
    print("✓ Pre-entrenado en ImageNet para transfer learning")
    print("✓ Fine-tuning para reconocimiento de dígitos")
    
    # 7. Técnicas de Deep Learning aplicadas
    print("\n7. TÉCNICAS DE DEEP LEARNING APLICADAS")
    print("✓ Transfer Learning con ConvNeXT pre-entrenado")
    print("✓ Data Augmentation para regularización")
    print("✓ Weight Decay (L2 regularization)")
    print("✓ Learning Rate Scheduling adaptativo")
    print("✓ Gradient Clipping para estabilidad")
    print("✓ Early Stopping para prevenir overfitting")
    
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_model_effectiveness()