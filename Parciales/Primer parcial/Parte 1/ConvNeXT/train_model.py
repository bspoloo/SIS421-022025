import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from classes.metrics import ModelEvaluator
from models.model import DigitRecognizer
import os
import json
import torchvision.transforms.functional as F

# Funciones auxiliares para evitar lambdas
def rotate_90_degrees(img):
    """Rotar imagen 90 grados en sentido antihorario"""
    return F.rotate(img, -90)

def horizontal_flip(img):
    """Voltear imagen horizontalmente"""
    return F.hflip(img)

def train_digit_recognizer():
    # Transformaciones con data augmentation para regularización
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  # Regularización: aumento de datos
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Pequeñas translaciones
        transforms.Resize((64, 64)),
        transforms.Lambda(rotate_90_degrees),  # EMNIST está rotado
        transforms.Lambda(horizontal_flip),  # EMNIST está volteado
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Lambda(rotate_90_degrees),  # EMNIST está rotado
        transforms.Lambda(horizontal_flip),  # EMNIST está volteado
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Cargando dataset EMNIST...")
    
    # Cargar datasets EMNIST - usando 'digits' para solo dígitos (0-9)
    try:
        train_dataset = datasets.EMNIST(
            root='./data', 
            split='digits',  # Solo dígitos: 0-9
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        test_dataset = datasets.EMNIST(
            root='./data', 
            split='digits',  # Solo dígitos: 0-9
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        print(f"✓ EMNIST cargado: {len(train_dataset)} ejemplos de entrenamiento, {len(test_dataset)} de prueba")
        print(f"✓ Clases: {train_dataset.classes}")
        
    except Exception as e:
        print(f"Error cargando EMNIST: {e}")
        print("Intentando con MNIST como fallback...")
        
        # Fallback a MNIST sin las transformaciones de rotación
        train_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        print("✓ MNIST cargado como fallback")
    
    # DataLoaders - SIN multiprocessing para Windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=0,  # Cambiado a 0 para Windows
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0,  # Cambiado a 0 para Windows
        pin_memory=True
    )
    
    # Modelo con optimizaciones
    model = DigitRecognizer(num_classes=10)  # 10 clases para dígitos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    model.to(device)
    
    # Optimizador con weight decay para regularización
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Regularización L2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Historial de entrenamiento
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    # Entrenamiento
    best_accuracy = 0.0
    early_stopping_patience = 5
    patience_counter = 0
    
    print("\nIniciando entrenamiento...")
    for epoch in range(20):
        # Fase de entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calcular métricas
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Actualizar historial
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduling de learning rate
        scheduler.step(val_accuracy)
        
        print(f'Epoch {epoch+1}/20')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # Guardar mejor modelo
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"✓ Mejor modelo guardado con accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    # Evaluación final
    print("\nEvaluación final del modelo...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate_model(test_loader)
    
    # Generar reportes y visualizaciones
    report = evaluator.generate_report(metrics, history)
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], class_names=[str(i) for i in range(10)])
    evaluator.plot_training_history(history)
    
    # Guardar reporte completo
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n🎯 Mejor precisión en validación: {best_accuracy:.2f}%")
    print("📊 Reporte de entrenamiento guardado en 'training_report.json'")
    print("📈 Gráficos de entrenamiento guardados")
    
    return model, metrics, history

if __name__ == "__main__":
    train_digit_recognizer()