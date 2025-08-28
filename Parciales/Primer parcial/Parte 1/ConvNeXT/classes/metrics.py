import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
import pandas as pd

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate_model(self, test_loader):
        """Evaluación completa del modelo"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return self._calculate_metrics(all_labels, all_preds)
    
    def _calculate_metrics(self, true_labels, predictions):
        """Calcular todas las métricas"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names=None):
        """Visualizar matriz de confusión"""
        if class_names is None:
            class_names = [str(i) for i in range(10)]  # Dígitos 0-9 para EMNIST
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Número de muestras'})
        plt.title('Matriz de Confusión - Reconocimiento de Dígitos (EMNIST)')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig('confusion_matrix_emnist.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """Visualizar historial de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pérdida
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Pérdida durante el Entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precisión
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Precisión durante el Entrenamiento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning Rate
        ax3.plot(history['learning_rates'], label='Learning Rate', linewidth=2, color='purple')
        ax3.set_title('Evolución del Learning Rate')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Comparación Precisión vs Pérdida
        ax4.plot(history['val_acc'], history['val_loss'], 'o-', linewidth=2)
        ax4.set_title('Precisión vs Pérdida (Validation)')
        ax4.set_xlabel('Precisión (%)')
        ax4.set_ylabel('Pérdida')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_emnist.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, metrics, history):
        """Generar reporte completo"""
        report = {
            'dataset': 'EMNIST Digits',
            'model': 'ConvNeXT-Tiny',
            'final_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score'])
            },
            'training_history': {
                'final_train_accuracy': float(history['train_acc'][-1]),
                'final_val_accuracy': float(history['val_acc'][-1]),
                'final_train_loss': float(history['train_loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'best_val_accuracy': float(max(history['val_acc']))
            },
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'detailed_report': metrics['classification_report']
        }
        
        # Guardar métricas en CSV
        df_metrics = pd.DataFrame([metrics['classification_report']['weighted avg']])
        df_metrics['dataset'] = 'EMNIST'
        df_metrics.to_csv('model_metrics_emnist.csv', index=False)
        
        return report