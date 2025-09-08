import os
import shutil
import random

# ------------------------------
# Configuración de carpetas
# ------------------------------
data_dir = r"C:\Users\Admin\Ciencias\Residuos solidos\data"  # carpeta original con todas las clases
train_dir = os.path.join(data_dir, "train")  # nueva carpeta para entrenamiento
val_dir = os.path.join(data_dir, "val")      # nueva carpeta para validación
split_ratio = 0.2  # porcentaje para validación

# ------------------------------
# Crear carpetas train/ y val/ por clase
# ------------------------------
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Listar todas las imágenes de la clase
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        # Calcular cantidad para validación
        val_count = int(len(images) * split_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Mover imágenes
        for img in val_images:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

        # Opcional: borrar carpeta original si está vacía
        if not os.listdir(class_path):
            os.rmdir(class_path)

print("Separación completa: 80% train / 20% val")
