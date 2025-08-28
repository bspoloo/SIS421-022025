import os
import argparse
from classes.image_processor import ImageProcessor
from database.database import ActaDatabase
from models.model import load_model
from classes.utils import process_acta
import glob

def main():
    parser = argparse.ArgumentParser(description='Procesador de actas electorales')
    parser.add_argument('--image', '-i', help='Ruta de la imagen del acta a procesar')
    parser.add_argument('--folder', '-f', help='Carpeta con múltiples actas a procesar')
    parser.add_argument('--model', '-m', default='models/digit_recognizer.pth', help='Ruta del modelo preentrenado')
    
    args = parser.parse_args()
    
    # Verificar si el modelo existe
    if not os.path.exists(args.model):
        print(f"Advertencia: Modelo {args.model} no encontrado. Se usará modelo desde cero.")
    
    # Inicializar componentes
    db = ActaDatabase()
    image_processor = ImageProcessor()  # Esta línea estaba corrupta
    model = load_model(args.model)
    
    if args.image:
        # Procesar una sola imagen
        if os.path.exists(args.image):
            print(f"Procesando acta: {args.image}")
            results, confidences = process_acta(args.image, model, image_processor, db)
            
            print("\nResultados:")
            for key, value in results.items():
                print(f"{key}: {value} (confianza: {confidences.get(key, 0):.3f})")
        
        else:
            print(f"Error: Archivo {args.image} no encontrado")
    
    elif args.folder:
        # Procesar todas las imágenes en una carpeta
        if os.path.exists(args.folder):
            # Buscar imágenes en formatos comunes
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            
            for extension in image_extensions:
                image_files.extend(glob.glob(os.path.join(args.folder, extension)))
            
            if not image_files:
                print(f"No se encontraron imágenes en la carpeta {args.folder}")
                return
            
            print(f"Encontradas {len(image_files)} imágenes para procesar")
            
            for image_path in image_files:
                print(f"\nProcesando: {os.path.basename(image_path)}")
                
                try:
                    results, confidences = process_acta(image_path, model, image_processor, db)
                    if results:
                        print(f"Mesa {results.get('mesa', 'N/A')} procesada correctamente")
                    else:
                        print("Error: No se pudieron obtener resultados")
                
                except Exception as e:
                    print(f"Error procesando {os.path.basename(image_path)}: {e}")
        
        else:
            print(f"Error: Carpeta {args.folder} no encontrada")
    
    else:
        # Mostrar datos existentes
        actas = db.get_all_actas()
        if actas:
            print("Actas procesadas en la base de datos:")
            print("-" * 80)
            for acta in actas:
                print(f"Mesa {acta[1]}: "
                      f"AP={acta[2]}, LYP={acta[3]}, SÚMATE={acta[4]}, "
                      f"LIBRE={acta[5]}, FP={acta[6]}, MAS={acta[7]}, "
                      f"MORENA={acta[8]}, UNIDAD={acta[9]}, PDC={acta[10]}, "
                      f"VÁLIDOS={acta[11]}, BLANCOS={acta[12]}, NULOS={acta[13]}")
        else:
            print("No hay actas procesadas. Use --image o --folder para procesar actas.")
            print("\nEjemplos de uso:")
            print("  python app.py --image actas/mi_acta.jpg")
            print("  python app.py --folder actas/")
            print("  python app.py --image actas/mi_acta.jpg --model models/mi_modelo.pth")
    
    db.close()

if __name__ == "__main__":
    main()