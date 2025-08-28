import os
from PIL import Image
import matplotlib.pyplot as plt
from models.model import predict_digit

def create_zone_groups():
    """Agrupa las zonas por tipo de dato"""
    return {
        'mesa': ['MESA'],
        'ap': ['AP1', 'AP2', 'AP3'],
        'lyp_and': ['LYP_AND1', 'LYP_AND2', 'LYP_AND3'],
        'apt_sumate': ['APT_SÚMATE1', 'APT_SÚMATE2', 'APT_SÚMATE3'],
        'libre': ['LIBRE1', 'LIBRE2', 'LIBRE3'],
        'fp': ['FP1', 'FP2', 'FP3'],
        'mas_isp': ['MAS-IPSP1', 'MAS-IPSP2', 'MAS-IPSP3'],
        'morena': ['MORENA1', 'MORENA2', 'MORENA3'],
        'unidad': ['UNIDAD1', 'UNIDAD2', 'UNIDAD3'],
        'pdc': ['PDC1', 'PDC2', 'PDC3'],
        'validos': ['VOTOS_VALIDOS1', 'VOTOS_VALIDOS2', 'VOTOS_VALIDOS3'],
        'blancos': ['VOTOS_BLANCOS1', 'VOTOS_BLANCOS2', 'VOTOS_BLANCOS3'],
        'nulos': ['VOTOS_NULOS1', 'VOTOS_NULOS2', 'VOTOS_NULOS3']
    }

def combine_digits(digits_list, model, image_processor):
    """Combina dígitos individuales en un número completo"""
    number_str = ""
    confidence_sum = 0
    valid_digits = 0
    
    for digit_img in digits_list:
        try:
            processed = image_processor.transform(digit_img)
            digit, confidence = predict_digit(model, processed)
            number_str += str(digit)
            confidence_sum += confidence
            valid_digits += 1
        except Exception as e:
            print(f"Error procesando dígito: {e}")
            number_str += "0"  # Valor por defecto en caso de error
    
    if valid_digits == 0:
        return 0, 0.0
    
    return int(number_str) if number_str else 0, confidence_sum / valid_digits

def process_acta(image_path, model, image_processor, db):
    """Procesa un acta completa"""
    from PIL import Image
    
    zone_groups = create_zone_groups()
    
    try:
        digit_images = image_processor.extract_digits_from_zones(image_path, zone_groups)
        
        results = {}
        confidences = {}
        
        for group_name, digit_imgs in digit_images.items():
            try:
                if group_name == 'mesa':
                    # Solo un dígito para mesa
                    processed = image_processor.transform(digit_imgs[0])
                    results[group_name], confidences[group_name] = predict_digit(model, processed)
                else:
                    # Combinar 3 dígitos para votos
                    results[group_name], confidences[group_name] = combine_digits(
                        digit_imgs, model, image_processor
                    )
            except Exception as e:
                print(f"Error procesando grupo {group_name}: {e}")
                results[group_name] = 0
                confidences[group_name] = 0.0
        
        # Guardar en base de datos
        db.insert_acta(results)
        
        return results, confidences
        
    except Exception as e:
        print(f"Error procesando acta {image_path}: {e}")
        return {}, {}