from database.database import ActaDatabase

def setup_database():
    """Script para configurar la base de datos MySQL"""
    print("Configurando base de datos MySQL...")
    
    # Esto creará automáticamente la base de datos y tablas
    db = ActaDatabase()
    
    # Verificar que todo esté funcionando
    actas = db.get_all_actas()
    print(f"Total de actas en la base de datos: {len(actas)}")
    
    db.close()
    print("Configuración completada!")

if __name__ == "__main__":
    setup_database()