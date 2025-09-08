import mysql.connector
from mysql.connector import Error
from config.config import MYSQL_CONFIG
import time

class ActaDatabase:
    def __init__(self):
        self.connection = None
        self.connect()
        self.create_database()
        self.create_tables()
    
    def connect(self, retries=3, delay=2):
        """Establecer conexión con MySQL"""
        for attempt in range(retries):
            try:
                self.connection = mysql.connector.connect(
                    host=MYSQL_CONFIG['host'],
                    user=MYSQL_CONFIG['user'],
                    password=MYSQL_CONFIG['password'],
                    port=MYSQL_CONFIG['port']
                )
                print("Conexión a MySQL establecida correctamente")
                return True
            except Error as e:
                print(f"Intento {attempt + 1} de {retries}: Error conectando a MySQL - {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("No se pudo conectar a MySQL después de varios intentos")
                    return False
    
    def create_database(self):
        """Crear la base de datos si no existe"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
            cursor.execute(f"USE {MYSQL_CONFIG['database']}")
            print(f"Base de datos '{MYSQL_CONFIG['database']}' verificada/creada")
        except Error as e:
            print(f"Error creando base de datos: {e}")
    
    def create_tables(self):
        """Crear las tablas necesarias"""
        try:
            cursor = self.connection.cursor()
            
            # Seleccionar la base de datos
            cursor.execute(f"USE {MYSQL_CONFIG['database']}")
            
            # Crear tabla de actas
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS actas (
                id INT AUTO_INCREMENT PRIMARY KEY,
                numero_mesa INT NOT NULL,
                ap_votos INT,
                lyp_and_votos INT,
                apt_sumate_votos INT,
                libre_votos INT,
                fp_votos INT,
                mas_isp_votos INT,
                morena_votos INT,
                unidad_votos INT,
                pdc_votos INT,
                votos_validos INT,
                votos_blancos INT,
                votos_nulos INT,
                fecha_procesamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''')
            
            self.connection.commit()
            print("Tabla 'actas' verificada/creada correctamente")
            
        except Error as e:
            print(f"Error creando tablas: {e}")
    
    def get_connection(self):
        """Obtener conexión a la base de datos específica"""
        try:
            if not self.connection.is_connected():
                self.connect()
            
            # Asegurarse de usar la base de datos correcta
            cursor = self.connection.cursor()
            cursor.execute(f"USE {MYSQL_CONFIG['database']}")
            
            return self.connection
        except Error as e:
            print(f"Error obteniendo conexión: {e}")
            return None
    
    def insert_acta(self, data):
        """Insertar o actualizar un acta en la base de datos"""
        try:
            connection = self.get_connection()
            if not connection:
                return None
            
            cursor = connection.cursor()
            
            # Consulta para insertar o actualizar
            query = '''
            INSERT INTO actas 
            (numero_mesa, ap_votos, lyp_and_votos, apt_sumate_votos, libre_votos, 
             fp_votos, mas_isp_votos, morena_votos, unidad_votos, pdc_votos,
             votos_validos, votos_blancos, votos_nulos)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                ap_votos = VALUES(ap_votos),
                lyp_and_votos = VALUES(lyp_and_votos),
                apt_sumate_votos = VALUES(apt_sumate_votos),
                libre_votos = VALUES(libre_votos),
                fp_votos = VALUES(fp_votos),
                mas_isp_votos = VALUES(mas_isp_votos),
                morena_votos = VALUES(morena_votos),
                unidad_votos = VALUES(unidad_votos),
                pdc_votos = VALUES(pdc_votos),
                votos_validos = VALUES(votos_validos),
                votos_blancos = VALUES(votos_blancos),
                votos_nulos = VALUES(votos_nulos),
                updated_at = CURRENT_TIMESTAMP
            '''
            
            values = (
                data.get('mesa', 0),
                data.get('ap', 0),
                data.get('lyp_and', 0),
                data.get('apt_sumate', 0),
                data.get('libre', 0),
                data.get('fp', 0),
                data.get('mas_isp', 0),
                data.get('morena', 0),
                data.get('unidad', 0),
                data.get('pdc', 0),
                data.get('validos', 0),
                data.get('blancos', 0),
                data.get('nulos', 0)
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            print(f"Acta de mesa {data.get('mesa', 'N/A')} guardada en MySQL")
            return cursor.lastrowid
            
        except Error as e:
            print(f"Error insertando acta en MySQL: {e}")
            if connection:
                connection.rollback()
            return None
    
    def get_all_actas(self):
        """Obtener todas las actas de la base de datos"""
        try:
            connection = self.get_connection()
            if not connection:
                return []
            
            cursor = connection.cursor(dictionary=True)
            cursor.execute('''
                SELECT * FROM actas 
                ORDER BY numero_mesa
            ''')
            
            result = cursor.fetchall()
            return result
            
        except Error as e:
            print(f"Error obteniendo actas: {e}")
            return []
    
    def close(self):
        """Cerrar la conexión a la base de datos"""
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
                print("Conexión a MySQL cerrada")
        except Error as e:
            print(f"Error cerrando conexión: {e}")