# database.py
import sqlite3
import os
from typing import Dict, Any, Optional

class SophieMemoryDatabase:
    def __init__(self, db_path: str = "sophie_memory.db"):
        self.db_path = db_path
        print(f"🗃️ Inicializando base de datos: {db_path}")
        self.init_database()

    def init_database(self):
        """Inicializar la base de datos SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de mensajes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla de memoria de usuario
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    user_name TEXT,
                    user_age INTEGER,
                    user_gender TEXT,
                    user_location TEXT,
                    user_interests TEXT,
                    important_facts TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✅ Base de datos creada: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error creando base de datos: {e}")

    def save_message(self, session_id: str, role: str, content: str):
        """Guardar mensaje en la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()
            conn.close()
            print(f"💾 Mensaje guardado: {role} - {content[:20]}...")
        except Exception as e:
            print(f"Error guardando mensaje: {e}")

    def update_user_memory(self, session_id: str, memory_updates: Dict[str, Any]):
        """Actualizar memoria del usuario"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar si existe
            cursor.execute("SELECT id FROM user_memory WHERE session_id = ?", (session_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Actualizar
                set_clause = []
                params = []
                for key, value in memory_updates.items():
                    if value is not None:
                        set_clause.append(f"{key} = ?")
                        params.append(value)
                
                if set_clause:
                    params.append(session_id)
                    query = f"UPDATE user_memory SET {', '.join(set_clause)} WHERE session_id = ?"
                    cursor.execute(query, params)
            else:
                # Insertar nuevo
                fields = ['session_id']
                placeholders = ['?']
                values = [session_id]
                
                for key, value in memory_updates.items():
                    if value is not None:
                        fields.append(key)
                        placeholders.append('?')
                        values.append(value)
                
                query = f"INSERT INTO user_memory ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
                cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            print(f"🧠 Memoria actualizada: {memory_updates}")
            
        except Exception as e:
            print(f"Error actualizando memoria: {e}")

    def get_user_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener memoria del usuario"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_name, user_age, user_gender, user_location, user_interests, important_facts
                FROM user_memory WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "user_name": row[0],
                    "user_age": row[1],
                    "user_gender": row[2],
                    "user_location": row[3],
                    "user_interests": row[4],
                    "important_facts": row[5]
                }
            return None
            
        except Exception as e:
            print(f"Error obteniendo memoria: {e}")
            return None

    def _get_connection(self):
        """Obtener conexión a la BD (para tests)"""
        return sqlite3.connect(self.db_path)

# Crear instancia global
memory_db = SophieMemoryDatabase()