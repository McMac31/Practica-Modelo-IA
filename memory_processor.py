# memory_processor.py
import re
import logging
from typing import Dict
from database import memory_db

logger = logging.getLogger(__name__)

class MemoryProcessor:
    def __init__(self):
        self.db = memory_db
        
    def extract_user_info(self, message: str, session_id: str) -> Dict[str, str]:
        """Extraer información personal del mensaje"""
        updates = {}
        message_lower = message.lower()
        
        # Extraer nombre
        name_patterns = [
            r"(me llamo|mi nombre es|soy) ([A-Za-z]+)",
            r"(nombre es|me dicen) ([A-Za-z]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(2)
                if name and len(name) > 2:
                    updates["user_name"] = name.capitalize()
                    break
        
        return updates
    
    def process_conversation_for_memory(self, user_message: str, assistant_response: str, session_id: str):
        """Procesar conversación y guardar en SQLite"""
        try:
            # Extraer información del usuario
            user_updates = self.extract_user_info(user_message, session_id)
            
            # Guardar en base de datos
            if user_updates:
                self.db.update_user_memory(session_id, user_updates)
                print(f"🧠 Memoria actualizada: {user_updates}")
                
        except Exception as e:
            print(f"Error procesando memoria: {e}")
    
    def get_conversation_context(self, session_id: str) -> str:
        """Obtener contexto de memoria para el prompt"""
        memory = self.db.get_user_memory(session_id) or {}
        
        context_parts = []
        if memory.get("user_name"):
            context_parts.append(f"El usuario se llama {memory['user_name']}")
        
        return ". ".join(context_parts)

# Instancia global
memory_processor = MemoryProcessor()