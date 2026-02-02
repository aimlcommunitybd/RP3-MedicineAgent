import json
from pathlib import Path

class PersistentChatHistory(list):
    # Saves chat history to a JSON file
    def __init__(self, filepath="chat_history.json"):
        self.filepath = Path(filepath)
        super().__init__(self._load())
    
    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return [] 
    
    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(list(self), f, indent=2)
    
    def append(self, item):
        super().append(item)
        self._save()  # Auto-save on append
    
    def extend(self, items):
        super().extend(items)
        self._save()  # Auto-save on extend
    
    def clear(self):
        super().clear()
        self._save()
