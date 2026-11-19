"""
Script portable para usar el modelo entrenado en cualquier PC
Solo necesitas copiar:
  1. Este archivo (inferencia_portable.py)
  2. La carpeta 'qwen-socratic-feedback/final/' con los adaptadores

Requisitos: pip install torch transformers peft accelerate
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ============================================
# CONFIGURACIÓN
# ============================================
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "./qwen-socratic-feedback/final"  # Ajusta si lo pones en otro lugar

class ModeloSocratico:
    def __init__(self, adapter_path=ADAPTER_PATH):
        print("Cargando modelo...")
        
        # Detectar dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Dispositivo: {self.device}")
        
        # Cargar tokenizer desde los adaptadores (tiene configuración específica)
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, 
            trust_remote_code=True
        )
        
        # Cargar modelo base
        print("Descargando/cargando modelo base (solo la primera vez)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Cargar adaptadores LoRA
        print("Cargando adaptadores LoRA...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("¡Modelo listo!\n")
    
    def generar(self, prompt, max_tokens=256, temperature=0.7):
        """Genera una respuesta dado un prompt"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decodificar respuesta
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la respuesta del asistente
        if "assistant" in full_response.lower():
            response = full_response.split("assistant")[-1].strip()
        else:
            # Buscar después del prompt
            response = full_response[len(text):].strip()
        
        return response


# ============================================
# EJEMPLO DE USO
# ============================================
if __name__ == "__main__":
    # Verificar que existan los adaptadores
    if not os.path.exists(ADAPTER_PATH):
        print(f"ERROR: No se encontró la carpeta de adaptadores: {ADAPTER_PATH}")
        print("Asegúrate de copiar la carpeta 'qwen-socratic-feedback/final/' junto a este script")
        exit(1)
    
    # Cargar modelo
    modelo = ModeloSocratico()
    
    # Ejemplo de prompt (como los de tu dataset)
    ejemplo = """INSTRUCCIONES CRITICAS:
- Tu respuesta DEBE contener SOLO 1 o 2 bullet points.
- DIRIGETE AL ESTUDIANTE DIRECTAMENTE usando "tu".

### Funcion correcta de referencia:
```python
def duplicar(n):
    return n * 2
```

### Funcion del estudiante:
```python
def duplicar(n):
    return n + 2
```

### Doctest fallido:
- Entrada: duplicar(5)
- Salida esperada: 10
- Salida del estudiante: 7

Genera tus 1-2 bullet points:"""

    print("="*60)
    print("PRUEBA DEL MODELO")
    print("="*60)
    print("\nPrompt de prueba:")
    print("-"*40)
    print(ejemplo[:200] + "...")
    print("-"*40)
    
    respuesta = modelo.generar(ejemplo)
    
    print("\nRespuesta del modelo:")
    print("-"*40)
    print(respuesta)
    print("-"*40)
    
    # Modo interactivo
    print("\n" + "="*60)
    print("MODO INTERACTIVO")
    print("Escribe 'salir' para terminar")
    print("="*60)
    
    while True:
        try:
            print("\n")
            user_input = input("Tu prompt >>> ")
            if user_input.lower().strip() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            if not user_input.strip():
                continue
            
            respuesta = modelo.generar(user_input)
            print("\nRespuesta:")
            print(respuesta)
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break

