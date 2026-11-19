"""
Entrenamiento QLoRA de Qwen2.5-Coder-1.5B para retroalimentación socrática
Optimizado para RTX 4070 Ti (16GB VRAM)

Ejecutar:
    python entrenar_qwen_qlora.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import os

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Modelo base
OUTPUT_DIR = "./qwen-socratic-feedback"          # Directorio de salida
MAX_LENGTH = 2048                                 # Longitud máxima de secuencia

# Hiperparámetros de entrenamiento
LEARNING_RATE = 2e-4
BATCH_SIZE = 2                  # Ajustar si hay problemas de memoria
GRADIENT_ACCUMULATION = 8       # Simula batch_size efectivo de 16
NUM_EPOCHS = 3
WARMUP_STEPS = 50

# Configuración LoRA
LORA_R = 16                     # Rango de la matriz LoRA
LORA_ALPHA = 32                 # Factor de escala
LORA_DROPOUT = 0.05

# ============================================
# PREPARAR DATOS
# ============================================
def preparar_datos_si_no_existen():
    """Ejecuta el script de preparación si no existen los archivos"""
    if not os.path.exists("train.jsonl") or not os.path.exists("test.jsonl"):
        print("Preparando datos...")
        import preparar_datos
        preparar_datos.preparar_dataset()
        print("Datos preparados!\n")

# ============================================
# FUNCIONES DE TOKENIZACIÓN
# ============================================
def format_chat(example, tokenizer):
    """Formatea el ejemplo al formato de chat de Qwen"""
    messages = example["messages"]
    
    # Aplicar template de chat
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def tokenize_function(examples, tokenizer):
    """Tokeniza los ejemplos"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main():
    print("="*60)
    print("Entrenamiento QLoRA - Qwen2.5-Coder-1.5B")
    print("Retroalimentación Socrática para Programación")
    print("="*60)
    
    # Verificar CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Se requiere GPU para entrenar.")
    
    print(f"\nGPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Preparar datos
    preparar_datos_si_no_existen()
    
    # ----------------------------------------
    # Cargar tokenizer
    # ----------------------------------------
    print("\n[1/5] Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ----------------------------------------
    # Cargar modelo con cuantización 4-bit
    # ----------------------------------------
    print("[2/5] Cargando modelo con cuantización 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Preparar para entrenamiento k-bit
    model = prepare_model_for_kbit_training(model)
    
    # ----------------------------------------
    # Configurar LoRA
    # ----------------------------------------
    print("[3/5] Configurando adaptadores LoRA...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ----------------------------------------
    # Cargar y procesar dataset
    # ----------------------------------------
    print("[4/5] Cargando y procesando dataset...")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": "train.jsonl",
            "test": "test.jsonl"
        }
    )
    
    # Formatear al estilo chat
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )
    
    # Tokenizar
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    print(f"  Train: {len(dataset['train'])} ejemplos")
    print(f"  Test: {len(dataset['test'])} ejemplos")
    
    # ----------------------------------------
    # Configurar entrenamiento
    # ----------------------------------------
    print("[5/5] Iniciando entrenamiento...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,  # RTX 4070 Ti soporta bfloat16
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # Desactivar wandb/tensorboard
        dataloader_pin_memory=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    # Entrenar
    print("\n" + "="*60)
    print("ENTRENANDO...")
    print("="*60 + "\n")
    
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    
    # ----------------------------------------
    # Guardar modelo
    # ----------------------------------------
    print("\nGuardando modelo...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO!")
    print(f"Modelo guardado en: {OUTPUT_DIR}/final")
    print("="*60)

if __name__ == "__main__":
    main()

