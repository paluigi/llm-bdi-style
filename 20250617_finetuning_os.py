import os
import time

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# 📌 CONFIG

# Qwen
model_name = "Qwen2.5-7B-Instruct"
chat_template = "qwen-2.5"

# # Gemma
# model_name = "gemma-3-12b-it"
# chat_template = "gemma3"


MODEL_PATH = os.path.join("leonardo_work/try25_boigenai/Luigi", model_name) 
EXCEL_PATH =  os.path.join("leonardo/home/userexternal/lpalumbo/llm-bdi-style/data", "testi_randomized.xlsx")
OUTPUT_DIR = os.path.join("leonardo_work/try25_boigenai/Luigi", ) 
ADAPTER_DIR = os.path.join(OUTPUT_DIR, f"{model_name}_adapter_os_20250617")
MERGED_DIR = os.path.join(OUTPUT_DIR, f"{model_name}_finetuned_os_20250617")
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
STEPS = 600
BATCH = 2
MAX_LENGTH = 4096

def log_time(action, start):
    elapsed = time.time() - start
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {action} in {elapsed:.2f}s")
    return None

# --- STEP 1: Load tokenizer
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
log_time("tokenizer loading", t0)

# --- STEP 2: Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
log_time("model loading", t0)

# --- STEP 3: Load and preprocess data
# Load dataset
bdi_dataset = pd.read_excel(EXCEL_PATH)

# Support function for formatting the dataset

instruction_template = """Sei un economista della Banca d’Italia incaricato di riformulare testi prodotti da un giovane analista in fase di bozza.

La bozza riportata di seguito contiene tutte le informazioni rilevanti che devono essere mantenute nel testo finale.

Il tuo compito è riscrivere il testo in modo chiaro, ordinato e coerente, seguendo lo stile formale e professionale delle comunicazioni ufficiali della Banca d’Italia.

Istruzioni:
- Mantieni una stretta aderenza ai fatti, ai dati e ai valori numerici presenti nella bozza.
- Non introdurre nuove informazioni.
- Puoi riformulare liberamente la struttura delle frasi, chiarire i passaggi poco scorrevoli e migliorare la precisione del linguaggio.
- È ammesso un commento qualitativo o una valutazione, purché coerente con i dati forniti.

Scrivi il testo in italiano, in forma discorsiva, con uno stile formale ma accessibile, come nelle pubblicazioni ufficiali della Banca d’Italia.
Non inserire preamboli o conclusioni nella tua risposta, ma solo il testo riformulato della bozza.

### Esempio

**Bozza:**
{os_bozza}

**Testo riformulato:**
{os_testo}

### Ora tocca a te.

**Bozza:**
{bozza}

**Testo riformulato:**
"""

def format_as_text(row, prompt_template=instruction_template):

    
    messages = [
        {"role": "user", "content": instruction_template.format(os_bozza=row["os_input"], os_testo=row["os_testo"], bozza=row["input"])},
        {"role": "assistant", "content": row["testo"]}
    ]
    row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return row

# Apply the formatting
dataset = Dataset.from_pandas(bdi_dataset)
formatted_dataset = dataset.map(format_as_text, remove_columns=dataset.column_names)
split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42) 
train_ds = split_dataset["train"]
val_ds = split_dataset["test"] # The 'test' split is used for validation


def tok(x): return tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
train_ds = train_ds.map(tok, batched=True)
val_ds = val_ds.map(tok, batched=True)

# --- STEP 4: Configure LoRA + SFT with time logging
peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    task_type="CAUSAL_LM", bias="none"
)
sft_config = SFTConfig(
    max_length=MAX_LENGTH,
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=4,
    warmup_steps = 5,  
    max_steps = 600,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    peft_config=peft_config,
    completion_only_loss=True,
    save_pretrained=True,
    logging_steps=1,
    eval_steps=1,
    eval_strategy="steps",
    push_to_hub=False,
    seed = 3407,
    report_to = "none", 
)

# Override SFTTrainer callback to log losses
class TimeLoggingTrainer(SFTTrainer):
    def on_step_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1]
        if "loss" in logs:
            print(f"[step {state.global_step}] los={logs['loss']:.4f}", end="")
        if "eval_loss" in logs:
            print(f"  val_loss={logs['eval_loss']:.4f}")
        else:
            print()

trainer = TimeLoggingTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    args=sft_config
)

# --- STEP 5: Train with time logging
trainer.train()
log_time("training", t0)

# --- STEP 6: Save adapter
trainer.save_model(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}")

# --- STEP 7: Merge adapter and save
merged = trainer.model.merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}")
