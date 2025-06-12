import time
import os

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Utility function to get model size:
def get_model_size(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_size += buffer.nelement() * buffer.element_size()
    return model_size


tic = time.time()
str_tic = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tic))
print(f"Start time: {str_tic}")
# Set the environment variable to use the GPU

model_path = "/leonardo_work/try25_boigenai/Luigi"
input_path = "/leonardo/home/userexternal/lpalumbo/llm-bdi-style/data"

# Qwen
model_name = "Qwen2.5-7B-Instruct"
tokenizer_name = "Qwen2.5-7B-Instruct"
chat_template = "qwen-2.5"

# # Mistral
# model_name = "Mistral-Nemo-Instruct-FP8-2407"
# tokenizer_name = "Mistral-Nemo-Instruct-FP8-2407"
# chat_template = "mistral"

# # Gemma
# model_name = "gemma-3-12b-it"
# tokenizer_name = "gemma-3-12b-it"
# chat_template = "gemma3"

print(f"Finetuning {model_name} with bfloat16 mixed-precision...")

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

# Training arguments for bfloat16
training_args = TrainingArguments(
    output_dir=os.path.join(model_path, f"{model_name}_adapter_bf16"), # New directory for this run
    per_device_train_batch_size=2,        # IMPORTANT: Reduced batch size for higher memory usage
    gradient_accumulation_steps=4,        # Increase accumulation to maintain effective batch size
    warmup_steps = 5,
    learning_rate=2e-5,                   # Often a lower learning rate is better for full-precision tuning
    max_steps = 600,
    weight_decay = 0.01,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=1,
    optim="adamw_torch",                  # Standard AdamW optimizer
    bf16=True,                            # Enable bfloat16 mixed-precision training
    push_to_hub=False,
    seed = 3407,
    report_to = "none", # Use this for WandB etc
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, tokenizer_name))
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

# Define the response template for the data collator
if model_name == "gemma-3-12b-it":
    response_template = "<start_of_turn>model\n"
else:
    response_template = "<|im_start|>assistant\n"

# Logging
toc = time.time()
print(f"Tokenizer loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")


# Load dataset
bdi_dataset = pd.read_excel(os.path.join(input_path,'testi_randomized.xlsx'))

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

**Bozza:**
{bozza}

**Testo riformulato:**
"""

def format_as_text(row, prompt_template=instruction_template):

    
    messages = [
        {"role": "user", "content": instruction_template.format(bozza=row["input"])},
        {"role": "assistant", "content": row["testo"]}
    ]
    row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return row

# Apply the formatting
dataset = Dataset.from_pandas(bdi_dataset)
formatted_dataset = dataset.map(format_as_text, remove_columns=bdi_dataset.column_names)

# --- New: Split the dataset ---
split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42) # Use a seed for reproducibility
train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"] # The 'test' split is used for validation


# Initialize the data collator
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)


# Logging
toc = time.time()
print(f"Dataset formatted, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")


# Load model in 16-bit quantized mode
model = AutoModelForCausalLM.from_pretrained(
    os.path.join(model_path, tokenizer_name),
    # use bf16 precision for computation: a 16 bit precision designed to provide a similar range as FP32
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)
print(f"bf16 model GPU VRAM usage: {get_model_size(model)/1024**3:.2f} GB")



# Logging
toc = time.time()
print(f"Model loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Initialize the SFTTrainer with train and validation datasets
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,        # Pass the training split
    eval_dataset=validation_dataset,    # Pass the validation split
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=4096,
)

# --- 5. Training and Saving ---

print("Starting bfloat16 fine-tuning with validation...")
trainer.train()

# Logging
toc = time.time()
print(f"Model finetuned, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Save the LoRA adapter
adapter_path = training_args.output_dir
trainer.save_model(adapter_path)
print(f"Final LoRA adapter saved to {adapter_path}")

# Logging
toc = time.time()
print(f"Adapter saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# --- 6. Merge Adapter and Save Full Model ---
print("Merging the LoRA adapter with the base model...")

del model
del trainer
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    os.path.join(model_path, tokenizer_name),
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model_with_adapter.merge_and_unload()
print("Merging complete.")

merged_model_path = os.path.join(model_path, f"{model_name}_merged_bf16")
os.makedirs(merged_model_path, exist_ok=True)
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Full merged model saved to {merged_model_path}")
# Logging
toc = time.time()
print(f"Merged model saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")