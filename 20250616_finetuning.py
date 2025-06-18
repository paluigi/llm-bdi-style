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
from trl import SFTConfig

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


model_path = "/leonardo_work/try25_boigenai/Luigi"
input_path = "/leonardo/home/userexternal/lpalumbo/llm-bdi-style/data"

# Qwen
model_name = "Qwen2.5-7B-Instruct"
tokenizer_name = "Qwen2.5-7B-Instruct"
chat_template = "qwen-2.5"

# # Gemma
# model_name = "gemma-3-12b-it"
# tokenizer_name = "gemma-3-12b-it"
# chat_template = "gemma3"

print(f"Finetuning {model_name} with bfloat16 mixed-precision...")

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
    
    return {"messages": messages}

# Apply the formatting
dataset = Dataset.from_pandas(bdi_dataset)
dataset = dataset.map(format_as_text, remove_columns=dataset.features)

# --- New: Split the dataset ---
dataset = dataset.train_test_split(test_size=0.1, seed=42) # Use a seed for reproducibility

# Logging
toc = time.time()
print(f"Dataset formatted, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16


# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# Load model in 16-bit quantized mode
model = AutoModelForCausalLM.from_pretrained(
    os.path.join(model_path, tokenizer_name),
    # use bf16 precision for computation: a 16 bit precision designed to provide a similar range as FP32
    device_map="auto",
    **model_kwargs
)
print(f"bf16 model GPU VRAM usage: {get_model_size(model)/1024**3:.2f} GB")
# Logging
toc = time.time()
print(f"Model loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, tokenizer_name))
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

# Logging
toc = time.time()
print(f"Tokenizer loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir=os.path.join(model_path, f"{model_name}_adapter_bf16"), # New directory for this run
    max_seq_length=4096,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    # num_train_epochs=3,                     # number of training epochs
    max_steps = 600,                     # total number of training steps to perform
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="none",                # report metrics to none
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)
