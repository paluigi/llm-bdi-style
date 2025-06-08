import time
import os

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


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

# Load model and tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = os.path.join(model_path, model_name),
    max_seq_length = 4096, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    dtype = torch.bfloat16,
    full_finetuning = False, # [NEW!] We have full finetuning now!
    use_exact_model_name = True, # [NEW!] Use the exact model name for loading
    trust_remote_code = True, 
)

# Logging
toc = time.time()
print(f"Model loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template,
)

# Logging
toc = time.time()
print(f"Tokenizer loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Load dataset
bdi_dataset = pd.read_excel(os.path.join(input_path,'testi_randomized.xlsx').to_dict(orient='records'))

instruction_template = """SSei un economista della Banca d’Italia incaricato di riformulare testi prodotti da un giovane analista in fase di bozza.

La bozza riportata di seguito contiene tutte le informazioni rilevanti che devono essere mantenute nel testo finale:

{bozza}

Il tuo compito è riscrivere il testo in modo chiaro, ordinato e coerente, seguendo lo stile formale e professionale delle comunicazioni ufficiali della Banca d’Italia.

Istruzioni:
- Mantieni una stretta aderenza ai fatti, ai dati e ai valori numerici presenti nella bozza.
- Non introdurre nuove informazioni.
- Puoi riformulare liberamente la struttura delle frasi, chiarire i passaggi poco scorrevoli e migliorare la precisione del linguaggio.
- È ammesso un commento qualitativo o una valutazione, purché coerente con i dati forniti.

Scrivi il testo in italiano, in forma discorsiva, con uno stile formale ma accessibile, come nelle pubblicazioni ufficiali della Banca d’Italia.

"""

bdi_dataset_formatted = [
    {
        "conversations": [
            {
                "role": "user",
                "content": instruction_template.format(bozza=row["input"])
            },
            {
                "role": "assistant",
                "content": row["testo"]
            }
        ]
    }
    for row in bdi_dataset

]

bdi_dataset_formatted_dataset = Dataset.from_list(bdi_dataset_formatted)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
    return { "text" : texts, }

dataset = bdi_dataset_formatted_dataset.map(formatting_prompts_func, batched = True)

# Logging
toc = time.time()
print(f"Dataset loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        dataset_num_proc=2,
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()

# Logging
toc = time.time()
print(f"Training done, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")

# Save model adapter
model.save_pretrained(os.path.join(model_path, f"{model_name}_adapter"))  # Local saving
tokenizer.save_pretrained(os.path.join(model_path, f"{model_name}_adapter"))

# Logging
toc = time.time()
print(f"Adapter saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Save merged model
model.save_pretrained_merged(os.path.join(model_path, f"{model_name}_finetuned"), tokenizer)

# Logging
toc = time.time()
print(f"Merged model saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Save GGUF model
model.save_pretrained_gguf(
        os.path.join(model_path, f"{model_name}_gguf"),
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

# Logging
toc = time.time()
print(f"Ollama model saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")
