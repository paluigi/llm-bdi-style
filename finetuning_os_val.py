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
from transformers import TrainingArguments
import numpy as np

from evaluate import load
# Load the metrics from Hugging Face's evaluate library
bleu = load("bleu")
chrf = load("chrf")
wer = load("wer")
cer = load("cer")

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
bdi_dataset = pd.read_excel(os.path.join(input_path,'testi_randomized.xlsx')).to_dict(orient='records')

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

bdi_dataset_formatted = [
    {
        "conversations": [
            {
                "role": "user",
                "content": instruction_template.format(os_bozza=row["os_input"], os_testo=row["os_testo"], bozza=row["input"])
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
# Split dataset into train and validation (90% train, 10% validation)
split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def compute_metrics(p):
    print("=== In compute_metrics ===")

    (preds, labels), _ = p
    del _

    labels[labels == -100] = tokenizer.pad_token_id
    preds[preds == -100] = tokenizer.pad_token_id

    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    except Exception as e:
        print("Error during decoding predictions:", e)
        raise e
    try:
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print("Error during decoding labels:", e)
        raise e

    # For BLEU/CHRF, references should be a list of lists (one inner list per example).
    decoded_labels_bleu = [[label] for label in decoded_labels]

    # Compute metrics.
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    chrfpp_score = chrf.compute(predictions=decoded_preds, references=decoded_labels_bleu, word_order=2)  # CHRF++ (bigram)
    wer_score = wer.compute(predictions=decoded_preds, references=decoded_labels)
    cer_score = cer.compute(predictions=decoded_preds, references=decoded_labels)

    # print("Computed BLEU score:", bleu_score)
    metrics = {
        "bleu": bleu_score["bleu"],
        "chrf": chrf_score["score"],
        "chrf++": chrfpp_score["score"],
        "wer": wer_score,
        "cer": cer_score,
    }

    print(metrics)

    return metrics

# Logging
toc = time.time()
print(f"Dataset loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset, # Can set up evaluation!
    compute_metrics=compute_metrics,
    args = TrainingArguments(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        #num_train_epochs = 4, # Set this for 1 full training run.
        max_steps = 600, # Set this to -1 for full training run, or to a number for partial training
        max_seq_length = 4096, # Choose any for long context!
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        dataset_num_proc=2,
        eval_steps=2,  # Set how frequently to evaluate
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=2, #double as eval steps (to stop time taken during saving)
        greater_is_better=False,
        metric_for_best_model="eval_loss",
    ),
)

if model_name == "gemma-3-12b-it":
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
else:
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
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
model.save_pretrained(os.path.join(model_path, f"{model_name}_adapter_os_val"))  # Local saving
tokenizer.save_pretrained(os.path.join(model_path, f"{model_name}_adapter_os_val"))

# Logging
toc = time.time()
print(f"Adapter saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Save merged model
model.save_pretrained_merged(os.path.join(model_path, f"{model_name}_finetuned_os_val"), tokenizer)

# Logging
toc = time.time()
print(f"Merged model saved, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# # Save GGUF model
# model.save_pretrained_gguf(
#         os.path.join(model_path, f"{model_name}_gguf"),
#         quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
#     )

# # Logging
# toc = time.time()
# print(f"Ollama model saved, {toc-tic:.1f} seconds elapsed.")
# str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
# print(f"Time: {str_toc}")
