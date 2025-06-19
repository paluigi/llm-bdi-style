import time
import os

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tic = time.time()
str_tic = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tic))
print(f"Start time: {str_tic}")
# Set the environment variable to use the GPU

model_path = "/leonardo_work/try25_boigenai/Luigi"
output_path = "/leonardo/home/userexternal/lpalumbo/llm-bdi-style/results"

gemma_path = os.path.join(model_path, "gemma-3-12b-it_finetuned_val")
qwen_path = os.path.join(model_path, "Qwen2.5-7B-Instruct_finetuned_val")

input_list = [
    "In questa edizione sono state intervistate 1575 imprese di servizi e industria con almeno 50 addetti tra il 14 giugno e il 16 luglio.",
    "Nel secondo trimestre dell'anno ci sono ancora valutazioni pessimistiche sia della situazione corrente che delle prospettive per i prossimi 12 mesi. Migliorano però le aspettative sulle vendite correnti e per il trimestre prossimo. L'occupazione si prevede in miglioramento in tutti i settori tranne le costruzioni.",
    "Le condizioni per investire continuano ad essere percepite come sfavorevoli, ma le imprese si aspettano comunque un incremento degli investimenti nel 2025."
]


instruction_template = """"Sei un economista della Banca d’Italia incaricato di riformulare testi prodotti da un giovane analista in fase di bozza.

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


input_formatted = []

for row in input_list:
    new_conv = [
        {
            "role": "user",
            "content": instruction_template.format(bozza=row["input"], os_bozza=row["os_input"], os_testo=row["os_testo"])
        },
    ]
    input_formatted.append(new_conv)


# Gemma generation
model = AutoModelForCausalLM.from_pretrained(
    gemma_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(gemma_path)

# Logging
toc = time.time()
print(f"Gemma loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

output_gemma = []

for par in input_formatted:
    text = tokenizer.apply_chat_template(
        par,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_gemma.append(response)

output_file = os.path.join(output_path, "output_gemma.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for i, elem in enumerate(output_gemma):
        f.write(str(elem))
        if i != len(output_gemma) - 1:
            f.write("\n\n")

# Logging
toc = time.time()
print(f"Gemma generation done, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Show memory
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak reserved memory = {used_memory} GB.")


# qwen generation
model = AutoModelForCausalLM.from_pretrained(
    qwen_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(gemma_path)

# Logging
toc = time.time()
print(f"Qwen loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

output_qwen = []

for par in input_formatted:
    text = tokenizer.apply_chat_template(
        par,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_qwen.append(response)

output_file = os.path.join(output_path, "output_qwen.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for i, elem in enumerate(output_qwen):
        f.write(str(elem))
        if i != len(output_qwen) - 1:
            f.write("\n\n")

# Logging
toc = time.time()
print(f"Qwen generation done, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

# Show memory
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"Peak reserved memory = {used_memory} GB.")



