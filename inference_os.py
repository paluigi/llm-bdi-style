import time
import os

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"


from transformers import pipeline
import torch
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import pandas as pd


tic = time.time()
str_tic = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tic))
print(f"Start time: {str_tic}")
# Set the environment variable to use the GPU

model_path = "/leonardo_work/try25_boigenai/Luigi"
output_path = "/leonardo/home/userexternal/lpalumbo/llm-bdi-style/results"

gemma_path = os.path.join(model_path, "gemma-3-12b-it_finetuned_val") # Using gemma from regular finetuning
qwen_path = os.path.join(model_path, "Qwen2.5-7B-Instruct_finetuned_os_val")
embedding_path= os.path.join(model_path, "jina-embeddings-v3")


input_list = [
    "In questa edizione sono state intervistate 1575 imprese di servizi e industria con almeno 50 addetti tra il 14 giugno e il 16 luglio.",
    "Nel secondo trimestre dell'anno ci sono ancora valutazioni pessimistiche sia della situazione corrente che delle prospettive per i prossimi 12 mesi. Migliorano però le aspettative sulle vendite correnti e per il trimestre prossimo. L'occupazione si prevede in miglioramento in tutti i settori tranne le costruzioni.",
    "Le condizioni per investire continuano ad essere percepite come sfavorevoli, ma le imprese si aspettano comunque un incremento degli investimenti nel 2025."
]


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

# Load examples for one-shot prompting
all_df = pd.read_excel("data/testi_all.xlsx")
all_dicts = all_df.to_dict(orient="records")
client = QdrantClient(":memory:")  # Qdrant is running from RAM.
emb = TextEmbedding("jinaai/jina-embeddings-v3", specific_model_path=embedding_path)

_ = client.create_collection(
    collection_name="examples",
    vectors_config=models.VectorParams(
        size=emb.model.model_description.dim, 
        distance=models.Distance.COSINE
    ),  # size and distance are model dependent
)

# Embedding inputs for search
_ = client.upload_collection(
    collection_name="examples",
    vectors=[emb.embed(doc["input"]).send(None) for doc in all_dicts],
    payload=all_dicts,
    ids=range(len(all_dicts)),
)

# Logging
toc = time.time()
print(f"Qdrant database created, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

input_os_list = []

for i in input_list:
    os_dict = client.query_points(
        collection_name="examples",
        query=emb.embed(i).send(None),
        limit=1,
    ).points[0].payload
    new_row = {
        "input": i,
        "os_testo": os_dict["testo"],
        "os_input": os_dict["input"],

    }
    input_os_list.append(new_row)


input_formatted = [
    [
        {
            "role": "user",
            "content": instruction_template.format(bozza=row["input"], os_bozza=row["os_input"], os_testo=row["os_testo"])
        },
    ]
    for row in input_os_list

]

# Logging
toc = time.time()
print(f"Input list built with vecotor search, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")


# Gemma generation
pipe = pipeline(
    "text-generation",
    model=gemma_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Logging
toc = time.time()
print(f"Gemma loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

output_gemma = []

for par in input_formatted:
    out = pipe(text_inputs=par, max_new_tokens=1024)
    print(out)
    output_gemma.append(out[0]["generated_text"][-1]["content"])

output_file = os.path.join(output_path, "output_gemma_os.txt")
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
pipe = pipeline(
    "text-generation",
    model=qwen_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Logging
toc = time.time()
print(f"Qwen loaded, {toc-tic:.1f} seconds elapsed.")
str_toc = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(toc))
print(f"Time: {str_toc}")

output_qwen = []

for par in input_formatted:
    out = pipe(text_inputs=par, max_new_tokens=1024)
    print(out)
    output_qwen.append(out[0]["generated_text"][-1]["content"])

output_file = os.path.join(output_path, "output_qwen_os.txt")
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



