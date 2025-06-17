import os
import time

# ATTENTION: These two variables must be defined BEFORE loading transformers. Otherwise,
# the library will search the models using the default paths.
# Model folder. Change it if needed. Must be the same dir used in the download_models.sh script
os.environ["HF_HOME"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_CACHE"]="/leonardo_work/try25_boigenai/Luigi"
os.environ["HF_HUB_OFFLINE"]="1"


import json
import pandas as pd
import torch
import time
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
import evaluate
from typing import Dict, List, Optional


class TrainingProgressCallback(TrainerCallback):
    """Custom callback to print training and validation losses"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print training loss
            if 'loss' in logs:
                print(f"[{timestamp}] Step {state.global_step}: Training Loss = {logs['loss']:.4f}")
            
            # Print validation loss
            if 'eval_loss' in logs:
                print(f"[{timestamp}] Step {state.global_step}: Validation Loss = {logs['eval_loss']:.4f}")
            
            # Print learning rate if available
            if 'learning_rate' in logs:
                print(f"[{timestamp}] Step {state.global_step}: Learning Rate = {logs['learning_rate']:.2e}")
    
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        """Called at the end of each epoch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ========== Epoch {state.epoch} completed ==========\n")


class LoraFineTuner:
    def __init__(
        self,
        model_path: str,
        excel_file_path: str,
        output_dir: str,
        adapter_output_dir: str,
        merged_model_output_dir: str
    ):
        """
        Initialize the LORA fine-tuner
        
        Args:
            model_path: Path to locally downloaded model
            excel_file_path: Path to Excel file with training data
            output_dir: Directory for training outputs
            adapter_output_dir: Directory to save LORA adapter
            merged_model_output_dir: Directory to save merged model
        """
        self.model_path = model_path
        self.excel_file_path = excel_file_path
        self.output_dir = output_dir
        self.adapter_output_dir = adapter_output_dir
        self.merged_model_output_dir = merged_model_output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(adapter_output_dir, exist_ok=True)
        os.makedirs(merged_model_output_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Initialize timing
        self.start_time = None
        self.step_times = {}

    def _log_step(self, step_name: str):
        """Log timestamp and elapsed time for a step"""
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.start_time is None:
            self.start_time = current_time
            elapsed_total = 0
        else:
            elapsed_total = current_time - self.start_time
        
        if step_name in self.step_times:
            elapsed_step = current_time - self.step_times[step_name]
            print(f"[{timestamp}] {step_name} completed in {elapsed_step:.2f}s (Total elapsed: {elapsed_total:.2f}s)")
        else:
            self.step_times[step_name] = current_time
            print(f"[{timestamp}] Starting {step_name}...")
        
        return current_time

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer from local directory"""
        # Load tokenizer
        self._log_step("Loading tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self._log_step("Loading tokenizer")
        
        # Load model
        self._log_step("Loading model")
        
        # # Configure quantization for memory efficiency
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            # quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        self._log_step("Loading model")

    def prepare_dataset(self, test_size: float = 0.1, conversation_column: str = "conversation"):
        """
        Read Excel file and prepare dataset in conversation format
        
        Args:
            test_size: Fraction of data to use for validation
            conversation_column: Name of column containing conversation data
        """
        self._log_step("Preparing dataset")
        
        # Read Excel file
        df = pd.read_excel(self.excel_file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Loaded {len(df)} rows from Excel file")

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
            row["text"] = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return row

        # Apply the formatting
        dataset = Dataset.from_pandas(df)
        formatted_dataset = dataset.map(format_as_text, remove_columns=dataset.column_names)
        split_dataset = formatted_dataset.train_test_split(test_size=test_size, seed=42) 
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"] # The 'test' split is used for validation
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training set: {len(self.train_dataset)} samples")
        print(f"[{timestamp}] Validation set: {len(self.eval_dataset)} samples")
        
        self._log_step("Preparing dataset")

    def setup_lora_config(self):
        """Configure LORA parameters"""
        self._log_step("Setting up LORA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            lora_dropout=0.1,  # Dropout probability
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target modules for LORA (adjust based on your model architecture)
            bias="none",
        )
        
        # Apply LORA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        self._log_step("Setting up LORA configuration")
        
        return lora_config

    def create_sft_config(self):
        """Create SFT configuration"""
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            max_steps = 600,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            #save_steps=500,
            logging_steps=1,  # More frequent logging to see training progress
            learning_rate=2e-5,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",
            evaluation_strategy="steps",  # Evaluate at regular steps
            eval_steps=5,  # Evaluate every 5 steps
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=42,
            data_seed=42,
            dataloader_num_workers=0,
            max_seq_length=4096,
            packing=False,  # Set to True if you want to pack sequences
            dataset_text_field="text",
            remove_unused_columns=False,
        )
        
        return sft_config

    def train_model(self):
        """Perform LORA fine-tuning using SFTTrainer"""
        self._log_step("Training model")
        
        # Setup LORA
        lora_config = self.setup_lora_config()
        
        # Create SFT configuration
        sft_config = self.create_sft_config()
        
        # Initialize custom callback for detailed logging
        progress_callback = TrainingProgressCallback()
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
            data_collator=None,  # Will use default
            callbacks=[progress_callback],  # Add custom callback
        )
        
        # Print initial training information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ========== Starting Training ==========")
        print(f"[{timestamp}] Total training samples: {len(self.train_dataset)}")
        print(f"[{timestamp}] Total validation samples: {len(self.eval_dataset)}")
        print(f"[{timestamp}] Number of epochs: {sft_config.num_train_epochs}")
        print(f"[{timestamp}] Batch size per device: {sft_config.per_device_train_batch_size}")
        print(f"[{timestamp}] Gradient accumulation steps: {sft_config.gradient_accumulation_steps}")
        print(f"[{timestamp}] Learning rate: {sft_config.learning_rate}")
        print(f"[{timestamp}] ===============================================\n")
        
        # Start training
        train_start_time = time.time()
        trainer.train()
        train_end_time = time.time()
        
        training_time = train_end_time - train_start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ========== Training Completed ==========")
        print(f"[{timestamp}] Total training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
        print(f"[{timestamp}] ==========================================\n")
        
        # Save training metrics
        training_history = trainer.state.log_history
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
        
        # Print final metrics if available
        if training_history:
            final_metrics = training_history[-1] if training_history else {}
            if 'train_loss' in final_metrics:
                print(f"[{timestamp}] Final Training Loss: {final_metrics['train_loss']:.4f}")
            if 'eval_loss' in final_metrics:
                print(f"[{timestamp}] Final Validation Loss: {final_metrics['eval_loss']:.4f}")
        
        self._log_step("Training model")
        
        return trainer

    def save_adapter_and_merged_model(self, trainer):
        """Save LORA adapter and merged model"""
        self._log_step("Saving LORA adapter")
        
        # Save LORA adapter
        trainer.model.save_pretrained(self.adapter_output_dir)
        self.tokenizer.save_pretrained(self.adapter_output_dir)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] LORA adapter saved to {self.adapter_output_dir}")
        
        self._log_step("Saving LORA adapter")
        
        # Merge adapter with base model and save
        self._log_step("Merging and saving model")
        
        # Load base model without quantization for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Apply LORA config and load adapter weights
        from peft import PeftModel
        
        # Load the trained adapter
        merged_model = PeftModel.from_pretrained(base_model, self.adapter_output_dir)
        merged_model = merged_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(self.merged_model_output_dir)
        self.tokenizer.save_pretrained(self.merged_model_output_dir)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Merged model saved to {self.merged_model_output_dir}")
        
        self._log_step("Merging and saving model")

    def run_complete_training(self):
        """Run the complete training pipeline"""
        try:
            start_time = datetime.now()
            print(f"\n{'='*60}")
            print(f"LORA Fine-tuning Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            # Step 1: Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Step 2: Prepare dataset
            self.prepare_dataset()
            
            # Step 3: Train model
            trainer = self.train_model()
            
            # Step 4: Save adapter and merged model
            self.save_adapter_and_merged_model(trainer)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print(f"\n{'='*60}")
            print(f"LORA Fine-tuning Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total pipeline time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"{'='*60}\n")
            
        except Exception as e:
            error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{error_time}] Error during training: {str(e)}")
            raise


def main():
    """Main function to run the fine-tuning"""
    # Qwen
    model_name = "Qwen2.5-7B-Instruct"
    chat_template = "qwen-2.5"

    # # Gemma
    # model_name = "gemma-3-12b-it"
    # chat_template = "gemma3"
    OUTPUT_DIR = "leonardo_work/try25_boigenai/Luigi"
    # Configuration - Modify these paths according to your setup
    config = {
        "model_path": os.path.join("leonardo_work/try25_boigenai/Luigi", model_name) ,  # Path to your downloaded model
        "excel_file_path": os.path.join("leonardo/home/userexternal/lpalumbo/llm-bdi-style/data", "testi_randomized.xlsx"),
        "output_dir": OUTPUT_DIR,  # Training output directory
        "adapter_output_dir": os.path.join(OUTPUT_DIR, f"{model_name}_adapter_os_20250617"),  # LORA adapter output directory
        "merged_model_output_dir": os.path.join(OUTPUT_DIR, f"{model_name}_finetuned_os_20250617")
    }
    
    # Create fine-tuner instance
    fine_tuner = LoraFineTuner(
        model_path=config["model_path"],
        excel_file_path=config["excel_file_path"],
        output_dir=config["output_dir"],
        adapter_output_dir=config["adapter_output_dir"],
        merged_model_output_dir=config["merged_model_output_dir"]
    )
    
    # Run complete training pipeline
    fine_tuner.run_complete_training()


if __name__ == "__main__":
    main()
