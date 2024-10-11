import datasets
import re
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16, float16
import os
import deepspeed
from torch.utils.data import DataLoader
import argparse
from mpi4py import MPI
import torch
from torch.utils.data import DataLoader, RandomSampler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


# set key environment variables for the MPI process
def set_mpi(masteradd):
    comm = MPI.COMM_WORLD 
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["RANK"] = str(comm.Get_rank())
    os.environ['WORLD_SIZE'] = str(comm.Get_size())
    os.environ["MASTER_ADDR"] = masteradd
    os.environ["MASTER_PORT"] = "1234" #add masterport number here

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument("-m", "--master_add", dest="masteradd", help="Master address for MPI")
args = parser.parse_args()

# Set up MPI environment variables
set_mpi(args.masteradd)

# Adjust to use environment variable if local_rank is not explicitly set
if args.local_rank == -1:
    args.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", -1))

if args.local_rank == -1:
    raise ValueError("Local rank is not set. Ensure that the distributed launcher correctly assigns local ranks.")


# Set device based on local rank
torch.cuda.set_device(args.local_rank)
device = torch.device(f"cuda:{args.local_rank}")

# Debugging information
import socket
hostname = socket.gethostname()
print(f"Hostname: {hostname}, Node: {os.environ.get('MASTER_ADDR', 'Unknown')}, Local Rank: {args.local_rank}, Using device: {device}")
print(f"GPU Device Name: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

torch.set_default_device('cuda') 


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Call the original collator to process the batch
        batch = super().__call__(features)
        
        # Move the batch to the GPU
        device ='cuda'
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, data_collator, *kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, data_collator=data_collator, *kwargs)

    def get_train_dataloader(self):
        generator = torch.Generator(device="cuda")

        # Create the custom DataLoader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(self.train_dataset, generator=generator),
            collate_fn=self.data_collator,
            pin_memory=False
        )
        return train_dataloader
      
torch.cuda.empty_cache()

def main():
    model_id='Meta-Llama-3.1-70B-Instruct'#your path to hugging face model
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    timestamp = time.strftime("%Y%m%d_%H")
    filename=re.search(r'(?<=Llama-3\.1-)\d+B', model_id).group(0)
    out_dir = f'finetuned_models/{filename}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    dataset=datasets.load_dataset('json', data_files="training_data.jsonl", split='train') #tokenized training data with input_ids, attention_mask, and labels
 
    training_args = TrainingArguments(
        gradient_checkpointing_kwargs={"use_reentrant": False},
        output_dir=out_dir,
        deepspeed='deepspeed_config.json',
        overwrite_output_dir=True,
        seed=42,
        do_eval=False,
        logging_strategy="steps",
        logging_steps=1000, 
        learning_rate=2e-5,
        warmup_steps=50,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # tf32=True,
        bf16=True, 
        # fp16=True,
        weight_decay=0.1,
        push_to_hub=False,
        save_strategy="steps",
        num_train_epochs=20,
        save_steps=50, 
        save_on_each_node=False,
        save_total_limit=5,
        optim="paged_adamw_32bit", # adamw_bnb_8bit (2 bytes), adafactor (4 bytes), paged_adamw_8bit can page out to CPU memory
        )
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=bfloat16,
        )
    model.to('cuda')
    
    
    data_collator = CustomDataCollatorForSeq2Seq(tokenizer)
    trainer = CustomTrainer(
          model=model,                  
          args=training_args,            
          train_dataset=dataset,       
          data_collator=data_collator   
      )

    trainer.train()
    trainer.save_model()
    print('Training DONE')

main()
