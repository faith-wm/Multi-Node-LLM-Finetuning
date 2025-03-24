import os
import re
import time
import json
import yaml
import torch
import datasets
import argparse
import socket
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch.utils.data import DataLoader, RandomSampler
from mpi4py import MPI
from torch import bfloat16

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


def set_mpi(masteradd):
    comm = MPI.COMM_WORLD
    os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    os.environ["RANK"] = str(comm.Get_rank())
    os.environ['WORLD_SIZE'] = str(comm.Get_size())
    os.environ["MASTER_ADDR"] = masteradd
    os.environ["MASTER_PORT"] = "1234"


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to("cuda")
        return batch


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        generator = torch.Generator(device="cuda")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(self.train_dataset, generator=generator),
            collate_fn=self.data_collator,
            pin_memory=False
        )


def load_config(path):
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Only .json and .yml config files are supported")


def main(config):
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    timestamp = time.strftime("%Y%m%d_%H")
    modelname = re.search(r'(?<=Llama-3\.1-)\d+B', model_id).group(0)
    out_dir = f'{config["output_base_dir"]}/{modelname}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    dataset = datasets.load_dataset('json', data_files=config["dataset_path"], split='train')

    training_args = TrainingArguments(
        output_dir=out_dir,
        **config["training_args"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=bfloat16,
    ).to("cuda")

    data_collator = CustomDataCollatorForSeq2Seq(tokenizer)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model()
    print("Training DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json or config.yml")
    parser.add_argument("--master_add", required=True, help="Master address for MPI")
    args = parser.parse_args()

    set_mpi(args.master_add)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"Hostname: {socket.gethostname()}, Rank: {os.environ['RANK']}, Device: cuda:{os.environ['LOCAL_RANK']}")

    config = load_config(args.config)
    main(config)
