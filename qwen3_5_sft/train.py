import torch

from transformers import Qwen3_5ForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from datasets import load_dataset


DATA_PATH = "/wbl-fast/usrs/ethan/small-agent/sft/data/balanced-dataset/dataset"
BASE_CHECKPOINT_PATH = "/wbl-fast/usrs/ethan/small-agent/sft/checkpoints/Qwen3.5-4B"
OUTPUT_PATH = "/wbl-fast/usrs/ethan/small-agent/qwen3_5_sft"
NUM_PROCS = 32


model: Qwen3_5ForCausalLM = Qwen3_5ForCausalLM.from_pretrained(
    pretrained_model_name_or_path=BASE_CHECKPOINT_PATH,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=BASE_CHECKPOINT_PATH,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    path=DATA_PATH,
    split="train",
)


def _apply_chat_template(example: dict) -> dict:
    return {
        "text": tokenizer.apply_chat_template(
            conversation=example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = dataset.map(
    function=_apply_chat_template,
    num_proc=NUM_PROCS,
)

training_args = SFTConfig(
    output_dir=OUTPUT_PATH,
    assistant_only_loss=True,
    dataset_text_field="text",
    dataset_num_proc=NUM_PROCS,
    optim="adam_w",
    lr_scheduler_type="constant",
    learning_rate=1e-7,
    weight_decay=0.1,
    warmup_steps=1000,
    num_train_epochs=1,
    use_liger_kernel=True,
    packing=True,
    max_length=32768,
    per_device_train_batch_size=1,
    torch_compile=True,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)
trainer.train()
