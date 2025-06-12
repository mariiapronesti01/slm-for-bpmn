import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import wandb
import os

def formatting_prompts_func(examples):
    convos = [
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        for instruction, input_text, output_text in zip(
            examples["instruction"], examples["input"], examples["output"]
        )
    ]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth")

    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=6024, help="Maximum sequence length for the model")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "auto"], default="auto", help="Data type for the model (float16, bfloat16, or auto)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--hf_token", type=str, required=True, default="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", help="Hugging Face token for authentication")
    parser.add_argument("--train_data_path", type=str, default="./dataset/train.json", help="Path to the training data file")
    parser.add_argument("--val_data_path", type=str, default="./dataset/val.json", help="Path to the validation data file")
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save the model checkpoints and outputs")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging")
    parser.add_argument("--wandb_entity", type=str, default="mariapronesti-mp-universit-di-trieste", help="WandB entity name for logging")
    parser.add_argument("--wandb_project", type=str, default="huggingface", help="WandB project name for logging")
    parser.add_argument("--wandb_run_id", type=str, help="WandB run ID for logging")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name for logging")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for training")
    

    args = parser.parse_args()
    
    # Check if model name exists
    if not os.path.exists(args.model_name):
        raise FileNotFoundError(f"Model name {args.model_name} does not exist.")
    
    # Check paths exist
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data path {args.train_data_path} does not exist.")
    if not os.path.exists(args.val_data_path):
        raise FileNotFoundError(f"Validation data path {args.val_data_path} does not exist.")
    
    # Check output directory exists or create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    # Map dtype string to actual torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": None
    }
    dtype = dtype_map[args.dtype]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        token=args.hf_token,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    dataset = load_dataset("json", data_files={"train": args.train_data_path, "test": args.val_data_path})
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run_id,
            name=args.wandb_run_name,
            resume="allow" if args.resume else None
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=args.n_epochs,
            learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_steps=30,
            save_total_limit=5,
            output_dir=args.output_dir,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    trainer.train(resume_from_checkpoint=args.resume)
