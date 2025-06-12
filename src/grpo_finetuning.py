from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset

from unsloth import is_bfloat16_supported
from unsloth import get_chat_template

from utils.grpo_rewards import *

import argparse
import os


def get_dataset(dataset_path, split = "train") -> Dataset:
    data = load_dataset("json", data_files={"train": dataset_path})[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': x['instruction']},
            {'role': 'user', 'content': x['input']}
        ],
        'true_actors': x['true_actors'],
        'true_task' : x['true_task']
    }) 
    return data 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Fine-tuning")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--max_seq_length", type=int, default=4000, help="Max sequence length")
    parser.add_argument("--n_generations", type=int, default=6, help="Number of generations")
    parser.add_argument("--dataset_path", type=str, default="dataset/GRPO/trainGRPO.json", help="Path to the dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving the model")
    
    args = parser.parse_args()
    
    
    # Check if model path exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
    
    # Check if dataset path exist
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")
    
    # Check if output directory exists or create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    
    model_path = args.model_path
    lora_rank = args.lora_rank
    max_seq_length = args.max_seq_length
    n_generations = args.n_generations
    dataset_path = args.dataset_path
    use_wandb = args.use_wandb
    output_dir = args.output_dir
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        fast_inference = False,
        gpu_memory_utilization = 0.7,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1"
    )

    dataset = get_dataset(dataset_path)

    training_args = GRPOConfig(
        use_vllm = False, # use vLLM for fast inference!
        learning_rate = 1e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 4, 
        num_generations = n_generations, 
        max_prompt_length = 1000,
        max_completion_length = 3000,
        num_train_epochs = 1, 
        save_steps=10,
        save_total_limit=5,
        report_to = "wandb" if use_wandb else None, # Can use Weights & Biases
        output_dir = output_dir,
    )


    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            validation_reward_func,
            task_reward_funct,
            actor_reward_funct,
            exist_gateway_reward_func,
            dead_activities_reward_func,
            exist_end_event_reward_func,
            option_to_complete_reward_func,
            no_infinite_loop_reward_func,
            safeness_reward_func,
            correct_gateway_reward_func,
            sincronization_reward_funct,
            task_type_reward_funct
        ],
        args = training_args,
        train_dataset = dataset,
    )

    trainer.train()