from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrlParser
from math_verify import verify, parse
import pandas as pd
from dataclasses import dataclass


@dataclass
class MyArguments: 
    model_name: str
    output_dir: str
    run_name: str
    learning_rate: float
    beta: float
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    lr_scheduler_type: str
    logging_steps: float
    bf16: bool
    bf16_full_eval: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    save_steps: int
    max_grad_norm: float
    report_to: str
    use_vllm: bool
    vllm_mode: str
    truncated_cot_prompt_path: str
    max_steps: int
    log_completions: bool
    evaluation_strategy: str
    eval_steps: int
    eval_on_start: bool
    scale_rewards: str
    epsilon: float
    epsilon_high: float
    loss_type: str
    num_iterations: int
    on_truncated: bool
    vllm_tensor_parallel_size: int
    vllm_gpu_memory_utilization: float
    checkpoint_path: str = None


parser = TrlParser(dataclass_types=[MyArguments])
training_args = parser.parse_args_and_config()[0]
print(training_args)

SYSTEM_PROMPT = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>{truncated_cot}"


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval

def reward_func(completions, answer, **kwargs):
    def reward(s, gt):
        # add the last boxed tag
        last_boxed = last_boxed_only_string(s)
        if last_boxed is not None:
            s = last_boxed
        try:
            is_correct = verify(parse(s), parse(gt))
            return 1 if is_correct else 0
        except:
            return 0  # parsing/verification failed
    return [reward(c, gt) for c, gt in zip(completions, answer)]

# 2) Update get_dataset to support "both"
def get_dataset(csv_path: str,):
    # Load RLVR train set
    df = pd.read_csv(csv_path)
    df = df[["question", "truncated_cot", "ground_truth"]].copy()
    df["answer"] = df["ground_truth"].apply(last_boxed_only_string)

    frames = []

    df_trunc = df.copy()
    df_trunc["prompt"] = df_trunc.apply(
        lambda row: SYSTEM_PROMPT.format(
            prompt=row["question"], truncated_cot=row["truncated_cot"]
        ),
        axis=1,
    )
    frames.append(df_trunc[["prompt", "answer"]])


    # Concatenate selected prompt styles
    if len(frames) == 0:
        raise ValueError("No prompts constructed. Check on_truncated flag.")
    df_final = pd.concat(frames, axis=0, ignore_index=True)

    train = Dataset.from_pandas(df_final, preserve_index=False)

    # Load eval set (if any)
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test = test.map(
        lambda x: {
            "prompt": x["problem"],
            "answer": x["answer"],
            "level": x["level"],
        }
    )
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])

    return train, test

train, test = get_dataset(csv_path=training_args.truncated_cot_prompt_path)

model_path = training_args.model_name 
model_name = AutoModelForCausalLM.from_pretrained(model_path)

grpo_config_args = GRPOConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    beta=training_args.beta,
    adam_beta1=training_args.adam_beta1,
    adam_beta2=training_args.adam_beta2,
    weight_decay=training_args.weight_decay,
    lr_scheduler_type=training_args.lr_scheduler_type,
    logging_steps=training_args.logging_steps,
    bf16=training_args.bf16,
    bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    num_generations=training_args.num_generations,
    max_prompt_length=training_args.max_prompt_length,
    max_completion_length=training_args.max_completion_length,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=training_args.use_vllm,
    vllm_mode=training_args.vllm_mode,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    eval_strategy=training_args.evaluation_strategy,
    eval_steps = training_args.eval_steps,
    eval_on_start=training_args.eval_on_start,
    epsilon=training_args.epsilon,
    epsilon_high=training_args.epsilon_high,
    loss_type=training_args.loss_type,
    num_iterations=training_args.num_iterations,
    scale_rewards=training_args.scale_rewards,
    vllm_tensor_parallel_size=training_args.vllm_tensor_parallel_size,
    vllm_gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,
)


trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_func],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)

trainer.train()