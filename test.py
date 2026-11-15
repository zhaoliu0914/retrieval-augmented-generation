import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    num = tokenizer("Hello Model!!!!!")
    print(num["input_ids"])
    print(tokenizer.decode(num["input_ids"]))


def tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)

    inputs = tokenizer("The capital of Italy is Roma", return_tensors="pt")
    print(f"inputs = {inputs}")

    output_nums = model.generate(**inputs, max_new_tokens=20)
    print(f"output_nums = {output_nums}")

    output_text = tokenizer.decode(output_nums[0], skip_special_tokens=True)
    print(f"output_text = {output_text}")


def tokenizer_model_chat():
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)

    question = "What is capital of France?"
    messages = [
        {"role": "user",
         "content": question,
         },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    print(f"inputs = {inputs}")
    print(f"inputs in text = {tokenizer.decode(inputs['input_ids'][0])}")
    print(f"torch.Size = {inputs['input_ids'].shape}")
    print(f"batch size (one prompt) = {inputs['input_ids'].shape[0]}")
    print(f"number of tokens in the prompt = {inputs["input_ids"].shape[-1]}")
    input_token_size = inputs["input_ids"].shape[-1]

    output_nums = model.generate(**inputs,
                                 max_new_tokens=100,
                                 do_sample=False
                                 )
    print(f"output_nums = {output_nums}")

    output_text = tokenizer.decode(output_nums[0])
    print(f"output_text = {output_text}")

    answer_text = tokenizer.decode(output_nums[0][input_token_size:], skip_special_tokens=True)
    print(f"output_text = {answer_text}")


if __name__ == '__main__':
    # tokenizer()
    # tokenizer_model()
    tokenizer_model_chat()
