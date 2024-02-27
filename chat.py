import os
import torch
import time
from modelscope import AutoTokenizer, snapshot_download
from auto_gptq import AutoGPTQForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model_tokenizer(model_path):
    """
    Load model and tokenizer based on the given model name or local path of downloaded model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              lagecy=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")

    model = AutoGPTQForCausalLM.from_quantized(model_path,
                                               inject_fused_attention=False,
                                               inject_fused_mlp=False,
                                               use_safetensors=True,
                                               use_cuda_fp16=True,
                                               disable_exllama=False,
                                               device_map='auto'  # Support multi-gpus
                                               )
    return model, tokenizer


def inference(model, tokenizer, prompt):
    """
    Uset the given model and tokenizer to generate an answer for the speicifed prompt.
    """
    st = time.time()
    prompt = prompt if prompt.endswith('\n') else f'{prompt}\n'
    inputs = f"<s>human\n{prompt}<s>bot\n"

    input_ids = tokenizer.encode(inputs,
                                 return_tensors="pt",
                                 padding=True,
                                 add_special_tokens=False).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            top_p=0.95,
            temperature=0.1,
            do_sample=True,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    print(f'generated tokens num is {len(generated_ids[0][input_ids.size(1):])}')
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(f'generate text is {outputs[0][len(inputs):]}')
    latency = time.time() - st
    print('latency is {} seconds'.format(latency))


if __name__ == "__main__":
    model_dir = '../CodeFuse-DeepSeek-33B-4bits'

    prompt = 'Please write a QuickSort program in Python'

    model, tokenizer = load_model_tokenizer(model_dir)
    inference(model, tokenizer, prompt)
