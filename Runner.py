import os
print(os.getcwd())
import argparse
import torch
import yaml
from transformers import AutoTokenizer
from dynamic_qwen2 import Qwen2ForCausalLM, Qwen2Config  # 从 modified_qwen2.py 导入修改后的类

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_layer_ranges(config):
    layer_ranges = []
    for slice_cfg in config["slices"]:
        for source in slice_cfg["sources"]:
            layer_range = tuple(source["layer_range"])
            layer_ranges.append(layer_range)
    return layer_ranges

def main(args):
    instruction = '''
        def print_prime(n):
            """
            Print all primes between 1 and n
            """
    '''

    torch.set_default_device("cuda")

    # Load the configuration
    config = load_config(args.config_path)
    layer_ranges = get_layer_ranges(config)

    # Load the model configuration with layer_ranges
    model_config = Qwen2Config.from_pretrained(
        args.model_path,
        layer_ranges=layer_ranges,
        trust_remote_code=True,
    )

    # Load the modified Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Tokenize the input string
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        return_attention_mask=False,
    )

    # Generate text using the model
    outputs = model.generate(
        **inputs,
        max_length=200,
    )

    # Decode and print the output
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamic merging with Qwen2 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Qwen2 model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the layer range configuration file")
    args = parser.parse_args()

    main(args)