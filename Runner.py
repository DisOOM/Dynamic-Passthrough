import torch
from transformers import AutoTokenizer
from modeling import Qwen2ForCausalLM
from configuration_qwen2 import Qwen2Config

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 指定本地模型路径
model_path = "D:/Qwen1.8B"

# 加载并修改配置
config = Qwen2Config.from_pretrained(model_path)
config.num_virtual_layers = 4  # 设置虚拟层数量

# 使用修改后的配置加载模型
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置提示
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 准备输入
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 打印结果
print("Response:", response)

# 为了在PowerShell中暂停执行，等待用户输入
input("Press Enter to exit...")