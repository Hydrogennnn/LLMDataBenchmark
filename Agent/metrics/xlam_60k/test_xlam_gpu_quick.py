#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick GPU test for xLAM-1b-fc-r model
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("=" * 80)
print("🚀 xLAM-1b-fc-r GPU 快速测试")
print("=" * 80)

# Check GPU availability
print(f"\n🔍 检查环境:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")

# Model configuration
model_path = "/mnt/petrelfs/liuhaoze/models1/xlam-1b-fc-r"

print(f"\n📁 加载模型: {model_path}")
start_time = time.time()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配到 GPU
    torch_dtype=torch.float16,  # 使用 fp16 加速
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set pad_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_time = time.time() - start_time
print(f"✅ 模型加载完成 (耗时: {load_time:.2f}s)")
print(f"   设备: {model.device}")
print(f"   数据类型: {model.dtype}")

# Memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   GPU 内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# Task instruction
task_instruction = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

format_instruction = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
```json
{
  "tool_calls": [
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
  ]
}
```
""".strip()

def convert_to_xlam_tool(tools):
    """Convert OpenAI format to xLAM format"""
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    return tools

def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt

def run_inference(query: str, tools: list):
    print("\n" + "=" * 80)
    print(f"🔍 Query: {query}")
    print("=" * 80)
    
    xlam_format_tools = convert_to_xlam_tool(tools)
    content = build_prompt(task_instruction, format_instruction, xlam_format_tools, query)
    
    messages = [{'role': 'user', 'content': content}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    print(f"⚙️  生成中...")
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    inference_time = time.time() - start
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    print(f"✅ 生成完成 (耗时: {inference_time:.2f}s)")
    print(f"\n📤 输出:")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except:
        print(result)
    
    return result, inference_time

# Test Case 1: Weather query
print("\n" + "=" * 80)
print("📝 测试用例 1: 天气查询")
print("=" * 80)

get_weather_api = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature to return"
            }
        },
        "required": ["location"]
    }
}

search_api = {
    "name": "search",
    "description": "Search for information on the internet",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}

query1 = "What's the weather like in New York in fahrenheit?"
tools1 = [get_weather_api, search_api]
_, time1 = run_inference(query1, tools1)

# Test Case 2: Dataset example
print("\n" + "=" * 80)
print("📝 测试用例 2: 数据集样例")
print("=" * 80)

live_giveaways_api = {
    "name": "live_giveaways_by_type",
    "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                "default": "game"
            }
        },
        "required": []
    }
}

query2 = "Where can I find live giveaways for beta access and games?"
tools2 = [live_giveaways_api]
_, time2 = run_inference(query2, tools2)

# Summary
print("\n" + "=" * 80)
print("📊 性能总结")
print("=" * 80)
print(f"模型加载时间: {load_time:.2f}s")
print(f"测试用例 1 推理时间: {time1:.2f}s")
print(f"测试用例 2 推理时间: {time2:.2f}s")
print(f"平均推理时间: {(time1 + time2) / 2:.2f}s")

if torch.cuda.is_available():
    print(f"\n💾 最终 GPU 内存使用:")
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   已分配: {memory_allocated:.2f}GB")
    print(f"   已预留: {memory_reserved:.2f}GB")

print("\n" + "=" * 80)
print("✨ 测试完成！")
print("=" * 80)

