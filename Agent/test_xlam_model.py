#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test xLAM-1b-fc-r model for function calling
Based on: https://huggingface.co/Salesforce/xLAM-1b-fc-r
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 80)
print("🚀 Loading xLAM-1b-fc-r Model")
print("=" * 80)

# Set random seed for reproducibility
torch.random.manual_seed(0)

# Model configuration
model_name = "/mnt/petrelfs/liuhaoze/models1/xlam-1b-fc-r"  # 本地路径
cache_dir = None  # 使用本地模型，不需要cache_dir

print(f"\n📁 Loading model from: {model_name}")

# Load model and tokenizer from local path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype="auto",  # 使用 dtype 替代 torch_dtype
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token if not set to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\n✅ Model loaded successfully!")
print(f"Model size: ~1.35B parameters")
print(f"Device: {model.device}")

# Task instruction (recommended by xLAM)
task_instruction = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

# Format instruction
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

# Helper function to convert openai format tools to xLAM format
def convert_to_xlam_tool(tools):
    """Convert OpenAI format to more concise xLAM format"""
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    else:
        return tools

# Helper function to build the input prompt
def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt

# Function to run inference
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
    
    print(f"\n⚙️  Generating response...")
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    print(f"\n✅ Model Output:")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except:
        print(result)
    
    return result

# Test Case 1: Weather query
print("\n" + "=" * 80)
print("📝 Test Case 1: Weather Query")
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
                "description": "The search query, e.g. 'latest news on AI'"
            }
        },
        "required": ["query"]
    }
}

query1 = "What's the weather like in New York in fahrenheit?"
tools1 = [get_weather_api, search_api]
run_inference(query1, tools1)

# Test Case 2: Multiple tool calls
print("\n" + "=" * 80)
print("📝 Test Case 2: Multiple Tool Calls")
print("=" * 80)

query2 = "What's the weather in London and also search for tourist attractions there?"
run_inference(query2, tools1)

# Test Case 3: From the actual dataset
print("\n" + "=" * 80)
print("📝 Test Case 3: From xlam_60k Dataset")
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

query3 = "Where can I find live giveaways for beta access and games?"
tools3 = [live_giveaways_api]
run_inference(query3, tools3)

print("\n" + "=" * 80)
print("✨ All tests completed!")
print("=" * 80)
print(f"📁 Model cache location: {cache_dir}")

