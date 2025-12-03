#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xLAM-60k Dataset Basic Quality Evaluation
Evaluates: Format Correctness, Executability, Diversity, Data Noise
"""

import json
import sys
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

SEMANTIC_PROMPT_TEMPLATE = """You are a meticulous auditor who checks whether tool calls correctly satisfy a user query.

==================== INPUT ====================
User Query:
{query}

Available Tool Definitions (JSON):
{tools}

Provided Tool Calls (JSON):
{answers}
===============================================

Task:
1. Determine whether the tool calls fully satisfy the query intent.
2. If anything is missing, irrelevant, or incorrectly parameterized, explain it.
3. Consider parameter values, number of calls, and whether the chosen tools are sufficient.

Output ONLY valid JSON with this schema:
{{
  "pass": true/false,
  "score": 0-100,
  "issues": ["..."],
  "reason": "short explanation referencing the query, tools, and calls"
}}

Rules:
- Be strict: if even part of the query is unmet, set pass=false and list the missing pieces.
- If multiple issues exist, enumerate each one inside "issues".
- The "reason" must reference concrete elements from the query/tools/calls so the decision is auditable."""


class XLAMBasicEvaluator:
    """Basic quality metrics evaluator for xLAM-60k dataset"""
    
    def __init__(self, dataset_path: str, output_dir: str = None,
                 semantic_model: str = None,
                 semantic_batch_size: int = 2,
                 semantic_max_samples: int = 200,
                 semantic_max_new_tokens: int = 512,
                 use_accelerate: bool = False):
        self.dataset_path = dataset_path
        self.data = []
        self.stats = {
            'total_samples': 0,
            'format_errors': [],
            'executability_errors': [],
            'categories': Counter(),
            'query_lengths': [],
            'num_tools': [],
            'num_calls': [],
            'duplicates': [],
        }
        
        # Initialize Accelerator if requested
        self.accelerator = None
        if use_accelerate:
            if not HAS_ACCELERATE:
                raise ImportError("Accelerate not installed. Run: pip install accelerate")
            self.accelerator = Accelerator()
            if self.accelerator.is_main_process:
                print(f"🚀 Accelerate initialized: {self.accelerator.num_processes} processes")
        
        # Set output directory based on dataset name
        if output_dir is None:
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            # 输出到 base_metric 子目录
            self.output_dir = os.path.join(os.path.dirname(dataset_path), f"{dataset_name}_eval_logs", "base_metric")
        else:
            self.output_dir = output_dir
        
        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Log output directory: {self.output_dir}")
        
        # Semantic executability settings
        self.semantic_model_path = semantic_model
        self.semantic_batch_size = semantic_batch_size
        self.semantic_max_samples = semantic_max_samples
        self.semantic_max_new_tokens = semantic_max_new_tokens
        self.semantic_tokenizer = None
        self.semantic_model = None

    def _load_semantic_model(self):
        """Initialize local LLM for semantic executability"""
        if not self.semantic_model_path:
            if not self.accelerator or self.accelerator.is_main_process:
                print("⚠️  Semantic model path not provided. Skipping semantic executability.")
            return False
        if not HAS_TRANSFORMERS:
            if not self.accelerator or self.accelerator.is_main_process:
                print("❌ transformers / torch not installed. Cannot run semantic executability.")
            return False
        if self.semantic_model is not None:
            return True
        
        if not self.accelerator or self.accelerator.is_main_process:
            print(f"🔄 Loading semantic model: {self.semantic_model_path}")
        
        self.semantic_tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_path)
        if self.semantic_tokenizer.pad_token is None:
            self.semantic_tokenizer.pad_token = self.semantic_tokenizer.eos_token
        self.semantic_tokenizer.padding_side = "left"
        
        if self.accelerator:
            # Accelerate mode: each process loads its shard directly onto the assigned GPU.
            # We skip accelerator.prepare() to avoid wrapping with DDP, which would replicate
            # the full 32B model on each device and cause OOM.
            self.semantic_model = AutoModelForCausalLM.from_pretrained(
                self.semantic_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": self.accelerator.process_index}
            )
        else:
            # Single GPU mode: use device_map="auto"
            try:
                self.semantic_model = AutoModelForCausalLM.from_pretrained(
                    self.semantic_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except ValueError:
                # device_map not supported, fallback to cpu
                self.semantic_model = AutoModelForCausalLM.from_pretrained(
                    self.semantic_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.semantic_model.to(device)
        
        if not self.accelerator or self.accelerator.is_main_process:
            print("✅ Semantic model loaded")
        self.semantic_model.eval()
        return True

    def _semantic_generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts using the semantic model"""
        if not self.semantic_model or not self.semantic_tokenizer:
            return ["" for _ in prompts]
        
        chat_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.semantic_tokenizer, "apply_chat_template"):
                text = self.semantic_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to plain prompt
                text = prompt
            chat_texts.append(text)
        
        inputs = self.semantic_tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        if hasattr(self.semantic_model, "device"):
            device = self.semantic_model.device
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.semantic_model.generate(
                **inputs,
                max_new_tokens=self.semantic_max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.semantic_tokenizer.pad_token_id
            )
        
        responses = []
        for input_ids, output_ids in zip(inputs["input_ids"], outputs):
            gen_ids = output_ids[len(input_ids):]
            text = self.semantic_tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(text.strip())
        
        return responses

    def _build_semantic_prompt(self, record: Dict) -> str:
        """Build semantic evaluation prompt for a single record"""
        query = record.get('query', '')
        tools = record.get('tools', [])
        answers = record.get('answers', [])
        
        if isinstance(tools, str):
            try:
                tools_json = json.loads(tools)
            except json.JSONDecodeError:
                tools_json = tools
        else:
            tools_json = tools
        
        if isinstance(answers, str):
            try:
                answers_json = json.loads(answers)
            except json.JSONDecodeError:
                answers_json = answers
        else:
            answers_json = answers
        
        tools_str = json.dumps(tools_json, ensure_ascii=False, indent=2)[:2000]
        answers_str = json.dumps(answers_json, ensure_ascii=False, indent=2)[:2000]
        
        return SEMANTIC_PROMPT_TEMPLATE.format(
            query=query,
            tools=tools_str,
            answers=answers_str
        )

    def _process_semantic_batch(self, samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples for semantic executability evaluation"""
        results = []
        batched_prompts = []
        batched_records = []
        
        for record in tqdm(samples, desc="Semantic Eval", disable=self.accelerator and not self.accelerator.is_main_process):
            try:
                prompt = self._build_semantic_prompt(record)
                batched_prompts.append(prompt)
                batched_records.append(record)
                
                if len(batched_prompts) == self.semantic_batch_size:
                    responses = self._semantic_generate_batch(batched_prompts)
                    for rec, resp in zip(batched_records, responses):
                        parsed = self._parse_semantic_response(resp)
                        if not parsed:
                            parsed = {
                                'pass': False,
                                'score': 0,
                                'issues': ["LLM output invalid JSON"],
                                'reason': resp.strip()[:500]
                            }
                        results.append({
                            'id': rec.get('id'),
                            'query': rec.get('query'),
                            'pass': parsed['pass'],
                            'score': parsed['score'],
                            'issues': parsed['issues'],
                            'reason': parsed['reason']
                        })
                    batched_prompts = []
                    batched_records = []
            except Exception as e:
                results.append({
                    'id': record.get('id'),
                    'query': record.get('query'),
                    'pass': False,
                    'score': 0,
                    'issues': [f"Exception: {str(e)}"],
                    'reason': "Exception during prompt construction or generation."
                })
                batched_prompts = []
                batched_records = []
        
        # Process remaining batch
        if batched_prompts:
            responses = self._semantic_generate_batch(batched_prompts)
            for rec, resp in zip(batched_records, responses):
                parsed = self._parse_semantic_response(resp)
                if not parsed:
                    parsed = {
                        'pass': False,
                        'score': 0,
                        'issues': ["LLM output invalid JSON"],
                        'reason': resp.strip()[:500]
                    }
                results.append({
                    'id': rec.get('id'),
                    'query': rec.get('query'),
                    'pass': parsed['pass'],
                    'score': parsed['score'],
                    'issues': parsed['issues'],
                    'reason': parsed['reason']
                })
        
        return results

    def _parse_semantic_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON output from semantic model"""
        text = text.strip()
        if text.startswith("```"):
            lines = []
            for line in text.splitlines():
                if line.strip().startswith("```"):
                    continue
                lines.append(line)
            text = "\n" + "\n".join(lines).strip()
        try:
            # Extract first JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return None
            json_str = text[start:end+1]
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None
            # Validate keys
            if 'pass' not in data or 'reason' not in data:
                return None
            if 'score' not in data:
                data['score'] = 0
            if 'issues' not in data:
                data['issues'] = []
            # Normalize types
            data['pass'] = bool(data['pass'])
            try:
                data['score'] = int(data['score'])
            except (ValueError, TypeError):
                data['score'] = 0
            if not isinstance(data['issues'], list):
                data['issues'] = [str(data['issues'])]
            data['issues'] = [str(i) for i in data['issues']]
            data['reason'] = str(data['reason'])
            return data
        except Exception:
            return None
        
    def load_dataset(self):
        """Load JSONL dataset"""
        print(f"📂 Loading dataset from: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    self.data.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: JSON decode error - {e}")
        
        self.stats['total_samples'] = len(self.data)
        print(f"✅ Loaded {self.stats['total_samples']} samples\n")
    
    def evaluate_format_correctness(self):
        """Metric 1: Format Correctness"""
        
        format_errors = []
        
        for idx, record in enumerate(tqdm(self.data, desc="Checking format")):
            errors = []
            
            # Check required fields
            required_fields = ['id', 'query', 'answers', 'tools']
            for field in required_fields:
                if field not in record:
                    errors.append(f"Missing field: {field}")
            
            if 'query' in record and not isinstance(record['query'], str):
                errors.append("Query is not a string")
            
            # Check answers field
            if 'answers' in record:
                try:
                    if isinstance(record['answers'], str):
                        answers = json.loads(record['answers'])
                    else:
                        answers = record['answers']
                    
                    if not isinstance(answers, list):
                        errors.append("Answers is not a list")
                    else:
                        for i, ans in enumerate(answers):
                            if not isinstance(ans, dict):
                                errors.append(f"Answer {i} is not a dict")
                            elif 'name' not in ans:
                                errors.append(f"Answer {i} missing 'name'")
                            elif 'arguments' not in ans:
                                errors.append(f"Answer {i} missing 'arguments'")
                except json.JSONDecodeError:
                    errors.append("Answers field is not valid JSON")
                except Exception as e:
                    errors.append(f"Answers parsing error: {str(e)}")
            
                # Check tools field
            if 'tools' in record:
                try:
                    if isinstance(record['tools'], str):
                        tools = json.loads(record['tools'])
                    else:
                        tools = record['tools']
                    
                    if not isinstance(tools, list):
                        errors.append("Tools is not a list")
                    else:
                        for i, tool in enumerate(tools):
                            if not isinstance(tool, dict):
                                errors.append(f"Tool {i} is not a dict")
                            elif 'name' not in tool:
                                errors.append(f"Tool {i} missing 'name'")
                            elif 'description' not in tool:
                                errors.append(f"Tool {i} missing 'description'")
                            elif 'parameters' not in tool:
                                errors.append(f"Tool {i} missing 'parameters'")
                            else:
                                # Check parameter annotation consistency
                                params = tool.get('parameters', {})
                                for param_name, param_info in params.items():
                                    if isinstance(param_info, dict):
                                        has_default_field = 'default' in param_info
                                        description = param_info.get('description', '')
                                        param_type = param_info.get('type', '')
                                        is_optional = 'optional' in param_type.lower()
                                        base_type = param_type.split(',')[0].strip()
                                        
                                        # Check 1: default mentioned in description but not in 'default' field
                                        default_in_desc = 'default' in description.lower() or 'defaults to' in description.lower()
                                        # if default_in_desc and not has_default_field and not is_optional:
                                        if default_in_desc and not has_default_field:
                                            errors.append(
                                                f"Tool {i} '{tool['name']}' param '{param_name}': "
                                                f"ANNOTATION ISSUE - default mentioned in description but 'default' field missing"
                                            )
                                        
                                        # Check 2: default value type doesn't match declared type
                                        if has_default_field:
                                            default_val = param_info['default']
                                            type_mismatch = None
                                            
                                            if base_type == 'int':
                                                if isinstance(default_val, float) and not default_val.is_integer():
                                                    type_mismatch = f"type is 'int' but default is float ({default_val})"
                                                elif isinstance(default_val, str) and default_val != '':
                                                    # Check if string looks like a number
                                                    try:
                                                        float(default_val)
                                                        type_mismatch = f"type is 'int' but default is str ('{default_val}')"
                                                    except:
                                                        pass
                                            elif base_type == 'str':
                                                if isinstance(default_val, (int, float)) and default_val != '':
                                                    type_mismatch = f"type is 'str' but default is {type(default_val).__name__} ({default_val})"
                                            elif base_type == 'float':
                                                if isinstance(default_val, str) and default_val != '':
                                                    try:
                                                        float(default_val)
                                                        type_mismatch = f"type is 'float' but default is str ('{default_val}')"
                                                    except:
                                                        pass
                                            
                                            if type_mismatch:
                                                errors.append(
                                                    f"Tool {i} '{tool['name']}' param '{param_name}': "
                                                    f"TYPE MISMATCH - {type_mismatch}"
                                                )
                except json.JSONDecodeError:
                    errors.append("Tools field is not valid JSON")
                except Exception as e:
                    errors.append(f"Tools parsing error: {str(e)}")
            
            if errors:
                format_errors.append({
                    'id': record.get('id', idx),
                    'errors': errors
                })
        
        self.stats['format_errors'] = format_errors
        
        error_rate = len(format_errors) / self.stats['total_samples'] * 100
        
        print(f"\n📊 Format Correctness:")
        print(f"   Pass rate: {100 - error_rate:.2f}% ({self.stats['total_samples'] - len(format_errors)}/{self.stats['total_samples']})")
        print(f"   Errors: {len(format_errors)} samples")
        
        # Write all errors to log file
        log_path = os.path.join(self.output_dir, "format_errors.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Format Correctness Errors\n")
            f.write(f"# Total samples: {self.stats['total_samples']}\n")
            f.write(f"# Error samples: {len(format_errors)}\n")
            f.write(f"# Error rate: {error_rate:.2f}%\n")
            f.write("=" * 80 + "\n\n")
            
            for err in format_errors:
                f.write(f"ID {err['id']} ({len(err['errors'])} errors):\n")
                for e in err['errors']:
                    f.write(f"  - {e}\n")
                f.write("\n")
        
        print(f"   📝 Details: {log_path}")
        
        return error_rate
    
    def evaluate_semantic_executability(self):
        """
        Metric: Semantic Executability (Query-Answer Alignment)
        Uses a local LLM to judge whether the provided tool calls satisfy the query intent.
        """
        if not self.accelerator or self.accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("🧠 Metric: Semantic Executability (Query-Answer Alignment)")
            print("=" * 80)
        
        if not self.semantic_model_path:
            if not self.accelerator or self.accelerator.is_main_process:
                print("⚠️  semantic-model not provided. Skipping semantic executability.")
            return {'status': 'SKIPPED', 'semantic_executability_error_rate': None}
        
        if not self._load_semantic_model():
            return {'status': 'FAILED_TO_LOAD_MODEL', 'semantic_executability_error_rate': None}
        
        total_available = len(self.data)
        if self.semantic_max_samples is None or self.semantic_max_samples <= 0:
            total_samples = total_available
        else:
            total_samples = min(self.semantic_max_samples, total_available)
        if total_samples == 0:
            if not self.accelerator or self.accelerator.is_main_process:
                print("⚠️  No samples available.")
            return {'status': 'NO_DATA', 'semantic_executability_error_rate': None}
        
        if not self.accelerator or self.accelerator.is_main_process:
            if total_samples == total_available:
                print(f"📊 Evaluating semantic executability on ALL {total_samples} samples (batch={self.semantic_batch_size})")
            else:
                print(f"📊 Evaluating semantic executability on {total_samples}/{total_available} samples (batch={self.semantic_batch_size})")
        
        # Prepare samples to evaluate
        samples_to_eval = self.data[:total_samples]
        
        # Distribute samples across processes if using Accelerate
        if self.accelerator:
            with self.accelerator.split_between_processes(samples_to_eval) as process_samples:
                semantic_results = self._process_semantic_batch(process_samples)
            # Gather results from all processes
            semantic_results = self.accelerator.gather_for_metrics(semantic_results)
            # Flatten if nested
            if semantic_results and isinstance(semantic_results[0], list):
                semantic_results = [item for sublist in semantic_results for item in sublist]
        else:
            semantic_results = self._process_semantic_batch(samples_to_eval)
        
        # Only main process handles final statistics and logging
        if self.accelerator and not self.accelerator.is_main_process:
            return {'status': 'SUCCESS', 'semantic_executability_error_rate': None}
        
        failures = sum(1 for r in semantic_results if r.get('issues') and any('invalid' in str(i).lower() or 'exception' in str(i).lower() for i in r['issues']))
        
        pass_count = sum(1 for r in semantic_results if r['pass'])
        fail_count = len(semantic_results) - pass_count
        error_rate = fail_count / len(semantic_results) * 100 if semantic_results else 0
        
        print(f"\n📊 Semantic Executability:")
        print(f"   Pass: {pass_count}/{len(semantic_results)} ({pass_count/len(semantic_results)*100:.2f}%)")
        print(f"   Fail: {fail_count}/{len(semantic_results)} ({error_rate:.2f}%)")
        if failures:
            print(f"   ⚠️  Invalid/failed responses: {failures}")
        
        log_path = os.path.join(self.output_dir, "semantic_executability.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("# Semantic Executability Evaluation\n")
            f.write(f"# Samples evaluated: {len(semantic_results)}\n")
            f.write(f"# Pass rate: {pass_count/len(semantic_results)*100:.2f}%\n")
            f.write(f"# Fail rate: {error_rate:.2f}%\n")
            f.write(f"# Invalid responses: {failures}\n")
            f.write("=" * 80 + "\n\n")
            for item in semantic_results:
                f.write(f"ID {item['id']} | Pass: {item['pass']} | Score: {item['score']}\n")
                f.write(f"Issues: {item['issues']}\n")
                f.write(f"Reason: {item['reason']}\n")
                f.write("-" * 60 + "\n")
        
        print(f"   📝 Details: {log_path}")
        
        return {
            'semantic_executability_error_rate': error_rate / 100,
            'samples_evaluated': len(semantic_results),
            'pass_rate': pass_count / len(semantic_results),
            'status': 'OK'
        }
    
    def evaluate_executability(self):
        """Metric 2: Executability"""
        
        executability_errors = []
        
        for idx, record in enumerate(tqdm(self.data, desc="Checking executability")):
            errors = []
            
            try:
                # Parse answers and tools
                if isinstance(record.get('answers'), str):
                    answers = json.loads(record['answers'])
                else:
                    answers = record.get('answers', [])
                
                if isinstance(record.get('tools'), str):
                    tools = json.loads(record['tools'])
                else:
                    tools = record.get('tools', [])
                
                # Build tool lookup - handle duplicate function names from different APIs
                tool_dict = defaultdict(list)
                for tool_idx, tool in enumerate(tools):
                    if isinstance(tool, dict):
                        tool_dict[tool['name']].append((tool_idx, tool))
                
                # Check each answer
                for ans_idx, answer in enumerate(answers):
                    if not isinstance(answer, dict):
                        continue
                    
                    func_name = answer.get('name')
                    arguments = answer.get('arguments', {})
                    
                    # Check if function exists in tools
                    if func_name not in tool_dict:
                        errors.append(f"[Answer #{ans_idx}] Function '{func_name}' not in available tools")
                        continue
                    
                    # Check against ALL versions of this function - only need to match ONE
                    tool_match_results = []  # Store (tool_idx, error_list) for each tool version
                    
                    for tool_idx, tool in tool_dict[func_name]:
                        tool_params = tool.get('parameters', {})
                        current_errors = []
                        
                        # Check parameters
                        if isinstance(tool_params, dict):
                            param_defs = tool_params
                            
                            # 1. Check if all used parameters are defined
                            for param_name, param_value in arguments.items():
                                if param_name not in param_defs:
                                    current_errors.append(
                                        f"Param '{param_name}' NOT DEFINED"
                                    )
                                    continue
                                
                                # 2. Basic type checking
                                param_info = param_defs.get(param_name)
                                if isinstance(param_info, dict):
                                    param_type = param_info.get('type', '')
                                    
                                    # Extract base type (remove optional, etc.)
                                    base_type = param_type.split(',')[0].strip()
                                    
                                    # Type validation
                                    type_error = None
                                    if base_type == 'str' and not isinstance(param_value, str):
                                        type_error = f"expects str, got {type(param_value).__name__} (value: {param_value})"
                                    elif base_type == 'int' and not isinstance(param_value, int):
                                        type_error = f"expects int, got {type(param_value).__name__} (value: {param_value})"
                                    elif base_type == 'float' and not isinstance(param_value, (int, float)):
                                        type_error = f"expects float, got {type(param_value).__name__} (value: {param_value})"
                                    elif base_type == 'bool' and not isinstance(param_value, bool):
                                        type_error = f"expects bool, got {type(param_value).__name__} (value: {param_value})"
                                    elif (base_type.startswith('List') or base_type == 'list') and not isinstance(param_value, list):
                                        type_error = f"expects list, got {type(param_value).__name__} (value: {param_value})"
                                    # elif base_type.startswith('Tuple') and not isinstance(param_value, (list, tuple)):
                                    #     # Tuple in JSON is represented as list
                                    #     type_error = f"expects tuple/list, got {type(param_value).__name__} (value: {param_value})"
                                    # elif base_type == 'Dict' and not isinstance(param_value, dict):
                                    #     type_error = f"expects dict, got {type(param_value).__name__} (value: {param_value})"
                                    # elif base_type == 'set' and not isinstance(param_value, (list, set)):
                                    #     # set in JSON is represented as list
                                    #     type_error = f"expects set/list, got {type(param_value).__name__} (value: {param_value})"
                                    # elif base_type.startswith('Callable') and not isinstance(param_value, str):
                                    #     # Callable in JSON is represented as string (e.g., "lambda x: x**2")
                                    #     type_error = f"expects callable/str, got {type(param_value).__name__} (value: {param_value})"
                                    
                                    if type_error:
                                        current_errors.append(f"Param '{param_name}' {type_error}")
                            
                            # 3. Check for missing required parameters
                            # Required: no 'default' field AND type doesn't contain 'optional'
                            # NOTE: Annotation issues (default in desc but not in field) are checked in Format Correctness
                            for param_name, param_info in param_defs.items():
                                if isinstance(param_info, dict):
                                    has_default = 'default' in param_info
                                    param_type = param_info.get('type', '')
                                    is_optional = 'optional' in param_type.lower()
                                    description = param_info.get('description', '')
                                    default_in_desc = 'default' in description.lower() or 'defaults to' in description.lower()
                                    
                                    # Skip if it's an annotation issue (will be caught in Format check)
                                    if default_in_desc and not has_default:
                                        continue
                                    
                                    # This is a truly required parameter
                                    if not has_default and not is_optional:
                                        if param_name not in arguments:
                                            current_errors.append(f"MISSING required param '{param_name}'")
                        
                        tool_match_results.append((tool_idx, current_errors))
                    
                    # Only report error if NONE of the tool versions matched successfully
                    successful_matches = [r for r in tool_match_results if len(r[1]) == 0]
                    
                    if not successful_matches:
                        # No tool matched - find the best match (fewest errors)
                        best_match = min(tool_match_results, key=lambda x: len(x[1]))
                        best_tool_idx, best_errors = best_match
                        
                        if len(tool_match_results) == 1:
                            # Only one tool version
                            for err in best_errors:
                                errors.append(f"[Answer #{ans_idx} -> Tool #{best_tool_idx} '{func_name}'] {err}")
                        else:
                            # Multiple tool versions, none matched
                            for err in best_errors:
                                errors.append(
                                    f"[Answer #{ans_idx} -> '{func_name}' (best: Tool #{best_tool_idx}, "
                                    f"{len(tool_match_results)} versions)] {err}"
                                )
            
            except Exception as e:
                errors.append(f"Parsing error: {str(e)}")
            
            if errors:
                executability_errors.append({
                    'id': record.get('id', idx),
                    'errors': errors,
                    'error_count': len(errors)
                })
        
        self.stats['executability_errors'] = executability_errors
        
        error_rate = len(executability_errors) / self.stats['total_samples'] * 100
        
        # 简洁终端输出
        print(f"\n📊 Executability:")
        print(f"   Executable rate: {100 - error_rate:.2f}% ({self.stats['total_samples'] - len(executability_errors)}/{self.stats['total_samples']})")
        print(f"   Errors: {len(executability_errors)} samples")
        
        # Write all errors to log file
        log_path = os.path.join(self.output_dir, "executability_errors.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Executability Errors\n")
            f.write(f"# Total samples: {self.stats['total_samples']}\n")
            f.write(f"# Error samples: {len(executability_errors)}\n")
            f.write(f"# Error rate: {error_rate:.2f}%\n")
            f.write("=" * 80 + "\n\n")
            
            for err in executability_errors:
                f.write(f"ID {err['id']} ({err['error_count']} errors):\n")
                for e in err['errors']:
                    f.write(f"  - {e}\n")
                f.write("\n")
        
        print(f"   📝 Details: {log_path}")
        
        return error_rate
    
    def evaluate_diversity(self):
        """Metric 3: Diversity"""
        
        query_lengths = []
        num_tools_list = []
        num_calls_list = []
        api_names = Counter()
        param_types = Counter()
        
        for record in tqdm(self.data, desc="Analyzing diversity"):
            # Query length
            query = record.get('query', '')
            query_lengths.append(len(query.split()))
            
            try:
                # Parse tools and answers
                if isinstance(record.get('tools'), str):
                    tools = json.loads(record['tools'])
                else:
                    tools = record.get('tools', [])
                
                if isinstance(record.get('answers'), str):
                    answers = json.loads(record['answers'])
                else:
                    answers = record.get('answers', [])
                
                # Number of tools
                num_tools_list.append(len(tools))
                
                # Number of function calls
                num_calls_list.append(len(answers))
                
                # API names
                for answer in answers:
                    if isinstance(answer, dict):
                        api_names[answer.get('name', 'unknown')] += 1
                
                # Parameter types
                for tool in tools:
                    if isinstance(tool, dict):
                        params = tool.get('parameters', {})
                        if isinstance(params, dict):
                            for param_name, param_info in params.items():
                                if isinstance(param_info, dict):
                                    param_type = param_info.get('type', 'unknown')
                                    param_types[param_type] += 1
            
            except Exception as e:
                pass
        
        self.stats['query_lengths'] = query_lengths
        self.stats['num_tools'] = num_tools_list
        self.stats['num_calls'] = num_calls_list
        
        # Calculate statistics
        total_calls = sum(api_names.values())
        probs = [count / total_calls for count in api_names.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(api_names))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        single_call_ratio = sum(1 for x in num_calls_list if x == 1) / len(num_calls_list) * 100
        parallel_call_ratio = sum(1 for x in num_calls_list if x >= 2) / len(num_calls_list) * 100
        
        # 简洁终端输出
        print(f"\n📊 Key Metrics:")
        print(f"   Query length: {np.mean(query_lengths):.1f} avg, {min(query_lengths)}-{max(query_lengths)} range")
        print(f"   Tools per sample: {np.mean(num_tools_list):.2f} avg")
        print(f"   Calls per sample: {np.mean(num_calls_list):.2f} avg (single {single_call_ratio:.1f}%, parallel {parallel_call_ratio:.1f}%)")
        print(f"   Unique APIs: {len(api_names)}")
        print(f"   API Entropy: {entropy:.2f} bits (normalized: {normalized_entropy:.2f})")
        
        # 写入详细日志
        diversity_log_path = os.path.join(self.output_dir, "diversity_stats.log")
        with open(diversity_log_path, 'w', encoding='utf-8') as f:
            f.write("# Diversity Statistics\n")
            f.write("=" * 80 + "\n\n")
            
            # 指标说明
            f.write("## 指标说明\n\n")
            f.write("### API Entropy (信息熵)\n")
            f.write("公式: H = -Σ p_i * log2(p_i)\n")
            f.write("- p_i = 第i个API的调用次数 / 总调用次数\n")
            f.write("- 熵越高说明API使用越均匀分散\n")
            f.write("- 最大熵 = log2(API数量)，此时所有API使用频率相同\n\n")
            f.write("### Normalized Entropy (归一化熵)\n")
            f.write("公式: Normalized Entropy = Entropy / Max Entropy\n")
            f.write("- 范围: 0-1\n")
            f.write("- 0 = 所有调用集中在一个API\n")
            f.write("- 1 = 所有API使用频率完全相同\n\n")
            f.write("### Single/Parallel Call Ratio\n")
            f.write("- Single: 一个sample中只有1次API调用\n")
            f.write("- Parallel: 一个sample中有≥2次API调用\n\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("## Query Statistics\n")
            f.write(f"Average length: {np.mean(query_lengths):.1f} words\n")
            f.write(f"Median length: {np.median(query_lengths):.1f} words\n")
            f.write(f"Std deviation: {np.std(query_lengths):.1f} words\n")
            f.write(f"Min/Max length: {min(query_lengths)}/{max(query_lengths)} words\n\n")
            
            f.write("## Tool Statistics\n")
            f.write(f"Average tools per sample: {np.mean(num_tools_list):.2f}\n")
            f.write(f"Median tools per sample: {np.median(num_tools_list):.1f}\n")
            f.write(f"Min/Max tools: {min(num_tools_list)}/{max(num_tools_list)}\n\n")
            
            f.write("## Function Call Statistics\n")
            f.write(f"Average calls per sample: {np.mean(num_calls_list):.2f}\n")
            f.write(f"Median calls per sample: {np.median(num_calls_list):.1f}\n")
            f.write(f"Single calls: {sum(1 for x in num_calls_list if x == 1)} ({single_call_ratio:.1f}%)\n")
            f.write(f"Parallel calls (≥2): {sum(1 for x in num_calls_list if x >= 2)} ({parallel_call_ratio:.1f}%)\n\n")
            
            f.write("## API Diversity\n")
            f.write(f"Unique APIs: {len(api_names)}\n")
            f.write(f"Total API calls: {total_calls}\n")
            f.write(f"API Entropy: {entropy:.2f} bits\n")
            f.write(f"Max Entropy: {max_entropy:.2f} bits (= log2({len(api_names)}))\n")
            f.write(f"Normalized Entropy: {normalized_entropy:.4f}\n\n")
            
            f.write("## Top APIs (by usage)\n")
            for api, count in api_names.most_common():
                f.write(f"{api}\t{count}\t{count/total_calls*100:.2f}%\n")
            
            f.write("\n## Parameter Type Distribution (declared in tools)\n")
            total_params = sum(param_types.values())
            for ptype, count in param_types.most_common():
                f.write(f"{ptype}\t{count}\t{count/total_params*100:.1f}%\n")
        
        print(f"   📝 Details: {diversity_log_path}")
        
        # ====== API Call Diversity (参数多样性) ======
        api_call_stats = self._evaluate_api_call_diversity()
        
        # ====== Query Style Classification (论文 Section 3.3) ======
        query_style_stats = self._evaluate_query_style()
        
        # ====== Linguistic Diversity (论文 anti-template) ======
        self._evaluate_linguistic_diversity()
        
        # ====== Long-tail Coverage ======
        long_tail_stats = self._evaluate_long_tail_coverage(api_names)
        
        return {
            'query_length_stats': (np.mean(query_lengths), np.std(query_lengths)),
            'parallel_call_ratio': sum(1 for x in num_calls_list if x >= 2) / len(num_calls_list),
            'api_entropy': entropy,
            'unique_apis': len(api_names),
            'query_style': query_style_stats,
            'long_tail': long_tail_stats,
            'api_call': api_call_stats
        }
    
    def _evaluate_api_call_diversity(self):
        """评估参数多样性（按参数名聚合为主）"""
        # ====== 按参数名聚合（主要指标）======
        param_values_by_name = defaultdict(list)  # param_name -> [values]
        
        # ====== 辅助统计 ======
        all_param_values = []  # 全局参数值（辅助指标）
        param_value_types = Counter()  # 参数值类型统计
        param_type_counter = Counter()  # 声明的参数类型统计
        total_api_calls = 0
        total_params_passed = 0
        
        for record in self.data:
            try:
                if isinstance(record.get('answers'), str):
                    answers = json.loads(record['answers'])
                else:
                    answers = record.get('answers', [])
                
                for answer in answers:
                    if isinstance(answer, dict):
                        arguments = answer.get('arguments', {})
                        total_api_calls += 1
                        
                        if isinstance(arguments, dict):
                            for param_name, param_value in arguments.items():
                                total_params_passed += 1
                                
                                # 记录参数值类型
                                param_value_types[type(param_value).__name__] += 1
                                
                                # 转为字符串以便统计
                                if isinstance(param_value, (list, dict)):
                                    value_str = json.dumps(param_value, sort_keys=True)
                                else:
                                    value_str = str(param_value)
                                
                                # 按参数名聚合
                                param_values_by_name[param_name].append(value_str)
                                
                                # 全局参数值
                                all_param_values.append(value_str)
                
                # 统计 Tool 中声明的参数类型
                if isinstance(record.get('tools'), str):
                    tools = json.loads(record['tools'])
                else:
                    tools = record.get('tools', [])
                
                for tool in tools:
                    if isinstance(tool, dict) and 'parameters' in tool:
                        for param_info in tool['parameters'].values():
                            if isinstance(param_info, dict):
                                param_type_counter[param_info.get('type', 'unknown')] += 1
                                
            except Exception:
                pass
        
        # ====== 计算按参数名的多样性指标 ======
        param_stats = []
        for param_name, values in param_values_by_name.items():
            unique_count = len(set(values))
            total_count = len(values)
            diversity = unique_count / total_count if total_count > 0 else 0
            
            # 统计最常见的值
            value_counter = Counter(values)
            
            param_stats.append({
                'name': param_name,
                'unique': unique_count,
                'total': total_count,
                'diversity': diversity,
                'top_values': value_counter.most_common(5)
            })
        
        # 按多样性排序
        param_stats_sorted_low = sorted(param_stats, key=lambda x: x['diversity'])
        param_stats_sorted_high = sorted(param_stats, key=lambda x: x['diversity'], reverse=True)
        
        # 计算平均多样性
        avg_diversity = np.mean([s['diversity'] for s in param_stats]) if param_stats else 0
        
        # ====== 全局多样性（辅助指标）======
        global_unique = len(set(all_param_values))
        global_diversity = global_unique / total_params_passed if total_params_passed > 0 else 0
        
        # ====== 简洁终端输出 ======
        print(f"\n📦 Parameter Diversity (by Parameter Name):")
        print(f"   Unique param names: {len(param_values_by_name)}")
        print(f"   Total params passed: {total_params_passed:,}")
        print(f"   Avg diversity per param: {avg_diversity:.4f}")
        print(f"   Global value diversity: {global_diversity:.4f} ({global_unique:,} unique / {total_params_passed:,} total)")
        
        # 显示多样性最低的参数（潜在模板化问题）
        low_diversity_params = [s for s in param_stats_sorted_low if s['diversity'] < 0.1 and s['total'] >= 100]
        if low_diversity_params:
            print(f"   ⚠️  Low diversity params (potential templating): {len(low_diversity_params)}")
            for s in low_diversity_params[:3]:
                top_val = s['top_values'][0][0][:20] + "..." if len(s['top_values'][0][0]) > 20 else s['top_values'][0][0]
                print(f"      {s['name']}: {s['unique']}/{s['total']} = {s['diversity']:.3f} (top: {top_val})")
        
        # 参数值类型分布（简洁版）
        total_value_types = sum(param_value_types.values())
        top_types = param_value_types.most_common(5)
        type_str = ", ".join([f"{t}: {c/total_value_types*100:.1f}%" for t, c in top_types])
        print(f"   Param value types: {type_str}")
        
        # ====== 写入日志 ======
        log_path = os.path.join(self.output_dir, "parameter_diversity.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("# Parameter Diversity Analysis (按参数名聚合)\n")
            f.write("=" * 80 + "\n\n")
            
            # 指标说明
            f.write("## 指标说明\n\n")
            f.write("### 1. Per-Parameter Diversity (每参数多样性) - 主要指标\n")
            f.write("公式: Unique values / Total occurrences (对每个参数名单独计算)\n")
            f.write("- 不区分API，只看参数名（如所有的 city 参数放一起统计）\n")
            f.write("- 值越高说明该参数的值越多样化\n")
            f.write("- 值接近0说明该参数可能被模板化（总是用固定值）\n")
            f.write("- 例如: city参数只用了3个城市 → diversity = 3/5000 = 0.0006 ❌\n\n")
            f.write("### 2. Global Value Diversity (全局值多样性) - 辅助指标\n")
            f.write("公式: Unique values / Total params passed (所有参数值放一起统计)\n")
            f.write("- 作为总体指标，反映整体参数值的多样性\n")
            f.write("- 缺点：混淆了不同语义的参数（城市名、日期、数字等）\n\n")
            f.write("### 3. Avg Diversity Per Param (平均每参数多样性)\n")
            f.write("公式: Σ(每个参数的diversity) / 参数名数量\n")
            f.write("- 综合反映所有参数的多样性水平\n\n")
            f.write("### 如何解读？\n")
            f.write("- diversity < 0.1 且 total >= 100: ⚠️ 可能存在模板化问题\n")
            f.write("- diversity > 0.5: ✅ 多样性较好\n")
            f.write("- diversity > 0.9: ✅✅ 多样性很高（几乎每次都用不同值）\n\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体统计
            f.write("## Summary Statistics\n\n")
            f.write(f"Unique param names: {len(param_values_by_name)}\n")
            f.write(f"Total API calls: {total_api_calls:,}\n")
            f.write(f"Total params passed: {total_params_passed:,}\n")
            f.write(f"Avg diversity per param: {avg_diversity:.6f}\n")
            f.write(f"Global value diversity: {global_diversity:.6f} ({global_unique:,} unique / {total_params_passed:,} total)\n\n")
            
            # 参数值类型分布
            f.write("## Param Value Types (in answers)\n\n")
            for vtype, count in param_value_types.most_common():
                f.write(f"{vtype}\t{count}\t{count/total_value_types*100:.2f}%\n")
            
            f.write("\n## Declared Param Types (in tools)\n\n")
            total_declared = sum(param_type_counter.values())
            for ptype, count in param_type_counter.most_common():
                f.write(f"{ptype}\t{count}\t{count/total_declared*100:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # ====== 多样性最低的参数（潜在问题）======
            f.write("## ⚠️ LOW Diversity Parameters (potential templating)\n")
            f.write("筛选条件: diversity < 0.1 且 total >= 100\n\n")
            
            low_div_params = [s for s in param_stats_sorted_low if s['diversity'] < 0.1 and s['total'] >= 100]
            if low_div_params:
                for s in low_div_params:
                    f.write(f"### {s['name']}\n")
                    f.write(f"  Unique: {s['unique']}, Total: {s['total']}, Diversity: {s['diversity']:.6f}\n")
                    f.write(f"  Top values:\n")
                    for val, cnt in s['top_values']:
                        display_val = val[:80] + "..." if len(val) > 80 else val
                        f.write(f"    - {display_val} ({cnt} times, {cnt/s['total']*100:.1f}%)\n")
                    f.write("\n")
            else:
                f.write("  (无)\n\n")
            
            f.write("=" * 80 + "\n\n")
            
            # ====== 多样性最高的参数 ======
            f.write("## ✅ HIGH Diversity Parameters\n")
            f.write("筛选条件: diversity > 0.5 且 total >= 100\n\n")
            
            high_div_params = [s for s in param_stats_sorted_high if s['diversity'] > 0.5 and s['total'] >= 100][:20]
            if high_div_params:
                for s in high_div_params:
                    f.write(f"  {s['name']:40s}: {s['unique']:5d}/{s['total']:5d} = {s['diversity']:.4f}\n")
            else:
                f.write("  (无)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # ====== 所有参数的完整统计 ======
            f.write("## All Parameters (sorted by diversity ascending)\n\n")
            f.write(f"{'Parameter Name':<50s} {'Unique':>8s} {'Total':>8s} {'Diversity':>12s}\n")
            f.write("-" * 80 + "\n")
            for s in param_stats_sorted_low:
                f.write(f"{s['name']:<50s} {s['unique']:>8d} {s['total']:>8d} {s['diversity']:>12.6f}\n")
        
        print(f"   📝 Details: {log_path}")
        
        return {
            'unique_param_names': len(param_values_by_name),
            'avg_diversity': avg_diversity,
            'global_diversity': global_diversity,
            'low_diversity_count': len(low_diversity_params)
        }
    
    def _evaluate_query_style(self):
        """评估Query Style分类（论文Section 3.3核心！）
        
        4种风格：
        - Simple: 1个API可用，1次调用
        - Multiple: 多个API可选，1次调用  
        - Parallel: 1个API，多次调用（同一函数）
        - Parallel Multiple: 多个API，多次调用（不同函数）
        """
        style_counts = Counter()
        style_all_records = defaultdict(list)
        
        for record in self.data:
            try:
                if isinstance(record.get('tools'), str):
                    tools = json.loads(record['tools'])
                else:
                    tools = record.get('tools', [])
                
                if isinstance(record.get('answers'), str):
                    answers = json.loads(record['answers'])
                else:
                    answers = record.get('answers', [])
                
                num_tools = len(tools)
                num_calls = len(answers)
                
                called_funcs = set()
                for ans in answers:
                    if isinstance(ans, dict):
                        called_funcs.add(ans.get('name', ''))
                num_unique_funcs = len(called_funcs)
                
                if num_calls == 0:
                    style = "Invalid"
                elif num_calls == 1:
                    if num_tools == 1:
                        style = "Simple"
                    else:
                        style = "Multiple"
                else:
                    if num_tools == 1:
                        style = "Parallel"
                    else:
                        style = "Parallel Multiple"
                
                style_counts[style] += 1
                style_all_records[style].append({
                    'id': record.get('id'),
                    'query': record.get('query', ''),
                    'num_tools': num_tools,
                    'num_calls': num_calls,
                    'unique_funcs': num_unique_funcs
                })
                
            except Exception:
                style_counts["Parse Error"] += 1
        
        total = sum(style_counts.values())
        parallel_count = style_counts.get("Parallel", 0) + style_counts.get("Parallel Multiple", 0)
        parallel_ratio = parallel_count / total * 100
        
        # 简洁终端输出
        print(f"\n📋 Query Style:")
        print(f"   Simple: {style_counts.get('Simple', 0)} ({style_counts.get('Simple', 0)/total*100:.1f}%)")
        print(f"   Multiple: {style_counts.get('Multiple', 0)} ({style_counts.get('Multiple', 0)/total*100:.1f}%)")
        print(f"   Parallel: {style_counts.get('Parallel', 0)} ({style_counts.get('Parallel', 0)/total*100:.1f}%)")
        print(f"   Parallel Multiple: {style_counts.get('Parallel Multiple', 0)} ({style_counts.get('Parallel Multiple', 0)/total*100:.1f}%)")
        print(f"   Parallel总比例: {parallel_ratio:.1f}% {'✅' if parallel_ratio >= 20 else '❌'}")
        
        # 写入日志
        log_path = os.path.join(self.output_dir, "query_styles.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("# Query Style Distribution (Complete)\n")
            f.write(f"# Total samples: {total}\n")
            f.write("=" * 80 + "\n\n")
            
            # 指标说明
            f.write("## 指标说明 (参考 APIGen 论文 Section 3.3)\n\n")
            f.write("### Query Style 分类定义\n")
            f.write("基于两个维度：可用工具数(num_tools) 和 实际调用数(num_calls)\n\n")
            f.write("| Style            | num_tools | num_calls | 说明                           |\n")
            f.write("|------------------|-----------|-----------|--------------------------------|\n")
            f.write("| Simple           | = 1       | = 1       | 只有1个工具可选，调用1次       |\n")
            f.write("| Multiple         | > 1       | = 1       | 多个工具可选，只调用1次        |\n")
            f.write("| Parallel         | = 1       | ≥ 2       | 只有1个工具，多次调用          |\n")
            f.write("| Parallel Multiple| > 1       | ≥ 2       | 多个工具可选，多次调用         |\n")
            f.write("| Invalid          | any       | = 0       | 没有API调用（异常数据）        |\n\n")
            f.write("### Parallel 总比例\n")
            f.write("公式: (Parallel + Parallel Multiple) / Total\n")
            f.write("- 衡量数据集支持并行调用的能力\n")
            f.write("- APIGen论文目标: ≥20%\n\n")
            f.write("=" * 80 + "\n\n")
            
            # 先写摘要
            f.write("## Summary\n\n")
            for style in ["Simple", "Multiple", "Parallel", "Parallel Multiple", "Invalid", "Parse Error"]:
                count = style_counts.get(style, 0)
                if count > 0:
                    f.write(f"  {style}: {count} ({count/total*100:.2f}%)\n")
            f.write(f"\n  Parallel总比例: {parallel_ratio:.2f}%\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 再写每种风格的完整记录
            for style in ["Simple", "Multiple", "Parallel", "Parallel Multiple", "Invalid", "Parse Error"]:
                count = style_counts.get(style, 0)
                if count > 0:
                    f.write(f"## {style}: {count} samples ({count/total*100:.2f}%)\n")
                    f.write("-" * 60 + "\n\n")
                    
                    for rec in style_all_records[style]:
                        f.write(f"ID {rec['id']}:\n")
                        f.write(f"  Query: {rec['query']}\n")
                        f.write(f"  Tools: {rec['num_tools']}, Calls: {rec['num_calls']}, Unique funcs: {rec['unique_funcs']}\n\n")
        
        print(f"   📝 Details: {log_path}")
        
        return {
            'parallel_ratio': parallel_ratio / 100,
            'parallel_count': parallel_count
        }
    
    def _evaluate_linguistic_diversity(self):
        """评估语言表达多样性（论文anti-template）"""
        import re
        import string
        
        all_queries = [record.get('query', '') for record in self.data]
        
        # 预处理函数：去除标点符号，只保留字母和数字
        def tokenize(text):
            """分词并去除标点符号"""
            # 转小写
            text = text.lower()
            # 用正则提取单词（字母数字组合）
            words = re.findall(r'\b[a-z0-9]+\b', text)
            return words
        
        # 1. 词汇量统计
        all_words = []
        word_counter = Counter()
        for query in all_queries:
            words = tokenize(query)
            all_words.extend(words)
            word_counter.update(words)
        
        vocab_size = len(set(all_words))
        total_words = len(all_words)
        
        # 2. Distinct-n率（n-gram多样性）
        def get_ngrams(words, n):
            """从词列表生成n-gram"""
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        bigram_counter = Counter()
        trigram_counter = Counter()
        for query in all_queries:
            words = tokenize(query)
            bigrams_q = get_ngrams(words, 2)
            trigrams_q = get_ngrams(words, 3)
            bigram_counter.update(bigrams_q)
            trigram_counter.update(trigrams_q)
        
        total_bigrams = sum(bigram_counter.values())
        total_trigrams = sum(trigram_counter.values())
        
        distinct_1 = vocab_size / total_words if total_words > 0 else 0
        distinct_2 = len(bigram_counter) / total_bigrams if total_bigrams else 0
        distinct_3 = len(trigram_counter) / total_trigrams if total_trigrams else 0
        
        # 3. Query模板检测（增强版）
        import re
        
        def normalize_query_template(query):
            """将query标准化为模板形式"""
            template = query
            
            # 1. 数字（包括小数、负数）
            template = re.sub(r'-?\d+\.?\d*', 'NUM', template)
            
            # 2. 股票代码（2-5个大写字母）
            template = re.sub(r'\b[A-Z]{2,5}\b', 'SYMBOL', template)
            
            # 3. 引号内的字符串值
            template = re.sub(r"'[^']*'", 'STR', template)
            template = re.sub(r'"[^"]*"', 'STR', template)
            
            # 4. Email地址
            template = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'EMAIL', template)
            
            # 5. URL
            template = re.sub(r'https?://\S+', 'URL', template)
            
            # 6. 日期格式
            template = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'DATE', template)
            template = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'DATE', template)
            
            # 7. 时间格式
            template = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', 'TIME', template)
            
            return template.lower()
        
        # 同时记录每个query对应的模板和ID
        query_template_mapping = []
        for record in self.data:
            query = record.get('query', '')
            template = normalize_query_template(query)
            query_template_mapping.append({
                'id': record.get('id'),
                'query': query,
                'template': template
            })
        
        templates = [item['template'] for item in query_template_mapping]
        template_diversity = len(set(templates)) / len(templates) if templates else 0
        template_counter = Counter(templates)
        repeated_templates = [(t, c) for t, c in template_counter.most_common() if c > 1]
        
        # 简洁终端输出
        print(f"\n📝 Linguistic Diversity:")
        print(f"   Vocab: {vocab_size:,} unique / {total_words:,} total (TTR: {distinct_1:.4f})")
        print(f"   Distinct-2: {distinct_2:.4f} {'✅' if distinct_2 > 0.6 else '⚠️'}")
        print(f"   Distinct-3: {distinct_3:.4f}")
        print(f"   Template diversity: {template_diversity:.4f} {'✅' if template_diversity > 0.8 else '⚠️'}")
        
        # 按模板分组
        template_to_queries = defaultdict(list)
        for item in query_template_mapping:
            template_to_queries[item['template']].append({
                'id': item['id'],
                'query': item['query']
            })
        
        # 写入完整日志
        log_path = os.path.join(self.output_dir, "query_templates.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("# Query Template Analysis (Complete)\n")
            f.write(f"# Total queries: {len(templates)}\n")
            f.write(f"# Unique templates: {len(set(templates))}\n")
            f.write(f"# Template diversity: {template_diversity:.4f}\n")
            f.write("=" * 80 + "\n\n")
            
            # 摘要：重复模板统计
            f.write("## Summary: Repeated Templates\n\n")
            f.write(f"Total repeated templates (count > 1): {len(repeated_templates)}\n\n")
            for template, count in repeated_templates[:20]:
                f.write(f"  [{count}x] {template[:100]}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 详细：每个模板及其匹配的queries
            f.write("## Detailed: Templates and Matched Queries\n\n")
            
            # 按出现次数排序
            sorted_templates = sorted(template_to_queries.items(), 
                                      key=lambda x: len(x[1]), reverse=True)
            
            for template, queries in sorted_templates:
                f.write(f"### Template (matched {len(queries)} queries):\n")
                f.write(f"{template}\n\n")
                f.write("Matched queries:\n")
                for q in queries:
                    f.write(f"  ID {q['id']}: {q['query']}\n")
                f.write("\n" + "-" * 60 + "\n\n")
        
        # 写入词汇统计日志
        vocab_log_path = os.path.join(self.output_dir, "vocabulary_analysis.log")
        with open(vocab_log_path, 'w', encoding='utf-8') as f:
            f.write("# Vocabulary & N-gram Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            # 指标说明
            f.write("## 指标说明\n\n")
            f.write("### Distinct-N 计算公式\n")
            f.write("Distinct-N = Unique N-grams / Total N-grams\n\n")
            f.write("- Distinct-1 (TTR): 唯一单词数 / 总单词数\n")
            f.write("- Distinct-2: 唯一二元组数 / 总二元组数\n")
            f.write("- Distinct-3: 唯一三元组数 / 总三元组数\n\n")
            f.write("### 解读\n")
            f.write("- 值越高表示语言表达越多样化\n")
            f.write("- Distinct-2 > 0.6 通常认为是好的多样性\n")
            f.write("- 值越低说明重复表达越多，可能存在模板化问题\n\n")
            f.write("### 示例\n")
            f.write("假设有3个query: 'what is A', 'what is B', 'how to C'\n")
            f.write("- 总单词数=9, 唯一单词数=6 → Distinct-1 = 6/9 = 0.67\n")
            f.write("- 总bigram数=6, 唯一bigram数=4 → Distinct-2 = 4/6 = 0.67\n\n")
            f.write("=" * 80 + "\n\n")
            
            # 摘要统计
            f.write("## Summary Statistics\n\n")
            f.write(f"Total words (tokens): {total_words:,}\n")
            f.write(f"Vocabulary size (unique words): {vocab_size:,}\n")
            f.write(f"Distinct-1 (TTR): {vocab_size} / {total_words} = {distinct_1:.6f}\n\n")
            
            f.write(f"Total bigrams: {total_bigrams:,}\n")
            f.write(f"Unique bigrams: {len(bigram_counter):,}\n")
            f.write(f"Distinct-2: {len(bigram_counter)} / {total_bigrams} = {distinct_2:.6f}\n\n")
            
            f.write(f"Total trigrams: {total_trigrams:,}\n")
            f.write(f"Unique trigrams: {len(trigram_counter):,}\n")
            f.write(f"Distinct-3: {len(trigram_counter)} / {total_trigrams} = {distinct_3:.6f}\n\n")
            
            f.write("=" * 80 + "\n\n")
            
            # Unigram (词频统计)
            f.write("## Unigram (Word Frequency)\n\n")
            f.write(f"Top 100 most frequent words:\n\n")
            for i, (word, count) in enumerate(word_counter.most_common(100), 1):
                f.write(f"  {i:3d}. {word:30s} {count:8d} ({count/total_words*100:.3f}%)\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
            f.write(f"Rare words (appear only once): {sum(1 for c in word_counter.values() if c == 1):,}\n")
            f.write(f"Rare word ratio: {sum(1 for c in word_counter.values() if c == 1)/vocab_size*100:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Bigram (二元组统计) - Top 100
            f.write("## Bigram (2-gram Frequency) - Top 100\n\n")
            for i, (bigram, count) in enumerate(bigram_counter.most_common(100), 1):
                bigram_str = ' '.join(bigram)
                f.write(f"  {i:3d}. {bigram_str:50s} {count:8d} ({count/total_bigrams*100:.3f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Trigram (三元组统计) - Top 100
            f.write("## Trigram (3-gram Frequency) - Top 100\n\n")
            for i, (trigram, count) in enumerate(trigram_counter.most_common(100), 1):
                trigram_str = ' '.join(trigram)
                f.write(f"  {i:3d}. {trigram_str:60s} {count:8d} ({count/total_trigrams*100:.3f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 完整词表（按频率排序）
            f.write("## Complete Unigram Vocabulary (sorted by frequency)\n\n")
            for word, count in word_counter.most_common():
                f.write(f"{word}\t{count}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 完整 Bigram 列表（按频率排序）
            f.write("## Complete Bigram List (sorted by frequency)\n\n")
            for bigram, count in bigram_counter.most_common():
                bigram_str = ' '.join(bigram)
                f.write(f"{bigram_str}\t{count}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 完整 Trigram 列表（按频率排序）
            f.write("## Complete Trigram List (sorted by frequency)\n\n")
            for trigram, count in trigram_counter.most_common():
                trigram_str = ' '.join(trigram)
                f.write(f"{trigram_str}\t{count}\n")
        
        print(f"   📝 Details: {log_path}, {vocab_log_path}")
    
    def _evaluate_long_tail_coverage(self, api_names):
        """评估长尾覆盖"""
        total_calls = sum(api_names.values())
        
        # 按使用频率分层
        rare_apis = []      # <1%
        medium_apis = []    # 1%-5%
        head_apis = []      # >5%
        
        for api, count in api_names.items():
            ratio = count / total_calls
            if ratio < 0.01:
                rare_apis.append((api, count, ratio))
            elif ratio < 0.05:
                medium_apis.append((api, count, ratio))
            else:
                head_apis.append((api, count, ratio))
        
        # Gini系数计算
        counts = sorted(api_names.values())
        n = len(counts)
        if n > 0 and total_calls > 0:
            gini = (2 * sum((i+1) * count for i, count in enumerate(counts))) / (n * total_calls) - (n + 1) / n
        else:
            gini = 0
        
        # 简洁终端输出
        print(f"\n🦎 Long-tail Coverage:")
        print(f"   Head (>5%): {len(head_apis)} APIs | Medium (1-5%): {len(medium_apis)} APIs | Rare (<1%): {len(rare_apis)} APIs")
        print(f"   Gini: {gini:.3f} {'✅' if gini < 0.5 else '⚠️'}")
        
        # 写入日志
        log_path = os.path.join(self.output_dir, "long_tail_coverage.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("# Long-tail API Coverage\n")
            f.write("=" * 80 + "\n\n")
            
            # 指标说明
            f.write("## 指标说明\n\n")
            f.write("### Gini 系数计算公式\n")
            f.write("G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n+1)/n\n\n")
            f.write("其中:\n")
            f.write("- x_i: 从小到大排序后第i个API的调用次数\n")
            f.write("- n: API总数\n")
            f.write("- Σx_i: 总调用次数\n\n")
            f.write("### 解读\n")
            f.write("- Gini = 0: 完全均匀（每个API调用次数一样）\n")
            f.write("- Gini = 1: 极度集中（所有调用都集中在一个API）\n")
            f.write("- Gini < 0.3: 分布较均匀，长尾覆盖好 ✅\n")
            f.write("- Gini 0.3-0.5: 存在一定集中，但可接受\n")
            f.write("- Gini 0.5-0.7: 比较集中，部分API占主导 ⚠️\n")
            f.write("- Gini > 0.7: 高度集中，少数API占绝大多数调用 ❌\n\n")
            f.write("### API分层定义\n")
            f.write("- Head APIs: 单个API调用占比 > 5%\n")
            f.write("- Medium APIs: 单个API调用占比 1%-5%\n")
            f.write("- Rare APIs: 单个API调用占比 < 1%\n\n")
            f.write("=" * 80 + "\n\n")
            
            # 统计数据
            f.write("## Statistics\n\n")
            f.write(f"Total unique APIs: {len(api_names)}\n")
            f.write(f"Total API calls: {total_calls}\n")
            f.write(f"Gini coefficient: {gini:.4f}\n\n")
            
            f.write(f"Head APIs (>5%): {len(head_apis)}\n")
            f.write(f"Medium APIs (1-5%): {len(medium_apis)}\n")
            f.write(f"Rare APIs (<1%): {len(rare_apis)}\n")
            f.write(f"APIs used only once: {sum(1 for c in api_names.values() if c == 1)}\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## All APIs (sorted by usage)\n\n")
            for api, count in api_names.most_common():
                f.write(f"{api}\t{count}\t{count/total_calls*100:.2f}%\n")
        
        print(f"   📝 Details: {log_path}")
        
        return {
            'gini': gini,
            'rare_api_count': len(rare_apis),
            'rare_api_ratio': len(rare_apis) / len(api_names)
        }
    
    def evaluate_data_noise(self):
        """Metric 4: Data Noise (Duplicates, Contradictions)"""
        # Check for duplicate queries
        query_counter = Counter()
        query_to_ids = defaultdict(list)
        
        for record in tqdm(self.data, desc="Checking duplicates"):
            query = record.get('query', '').strip().lower()
            query_counter[query] += 1
            query_to_ids[query].append(record.get('id'))
        
        duplicates = {q: count for q, count in query_counter.items() if count > 1}
        
        # Check for contradictions (same query, different answers)
        contradictions = []
        for query, ids in query_to_ids.items():
            if len(ids) > 1:
                answers_set = set()
                for record in self.data:
                    if record.get('id') in ids:
                        ans = record.get('answers', '')
                        if isinstance(ans, str):
                            answers_set.add(ans)
                        else:
                            answers_set.add(json.dumps(ans, sort_keys=True))
                
                if len(answers_set) > 1:
                    contradictions.append({
                        'query': query,
                        'ids': ids,
                        'num_different_answers': len(answers_set)
                    })
        
        # Check for incomplete samples
        incomplete = []
        for record in self.data:
            if not record.get('query') or not record.get('answers') or not record.get('tools'):
                incomplete.append(record.get('id'))
        
        self.stats['duplicates'] = duplicates
        self.stats['contradictions'] = contradictions
        self.stats['incomplete'] = incomplete
        
        # 简洁终端输出
        dup_rate = len(duplicates) / len(query_counter) * 100
        complete_rate = (1 - len(incomplete) / self.stats['total_samples']) * 100
        
        print(f"\n📊 Data Noise:")
        print(f"   Duplicates: {len(duplicates)} ({dup_rate:.2f}%)")
        print(f"   Contradictions: {len(contradictions)}")
        print(f"   Completeness: {complete_rate:.2f}%")
        
        # Write noise analysis to log file
        log_path = os.path.join(self.output_dir, "noise_analysis.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Data Noise Analysis\n")
            f.write(f"# Total samples: {self.stats['total_samples']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("## Duplicates\n")
            f.write(f"Total unique queries: {len(query_counter)}\n")
            f.write(f"Duplicate queries: {len(duplicates)}\n")
            f.write(f"Duplication rate: {len(duplicates)/len(query_counter)*100:.2f}%\n\n")
            
            for query, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
                f.write(f"Query (appears {count} times): '{query}'\n")
                f.write(f"  IDs: {query_to_ids[query]}\n\n")
            
            f.write("\n## Contradictions\n")
            f.write(f"Total contradictions: {len(contradictions)}\n\n")
            for cont in contradictions:
                f.write(f"Query: '{cont['query']}'\n")
                f.write(f"  IDs with different answers: {cont['ids']}\n")
                f.write(f"  Number of different answers: {cont['num_different_answers']}\n\n")
            
            f.write("\n## Incomplete Samples\n")
            f.write(f"Total incomplete: {len(incomplete)}\n")
            f.write(f"IDs: {incomplete}\n")
        
        print(f"   📝 Details: {log_path}")
        
        return {
            'duplication_rate': len(duplicates) / len(query_counter),
            'contradiction_count': len(contradictions),
            'incompleteness_rate': len(incomplete) / self.stats['total_samples']
        }
    
    def run_evaluation(self):
        """Run all basic evaluations"""
        # Run all metrics
        format_error_rate = self.evaluate_format_correctness()
        exec_error_rate = self.evaluate_executability()
        diversity_metrics = self.evaluate_diversity()
        noise_metrics = self.evaluate_data_noise()
        semantic_metrics = None
        if self.semantic_model_path:
            semantic_metrics = self.evaluate_semantic_executability()
        
        # Summary
        print("\n" + "=" * 80)
        print("📈 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\n✅ Format Correctness: {100 - format_error_rate:.2f}% pass rate")
        print(f"✅ Executability: {100 - exec_error_rate:.2f}% executable")
        print(f"✅ Parallel Call Ratio: {diversity_metrics['parallel_call_ratio']*100:.1f}%")
        print(f"✅ Unique APIs: {diversity_metrics['unique_apis']}")
        print(f"✅ API Diversity (Entropy): {diversity_metrics['api_entropy']:.2f} bits")
        print(f"⚠️  Duplication Rate: {noise_metrics['duplication_rate']*100:.2f}%")
        print(f"⚠️  Contradictions: {noise_metrics['contradiction_count']}")
        print(f"⚠️  Incompleteness Rate: {noise_metrics['incompleteness_rate']*100:.2f}%")
        if semantic_metrics and semantic_metrics.get('status') == 'OK':
            print(f"✅ Semantic Executability Pass Rate: {semantic_metrics['pass_rate']*100:.2f}%")
        elif semantic_metrics:
            print(f"⚠️  Semantic Executability: {semantic_metrics.get('status')}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='xLAM-60k Dataset Basic Quality Evaluation')
    parser.add_argument('--dataset', '-d', type=str, 
                        default="/mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl",
                        help='Path to the dataset JSONL file')
    parser.add_argument('--metric', '-m', type=str, choices=['all', 'format', 'exec', 'semantic', 'diversity', 'noise'],
                        default='all',
                        help='Which metric to evaluate: all, format, exec, semantic, diversity, noise')
    parser.add_argument('--semantic-model', type=str, default=None,
                        help='Path to local LLM for semantic executability checking')
    parser.add_argument('--semantic-max-samples', type=int, default=200,
                        help='Maximum samples for semantic executability (<=0 means all samples)')
    parser.add_argument('--semantic-batch-size', type=int, default=2,
                        help='Batch size when generating semantic judgements')
    parser.add_argument('--semantic-max-new-tokens', type=int, default=512,
                        help='Max new tokens to generate for semantic evaluations')
    parser.add_argument('--accelerate', action='store_true',
                        help='Use Accelerate for multi-GPU parallel inference')
    
    args = parser.parse_args()
    
    evaluator = XLAMBasicEvaluator(
        args.dataset,
        semantic_model=args.semantic_model,
        semantic_batch_size=args.semantic_batch_size,
        semantic_max_samples=args.semantic_max_samples,
        semantic_max_new_tokens=args.semantic_max_new_tokens,
        use_accelerate=args.accelerate
    )
    
    print("\n" + "=" * 80)
    print("🚀 xLAM-60k Dataset Basic Quality Evaluation")
    print("=" * 80)
    print()
    
    evaluator.load_dataset()
    
    if args.metric == 'all':
        evaluator.run_evaluation()
    elif args.metric == 'format':
        evaluator.evaluate_format_correctness()
    elif args.metric == 'exec':
        evaluator.evaluate_executability()
    elif args.metric == 'semantic':
        evaluator.evaluate_semantic_executability()
    elif args.metric == 'diversity':
        evaluator.evaluate_diversity()
    elif args.metric == 'noise':
        evaluator.evaluate_data_noise()
    
    print("\n" + "=" * 80)
    print("✨ Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='xLAM-60k Dataset Basic Quality Evaluation')
    parser.add_argument('--dataset', '-d', type=str, 
                        default="/mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl",
                        help='Path to the dataset JSONL file')
    parser.add_argument('--metric', '-m', type=str, choices=['all', 'format', 'exec', 'semantic', 'diversity', 'noise'],
                        default='all',
                        help='Which metric to evaluate: all, format, exec, semantic, diversity, noise')
    parser.add_argument('--semantic-model', type=str, default=None,
                        help='Path to local LLM for semantic executability checking')
    parser.add_argument('--semantic-max-samples', type=int, default=200,
                        help='Maximum samples for semantic executability (<=0 means all samples)')
    parser.add_argument('--semantic-batch-size', type=int, default=2,
                        help='Batch size when generating semantic judgements')
    parser.add_argument('--semantic-max-new-tokens', type=int, default=512,
                        help='Max new tokens to generate for semantic evaluations')
    parser.add_argument('--accelerate', action='store_true',
                        help='Use Accelerate for multi-GPU parallel inference')
    
    args = parser.parse_args()
    
    evaluator = XLAMBasicEvaluator(
        args.dataset,
        semantic_model=args.semantic_model,
        semantic_batch_size=args.semantic_batch_size,
        semantic_max_samples=args.semantic_max_samples,
        semantic_max_new_tokens=args.semantic_max_new_tokens,
        use_accelerate=args.accelerate
    )
    
    print("\n" + "=" * 80)
    print("🚀 xLAM-60k Dataset Basic Quality Evaluation")
    print("=" * 80)
    print()
    
    evaluator.load_dataset()
    
    if args.metric == 'all':
        evaluator.run_evaluation()
    elif args.metric == 'format':
        evaluator.evaluate_format_correctness()
    elif args.metric == 'exec':
        evaluator.evaluate_executability()
    elif args.metric == 'semantic':
        evaluator.evaluate_semantic_executability()
    elif args.metric == 'diversity':
        evaluator.evaluate_diversity()
    elif args.metric == 'noise':
        evaluator.evaluate_data_noise()
    
    print("\n" + "=" * 80)
    print("✨ Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

