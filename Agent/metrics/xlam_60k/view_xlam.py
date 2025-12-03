#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretty viewer for xlam_60k.jsonl dataset
Usage: python view_xlam.py [line_number]
"""
import json
import sys

def view_record(line_num=0):
    """View a specific record from the JSONL file in a readable format"""
    jsonl_path = '/mnt/petrelfs/liuhaoze/datasets/xlam_60k.jsonl'
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == line_num:
                record = json.loads(line)
                
                print("=" * 80)
                print(f"📝 Record #{record['id']}")
                print("=" * 80)
                
                print(f"\n🔍 Query:")
                print(f"   {record['query']}")
                
                print(f"\n✅ Answers ({len(json.loads(record['answers']))}):")
                answers = json.loads(record['answers'])
                for idx, answer in enumerate(answers):
                    print(f"\n   Answer {idx}:")
                    print(json.dumps(answer, indent=6, ensure_ascii=False))
                
                print(f"\n🛠️  Tools ({len(json.loads(record['tools']))}):")
                tools = json.loads(record['tools'])
                for idx, tool in enumerate(tools):
                    print(f"\n   Tool {idx}: {tool['name']}")
                    print(f"   Description: {tool.get('description', 'N/A')}")
                    if 'parameters' in tool:
                        print(f"   Parameters:")
                        for param_name, param_info in tool['parameters'].items():
                            print(f"      • {param_name}:")
                            if isinstance(param_info, dict):
                                print(f"          type: {param_info.get('type', 'N/A')}")
                                print(f"          description: {param_info.get('description', 'N/A')}")
                                if 'default' in param_info:
                                    default_val = param_info['default']
                                    # 使用 repr() 显示原始值，字符串会带引号
                                    print(f"          default: {repr(default_val)}  (type: {type(default_val).__name__})")
                
                print("\n" + "=" * 80)
                return
    
    print(f"❌ Line {line_num} not found")

if __name__ == "__main__":
    line_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    view_record(line_num)

