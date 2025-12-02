import openai
import time
import json
import tiktoken
from rouge_score import rouge_scorer
import numpy as np
from openai import OpenAI
from transformers import AutoModelForSequenceClassification
from bleurt import score
import time

instruction = "Please generate a one-sentence summary for the given document. "
client = OpenAI(api_key="sk-2028f150e9644d748ec98f3b3b498ea0", base_url="https://api.deepseek.com")


def num_tokens_from_message(message, model="text-davinci-003"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens

def get_res_batch(context, instruction):
    
    prompt = instruction + context + " "
    # truncation
    if num_tokens_from_message(prompt) > 4096-128:
        truncation_length = 4096-128
        while(num_tokens_from_message(prompt)>truncation_length):
            prompt = " ".join(prompt.split()[:-1])
            
    messages = [{"role": "user", "content": prompt}]
    

    while True:
        try:
            # res = openai.Completion.create(
            #     model="deepseek-reasoner",
            #     # engine="text-davinci-003",
            #     prompt=prompt,
            #     temperature=0.0,
            #     max_tokens=128
            # )
            start_time = time.time()
            res = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )
            end_time = time.time()
            print(f"API response time: {end_time - start_time} seconds")
            break
        # except openai.error.RateLimitError:
        #     print('openai.error.RateLimitError\nRetrying...')
        #     time.sleep(60)
        # except openai.error.ServiceUnavailableError:
        #     print('openai.error.ServiceUnavailableError\nRetrying...')
        #     time.sleep(20)
        # except openai.error.Timeout:
        #     print('openai.error.Timeout\nRetrying...')
        #     time.sleep(20)
        # except openai.error.APIError:
        #     print('openai.error.APIError\nRetrying...')
        #     time.sleep(20)
        except openai.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
        except Exception as e:
            # 捕获其他未知错误
            print(f"Unexpected error: {str(e)}. Retrying...")
            time.sleep(20)  # 等待 20 秒后重试
    
    # print(res["choices"][0]['text'].strip())
    # return res["choices"][0]['text'].strip()
    sumary = res.choices[0].message.content
    print(sumary)
    return sumary



def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')
    # print('Wrote {} records to {}'.format(len(data), output_path))


def calculate_rouge(reference, hypothesis, n=2):
    """
    计算ROUGE分数
    :param reference: 参考摘要
    :param hypothesis: 生成的摘要
    :param n: 要计算的ROUGE指标（1 = ROUGE-1, 2 = ROUGE-2, 'L' = ROUGE-L）
    :return: ROUGE分数
    """
    # 定义ROUGE scorer
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    
    # 计算ROUGE分数
    scores = scorer.score(reference, hypothesis)
    
    # 返回结果
    return scores[f'rouge{n}']
    

def get_rouge_bleurt(data, device=None):
    
    #init bleurt
    bleurt_checkpoint = '/mnt/petrelfs/wangjiedong/ref_code/BLEURT-20'
    scorer = score.BleurtScorer(bleurt_checkpoint)
    
    
    # with open(file, 'r', encoding="utf-8") as f:
    #     data = []
    #     for line in f:
    #         data.append(json.loads(line))
    rouges = []
    bleurts = []
    for i in range(len(data)):
        context = data[i]["dialogue"]
        reference = data[i]["summary"]

        ans = get_res_batch(context, instruction)
        ans_list = [ans]
        reference_list = [reference]
        rouge_score = calculate_rouge(reference, ans, n=2)
        bleurt_score = scorer.score(references=[reference], candidates=[ans])
        rouges.append(rouge_score.precision)
        bleurts.append(bleurt_score[0])
        print(f"ID {i} rouge: {rouge_score.precision}, bleurt: {bleurt_score[0]}")
    return np.mean(rouges), np.mean(bleurts)









if __name__ == '__main__':
    file = "data/xsum.json"
    
    get_dataset(file, instruction)
