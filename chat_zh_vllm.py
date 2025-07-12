import argparse
import os
import copy
import random
import time
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from gradio import processing_utils
from tqdm import tqdm
import json
import pickle
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import GenerationConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Chat Demo")

    parser.add_argument(
        "--pred_database_path", type=str, default="",
        help="Path to the predicate database path"
    )
    parser.add_argument(
        "--agent_path", type=str, default="",
        help="Path to the predicate database path"
    )
    parser.add_argument(
        "--dataset_type", choices=['cpb', 'conll09'], default='cpb',
        help="which dataset type to infer"
    )
    # /data/lxx/blsp-main/data/cpb_test_predicate_noconv.jsonl
    # /data/lxx/blsp-main/filter_aishell_data/aishell_test_srl_final.jsonl
    parser.add_argument(
        "--input_file", type=str, default="",
        help="Path to the input file"
    )
    parser.add_argument(
        "--output_file", type=str, default="",
        help="Path to the output file"
    )
    parser.add_argument(
        "--model_path", type=str, default="",
        help="Path to the lora adpater"
    )
    ### args for generation
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="top_p for generation"
    )
    parser.add_argument(
        "--repetition_penalty", type=int, default=1,
        help="repetition_penalty for generation"
    )
    parser.add_argument(
        "--pred_max_steps", type=int, default=0,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--arg_max_steps", type=int, default=0,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--use_pred_agent", type=bool, action='store_true',
        help="whether use predicate interpretation"
    )
    parser.add_argument(
        "--use_frame_des", type=bool, action='store_true',
        help="whether use frame interpretation"
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================
print('Initializing Chat')
args = parse_args()
print("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    print("%s: %s" % (arg, value))
print("------------------------------------------------")

with open(args.pred_database_path, 'rb') as f:
    preds_dict = pickle.load(f)

with open(args.agent_path, 'rb') as f:
    pred_agent = pickle.load(f)


sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, repetition_penalty=1, max_tokens=args.max_tokens)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

llm = LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    dtype="float16",  # 或 "auto"
)

print('Initialization Finished')

datas = {}
pred_pattern = r"@@(.*?)##"
rl_pattern = r"<([^<>]+)>([^<>]+)</\1>"

with open(args.input_file, "r", encoding='utf-8') as fin, open(args.output_file, "w", encoding='utf-8') as fout:
    lines = fin.readlines()
    system_prompt = '你是一个有语言学背景并且善于理解文本，特别是在语义角色标注方面熟练的有帮助的助手。'
    for line in tqdm(lines):
        data = json.loads(line.strip())
        
        # 遍历对话
        # text = data['text']
        preds = data['srl']
        key = data['text'] 
        text = key[:]
        token = data['token']
        datas[text] = []
        if args.use_pred_agent:
            i = 0
            maybe_pred_pos = []
            maybe_pred_token = []
            while i < len(token):
                if token[i] in preds_dict:
                    maybe_pred_token.append(token[i])
                    maybe_pred_pos.append((len(''.join(token[:i])) +1, len(''.join(token[:i])) +len(token[i])))
                        
                i += 1

            maybe_pred_pos = sorted(maybe_pred_pos, key=lambda x: x[0])
            all_maybe_pr_str = text
            count = 0
            for a in maybe_pred_pos:
                b, e = a
                value = text[b - 1 : e]
                if e == len(text):
                    all_maybe_pr_str = all_maybe_pr_str[0: b-1 + count] + ' @@' + value + '##'
                    count += 5
                elif b == 1:
                    all_maybe_pr_str = '@@' + value + '## ' + all_maybe_pr_str[e+count:]
                    count += 5
                else:
                    all_maybe_pr_str = all_maybe_pr_str[0: b-1 + count] + ' @@' + value + '## ' + all_maybe_pr_str[e+count:]
                    count += 6
            pred_agent_des = ''
            maybe_pred_token = set(maybe_pred_token)
            maybe_pred_token = list(maybe_pred_token)
            for t in maybe_pred_token:
                if t in pred_agent:
                    pred_agent_des += f'当{t}作为谓词时，它的意思为：{", ".join(pred_agent[t])}\n'
        
        task_exp = '语义角色标注（SRL）旨在识别句子中的谓词并为它们的论元分配角色。\n'
        pred_exp = '谓词指的是句子中的核心词或短语，表示动作、事件或状态，并作为句子中其他成分的焦点。它通常是动词或形容词。\n'
       
        # label_exp = '论元是指与给定谓词在语义上相关的成分或短语。它进一步描述与句子中的谓词相关联的实体、动作或概念。以下是关于论元的所有角色标签及其解释：\nARG0：执行动作或事件的实体\nARG1：承受动作或事件的实体\nARG2：根据谓词的不同，通常是动作或事件的目标或对象\nARG3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nARG4：根据谓词的不同，通常是动作或事件的来源或起源\nARGM-ADV：副词性论元\nARGM-BNF：受益者\nARGM-CND：条件\nARGM-DIR：方向\nARGM-EXT：表示动作或事件的程度\nARGM-FRQ：表示动作或事件的频率\nARGM-LOC：表示动作或事件发生的地点\nARGM-MNR：表示动作或事件的执行方式\nARGM-PRP：目的\nARGM-TMP：时间\nARGM-TPC：主题\nARGM-DGR：程度\nARGM-DIS：话语\nARGM-CRD：并列\nARGM-PRD：子谓词\n前缀“ARG”表示核心论元，前缀“ARGM”表示语义修饰语。\n'
        task_exp += pred_exp
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_exp}
        ]
        # 谓词介绍
        prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)  
        # input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids.cuda()

        output = llm.generate(
            [prompt],
            sampling_params=sampling_params
        )

        response = output[0].outputs[0].text
        messages.append({'role': 'assistant', 'content': response})
        print(response+'\n')
        if args.use_pred_agent:
            question = f'文本: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？该文本中可能的谓词结果为: {all_maybe_pr_str}\n其中谓词由@@和##给定。\n结合给定的可能的谓词结果，请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。注意没有出现在谓词结果的词也可能是谓词。\n'+pred_agent_des
        else:
            question = f'文本: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。'
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True) 
        output = llm.generate(
            [prompt],
            sampling_params=sampling_params
        )

        response = output[0].outputs[0].text

        # 在这谓词检查
        # 在这里插入询问
        pre_response = None
        initial_response = response
        print("initial response", response)
        terminate = False
        for i in range(args.pred_max_steps):
            instruction = '重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\"。如果不存在错误，则输出停止检查。'
            instruction += f'\n原文本: {key}\n需要思考的输出结果：{response}\n输出：\n'
        
            messages.append({"role": "user", "content": instruction})
            prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)  
        
            output = llm.generate(
                [prompt],
                sampling_params=sampling_params
            )

            response = output[0].outputs[0].text
            messages.append({"role": "assistant", "content": response})
            print("response", response)
        
            if "停止检查" in response.lower():

                if i == 0:
                    pre_response = initial_response
                response = pre_response
                terminate = True
                break
            else:
                if '谓词识别结果：' in response:
                    response = response[response.find('谓词识别结果：')+len('谓词识别结果：'):].strip()
                    if len(response) != 0:

                        pre_response = response
                    # else:
                    #     pre_response = initial_response
        if not terminate:
            if '谓词识别结果：' in response:
                response = response[response.find('谓词识别结果：')+len('谓词识别结果：'):]
                
                if len(response) == 0:
                    if pre_response is not None:
                        response = pre_response
                    else:
                        response = None
        if response is None:
            response = initial_response

        if response == "":
            json_string = json.dumps(
                {'text': text, 'srl': datas[text], 'token':token},
                ensure_ascii=False
            )
            fout.write(json_string + "\n")
            continue
        pred_response = response.strip()
        messages.append({"role": "assistant", "content": response})

        
        messages = []
        preds = []
        pred_matches = re.finditer(pred_pattern, pred_response)
        print("pred_response", pred_response+'\n')
        count = 0
        for match in pred_matches:
            word = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            blank_num = 0
            for t in pred_response[:end_pos]:
                if t == ' ':
                    blank_num += 1
            temp_count = count
            count += 4
            start_pos = start_pos - temp_count-blank_num + 1
            pred_arg = {'pred': word, 'position': [start_pos, start_pos + len(word) - 1], 'arguments': []}
            if key[start_pos-1:start_pos + len(word) - 1] != word:
                print("提取错误")
                print("key", key)
                print(key[start_pos-1:start_pos + len(word) - 1], word)
                continue
            if args.dataset_type == 'cpb':
                instruction = "在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。"
                instruction += "论元分为核心论元和附加论元。\n"
                instruction += "所有附加论元的标签如下：\n"
                instruction += "ARGM-ADV：adverbial\nARGM-BNF：beneficiary\nARGM-CND：conditional\nARGM-DIR：direction\nARGM-EXT：extent\nARGM-FRQ：frequency\nARGM-LOC：location\nARGM-MNR：manner\nARGM-PRP：purpose\nARGM-TMP：temporal\nARGM-TPC：topic\nARGM-DGR：degree\nARGM-DIS：discourse marker\nARGM-CRD：coordinator\nARGM-PRD：predicate\n"
                instruction += "核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n"
            else:
                instruction = "在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。"
                instruction += "论元分为核心论元和附加论元。\n"
                instruction += "所有附加论元的标签如下：\n"
                instruction += "ADV: Adverbial argument\nBNF: Beneficiary\nCND: Condition\nDIR: Direction\nEXT: Degree\nFRQ: Frequency\nLOC: Location\nMNR: Manner of action or event execution\nPRP: Purpose\nTMP: Time\nTPC: Topic\nDGR: Degree\nDIS: Discourse marker\nPRD: Sub-predicate\nVOC: Particle\n"
                instruction += '对于复合短语，第一个短语的类型是论元的类型，例如A0，而后面的短语的类型是C-A0。'
                instruction += "核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n"
            messages.append({"role": "user", "content": instruction})
            prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)  
         
            ooutput = llm.generate(
                [prompt],
                sampling_params=sampling_params
            )

            response = output[0].outputs[0].text
            messages.append({"role": "assistant", "content": response})

            messages = messages[:8]
      
            text = key[:]
            pred = word
            position = pred_arg['position']
            if position[0] == 1:
                text = '@@' + pred + '## ' + key[position[1]:] 
            elif position[1] == len(key):
                text = key[0:position[0]-1] + ' @@' + pred + '##'
            else:
                text = key[0:position[0]-1] + ' @@' + pred + '## ' + key[position[1]:] 
            question = f"Text: {text}\n给定谓词的论元及其对应的角色是什么？谓词由@@和##给定。\n"
            instruction = question
            
            if args.use_frame_des:
                if args.dataset_type == 'cpb':
                    # 框架的组织
                    frameset_str = ''
                    if pred in preds_dict:
                        framesets = preds_dict[pred]

                        for idx, frameset in enumerate(framesets, 1):
                            frameset_str += f'框架 {idx}: \n它具有的核心论元是：\n'
                            for frame_role, frame_exp in frameset.items():
                                frameset_str += f"ARG{frame_role}: {frame_exp}\n"
                                # frameset_str += f"A{frame_role}: {frame_exp}\n"
                    if len(frameset_str) != 0:
                        instruction += f'对于谓词"{pred}", 它具有以下框架：\n'
                        instruction += frameset_str
                        instruction += "通过参考提供的框架，确定谓词所属的框架，以确定其核心论元。\n"
                        
                    else:
                        instruction += "所有核心论元的标签如下：\n"
                        instruction += "ARG0：执行动作或事件的实体\nARG1：承受动作或事件的实体\nARG2：根据谓词的不同，通常是动作或事件的目标或对象\nARG3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nARG4：根据谓词的不同，通常是动作或事件的来源或起源\n"
                else:
                    frameset_str = ''
                if pred in preds_dict:
                    framesets = preds_dict[pred]

                    for idx, frameset in enumerate(framesets, 1):
                        frameset_str += f'框架 {idx}: \n它具有的核心论元是：\n'
                        for frame_role, frame_exp in frameset.items():
                            frameset_str += f"A{frame_role}: {frame_exp}\n"
                            # frameset_str += f"A{frame_role}: {frame_exp}\n"
                if len(frameset_str) != 0:
                    instruction += f'对于谓词"{pred}", 它具有以下框架：\n'
                    instruction += frameset_str
                    instruction += "通过参考提供的框架，确定谓词所属的框架，以确定其核心论元。\n"
                    
                else:
                    instruction += "所有核心论元的标签如下：\n"
                    instruction += "A0：执行动作或事件的实体\nA1：承受动作或事件的实体\nA2：根据谓词的不同，通常是动作或事件的目标或对象\nA3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nA4：根据谓词的不同，通常是动作或事件的来源或起源\n"
            else:
                if args.dataset_type == 'cpb':
                    instruction += "所有核心论元的标签如下：\n"
                    instruction += "ARG0：执行动作或事件的实体\nARG1：承受动作或事件的实体\nARG2：根据谓词的不同，通常是动作或事件的目标或对象\nARG3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nARG4：根据谓词的不同，通常是动作或事件的来源或起源\n"
                else:
                    instruction += "所有核心论元的标签如下：\n"
                    instruction += "A0：执行动作或事件的实体\nA1：承受动作或事件的实体\nA2：根据谓词的不同，通常是动作或事件的目标或对象\nA3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nA4：根据谓词的不同，通常是动作或事件的来源或起源\n"
            instruction += "请重新编写给定文本，并使用相应的<label>和</label>标签将论元的开头和结尾括起来。\n"
            
            messages.append({"role": "user", "content": instruction})
            prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)  
        
            output = llm.generate(
                [prompt],
                sampling_params=sampling_params
            )

            response = output[0].outputs[0].text
            pre_response = None
            initial_response = response
            print("initial response", response)
            terminate = False
            for i in range(args.arg_max_steps):
                messages = []
                instruction = f"检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致；是否存在未识别的论元。输出该问题并进行纠正，论元标注结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n论元标注结果：\"。如果不存在错误，则输出停止检查。\n原文本: {key}\n需要思考的输出结果： {response}\n输出：\n"
        
                messages.append({"role": "user", "content": instruction})
                prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)  
           
                output = llm.generate(
                    [prompt],
                    sampling_params=sampling_params
                )
                response = output[0].outputs[0].text
                messages.append({"role": "assistant", "content": response})
                print("response", response)
                if "停止检查" in response.lower():
                    print(response)
                    if '论元标注结果：' in response:
                        print("process", response[response.find('论元标注结果：')+len('论元标注结果：'):])
                        response = response[response.find('论元标注结果：')+len('论元标注结果：'):]
                    else:
                        if pre_response is None:
                            response = initial_response
                        else:
                            response = pre_response
                    terminate = True
                    break
                else:
                    pre_response = response
            if not terminate:
                if '论元标注结果：' in response:
                    response = response[response.find('论元标注结果：')+len('论元标注结果：'):]
           
            pred_arg['arguments'] = response
            preds.append(pred_arg)
        datas[key] = preds
        json_string = json.dumps(
            {'text': text, 'srl': datas[text], 'token':token},
            ensure_ascii=False
        )
        fout.write(json_string + "\n")
      