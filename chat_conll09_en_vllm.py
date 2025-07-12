import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from lemminflect import getLemma
from nltk.corpus import wordnet
# ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'AUX']
def penn_to_lemma(tag):
    if tag.startswith('J'):
        return 'ADJ'
    elif tag.startswith('V'):
        return 'VERB'
    elif tag.startswith('N'):
        return 'NOUN'
    elif tag.startswith('R'):
        return 'ADV'
    elif tag.startswith('P'):
        return 'PROPN'
    else:
        return 'AUX'

def penn_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
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
    system_prompt = 'You are a helpful assistant who have a background in linguistics and are proficient in understanding text, especially semantic role labeling.'
    for line in tqdm(lines):
        data = json.loads(line.strip())
        
        # 遍历对话
        # text = data['text']
        token = data['words']
        pos = data['all_pos']
        lemmas = data['lemmas']
        text = ' '.join(token)
        datas[text] = []
        # if len(lemmas) != len(token):
        #     print(data)
        task_exp = 'Semantic Role Labeling (SRL) aims to identify predicates in a sentence and assign roles to their arguments.\n'
        pred_exp = 'A predicate refers to the core word or phrase in a sentence that conveys an action, event, or state and serves as the focus for other elements in the sentence.\n'
       
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

        output = llm.generate(
            [prompt],
            sampling_params=sampling_params
        )

        response = output[0].outputs[0].text
        messages.append({'role': 'assistant', 'content': response})
        print(response+'\n')
        
        if args.use_pred_agent:
            i = 0
            maybe_pred_pos = []
            maybe_pred_token = []
            while i < len(token):
                t = token[i].lower()
                temp_lemma = lemmas[i]
                # 先直接看在不在里面
                if t in preds_dict:
                    maybe_pred_pos.append(i)
                    maybe_pred_token.append(t)
                elif temp_lemma != '-':
                    if temp_lemma in preds_dict:
                        maybe_pred_pos.append(i)
                        maybe_pred_token.append(temp_lemma)
            
                else:
                    wn_tag = penn_to_wordnet(pos[i])
                    if wn_tag is not None:
                        if lemmatizer.lemmatize(token[i], pos=wn_tag) in preds_dict:
                            maybe_pred_pos.append(i)       
                            maybe_pred_token.append(lemmatizer.lemmatize(token[i], pos=wn_tag) )        
                i += 1

            all_maybe_pr_str = copy.deepcopy(token)
            # maybe_pred_token = set(maybe_pred_token)
            # maybe_pred_token = list(maybe_pred_token)
            pred_agent_des = ''
            for t in maybe_pred_token:
                for key, value in pred_agent[t].items():
                    if len(value) != 0:
                        pos_name = 'noun' if key == 'n' else 'verb'
                        pred_agent_des += f'When the {pos_name} {t} functions as a predicate, its interpretation is: {", ".join(value)}\n'
                

            for a in maybe_pred_pos:
                all_maybe_pr_str[a] ='@@'+ all_maybe_pr_str[a] + '##'
            all_maybe_pr_str = ' '.join(all_maybe_pr_str)
            question = f"Text: {text}\nFor the SRL task, what are the predicates in the given text? Possible predicate results in the text are: {all_maybe_pr_str}\nwhere predicates are specified by @@ and ##.\nBased on the given possible predicate results, please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively. Note that words not present in the predicate results may also be predicates.\n"+pred_agent_des
        else:
            question = f"Text: {text}\nFor the SRL task, what are the predicates in the given text? Please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively. \n"
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

        pre_response = None
        initial_response = response
        print("initial response", response)
        terminate = False
        messages = []
        for i in range(args.pred_max_steps):
            messages = []
            # instruction = f'对于生成的谓词识别结果：{response}'
            instruction = 'Re-evaluate the generated predicate recognition result. Check whether the format of the predicted output is correct, whether each predicate is correctly identified, and whether any predicates are missing. Correct the output based on the identified issues. The format of the result should be consistent with the previous outputs. Use the format: "Issue detected: ...\nPredicate recognition result:\n". If no errors are found, output "Stop checking.".\nn'
            instruction += f'Original text: {text}\nGenerated output to review: {response}\nOutput:\n'

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
        
            if "stop checking" in response.lower():
                print(response)
                # if 'Predicate identification result:' in pre_response:
                #     print("process", pre_response[pre_response.find('Predicate identification result:')+len('Predicate identification result:'):])
                #     if pre_response[pre_response.find('Predicate identification result:')+len('Predicate identification result:'):].strip() == '':
                #         response = 
                # else:
                if i == 0:
                    pre_response = initial_response
                response = pre_response
                terminate = True
                break
            else:
                if 'predicate recognition result:' in response.lower():
                    response = response[response.lower().find('predicate recognition result:')+len('predicate recognition result:'):].strip()
                    if len(response) != 0:

                        pre_response = response
                    # else:
                    #     pre_response = initial_response
        if not terminate:
            if 'predicate recognition result:' in response.lower():
                response = response[response.lower().find('predicate recognition result:')+len('predicate recognition result:'):]
                
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

        preds = []
        pred_matches = re.finditer(pred_pattern, pred_response)
        print("pred_response", pred_response+'\n')
        count = 0
        for match in pred_matches:
            word = match.group(1).strip()
            start_pos = match.start()
            # end_pos = match.end() - 2
            temp_count = count
            count += 4
            blank_space = 0
            for tok in pred_response[:start_pos]:
                if tok == ' ':
                    blank_space += 1
                    
                
            start_pos = start_pos - temp_count - blank_space

            match_index = [i for i, tok in enumerate(token) if tok.lower() == word.lower()]  

            if len(match_index) == 1:
                token_index = match_index[0]
            elif len(match_index) == 0:
                print("生成了新的词")
                continue
            else:
                # 选一个最近的
                token_index = match_index[0]
                min_dis = abs(len(''.join(token[:match_index[0]])) - start_pos)
                for match_i in match_index:
                    temp_dis = abs(len(''.join(token[:match_i])) - start_pos)
                    if temp_dis < min_dis:
                        min_dis = temp_dis
                        token_index = match_i
            pred_arg = {'pred': word, 'position': [token_index + 1, token_index + 1], 'arguments': [] }


            pred_str = copy.deepcopy(token)
            pred_str[token_index] = '@@' + pred_str[token_index] + '##'
            pred_str = ' '.join(pred_str)
            format_question = f'Text: {pred_str}\nWhat are the arguments and their corresponding roles for the given predicate? The predicate is specified by @@ and ##.\n' 


            
            instruction = "In SRL, arguments refer to the components or phrases semantically related to a given predicate. They further describe the entities, actions, or concepts associated with the predicate in the sentence."
            instruction += "Arguments are divided into core arguments and adjunct arguments.\n"
            instruction += "The labels for all adjunct arguments are as follows:\n"
            instruction += 'AM-TMP: temporal\nAM-LOC: location\nAM-MNR: manner\nAM-NEG: negation\n'
            instruction += 'AM-MOD: general modification\nAM-DIS: discourse\nAM-EXT: extent\nAM-ADV: adverbial modification\n'
            instruction += 'AM-PNC: purpose no cause\nAM-DIR: direction\nAM-PRD: secondary predication\nAM-CAU: cause\nAM: argument modification'
            instruction += 'AM-REC: recipricol (eg herself, etc)\nAM-PRT: particle\n'
            instruction+= 'AA: secondary agent\n'

            instruction += "Core arguments depend on the predicate, and a predicate may have different core argument frames. Within these frames, core arguments will have different interpretations.\n"

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
            instruction = format_question
            if args.use_frame_des:
                pred = word
                start = token_index + 1
                # 框架的组织
                frameset_str = ''
                framesets = {}
                lemma = ''
                if lemma not in preds_dict:
                    lemma = pred
                if lemma not in preds_dict:
                    lemma = pred.lower()
                if lemma not in preds_dict:
                    if lemmas[start - 1] != '-':
                        lemma = lemmas[start - 1]
                if lemma not in preds_dict:
                    lemma_list = getLemma(pred.lower(), penn_to_lemma(pos[start - 1]))
                    for lemma in lemma_list:
                        if lemma in preds_dict:
                            break
                
                if lemma not in preds_dict:
                    lemma_list = getLemma(pred.lower(), 'VERB')
                    for lemma in lemma_list:
                        if lemma in preds_dict:
                            break
                if lemma== 'liquefy':
                    lemma = 'liquify'
                if lemma not in preds_dict:
                    t= lemma
                    i = start - 1
                    if i != 0 and token[i-1] == '-':
                        temp_lemma = lemmas[i-2]
                        if ( temp_lemma+t) in preds_dict:
                            lemma = temp_lemma+t
                        elif ( temp_lemma +'-'+ t ) in preds_dict:
                            lemma = temp_lemma +'-'+ t 
                    elif i != len(token) - 1 and token[i + 1] == '-':
                        temp_lemma = lemmas[i+2]
                        if ( t + temp_lemma ) in preds_dict:
                            lemma = t + temp_lemma
                        elif ( t +'-'+ temp_lemma ) in preds_dict:
                            lemma=  t +'-'+ temp_lemma
                if lemma in preds_dict:
                    framesets = {}
                    temp_word = lemma
                    
                    for j in range(1, 5):
                        if start + j <= len(token):
                            if token[start + j - 1] == '-':
                                continue
                            temp_word += ' ' + token[start + j - 1]
                            if temp_word in preds_dict[lemma]:
                                # framesets.append(preds_dict[lemma][temp_word])
                                framesets[temp_word] = preds_dict[lemma][temp_word]
                                break
                    if len(framesets) == 0:
                        if lemma in preds_dict[lemma]:
                            # framesets.append(preds_dict[lemma][lemma])
                            framesets[lemma] = preds_dict[lemma][lemma]
                
                if lemma in preds_dict:
                    for k, _ in preds_dict[lemma].items():
                        # framesets.append(preds_dict[lemma][k])
                        framesets[k] = preds_dict[lemma][k]

                
                if len(framesets) == 0:
                    if pred in preds_dict:
                        if pred in preds_dict[pred]:
                            framesets[pred] = preds_dict[pred][pred]
                        
                        else:
                            if len(preds_dict[pred]) != 0:
                                for key, value in preds_dict[pred].items():
                                    # framesets.append(value)
                                    framesets[key] = value
                    elif lemma in preds_dict:
                        if lemma in preds_dict[lemma]:
                            # framesets.append(preds_dict[lemma][lemma] )
                            framesets[lemma] = preds_dict[lemma][lemma]
                            # framesets = framesets[v_or_n]
                        else:
                            if len(preds_dict[lemma]) != 0:
                                for key, value in preds_dict[lemma].items():
                                    framesets.append(value)
                

                for frame_name, frameset in framesets.items():
                    for key, value in frameset.items():
                        if len(value) != 0:
                            n = 'verb' if key == 'v' else 'noun'
                            frameset_str += f'For {frame_name} as a {n}\n'
                        for fram_index, f in enumerate(value):
                            if len(f) == 0:
                                continue
                            frameset_str += f'Frame {fram_index + 1}:\nThe core arguments it has are:\n'
                            for frame_role, frame_exp in f.items():
                                # frameset_str += f"ARG{frame_role}: {frame_exp}\n"
                                frameset_str += f"A{frame_role}: {frame_exp}\n" 

                if len(frameset_str) != 0:
                    instruction += f'For the predicate "{pred}" in this text, it has the following frames:\n'
                    instruction += frameset_str
                    instruction += "By referring to the provided frames, determine the frame to which the predicate belongs in order to identify its core arguments.\n"
                
                else:
                    instruction += "The labels for all core arguments are as follows:\n"
                    instruction += "A0: agent\nA1: patient\nA2: instrument, benefactive, attribute\nA3: starting point, benefactive, attribute\nA4: ending point\nA5: depend on predicate"
            else:
                instruction += "The labels for all core arguments are as follows:\n"
                instruction += "A0: agent\nA1: patient\nA2: instrument, benefactive, attribute\nA3: starting point, benefactive, attribute\nA4: ending point\nA5: depend on predicate"
            instruction += '"R-" arguments are arguments that are referencing another argument in the sentence. "C-" arguments are discontinous spans that all refer to the same argument. Please rewrite the given text and enclose the beginning and end of the arguments with the corresponding <label> and </label> tags.\n'
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
            messages =[]
            for i in range(args.arg_max_steps):
                messages = []
                instruction = '''Check the generated argument identification result. If any of the following issues exist:
    - Whether the generated text is consistent with the original text;
    - Whether the argument spans are correct;
    - Whether the role labels match the semantic relationship between the predicate and the argument;
    - Whether any arguments are missing.

    Output the identified issues and correct them. The format of the argument annotation should remain consistent.
    Output format example: "There are problems: ...\\nArgument labeling result:".
    If there are no errors, output "Stop checking."

    Original text: <<text>>
    Generated output to evaluate: <<output>>
    Output:
    '''
                instruction = instruction.replace('<<text>>', text)
                instruction = instruction.replace('<<output>>', response)
                # instruction = 'Rethink the generated predicate identification result, evaluating the accuracy of the output predicate format, verify the correct identification of each predicate, and check for any missing predicates.. Identify errors and make corrections accordingly. The predicate recognition result format remains consistent with the previous format. The output format example is: "Issues detected: ...\nPredicate identification result:". If no errors are found, output "Stop checking."'
                # instruction = "检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元标签是否在框架标签和附加标签里；输出格式是否正确；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致。输出该问题并进行纠正，论元标注结果的格式与之前保持一致，输出格式示例为：\"存在问题：论元跨度不正确\n论元标注结果：\"。如果不存在错误，则输出停止检查。"
                # instruction = "检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元标签是否在框架标签和附加标签里；输出格式是否正确；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致。如果存在问题则进行纠正，论元标注结果的格式与之前保持一致。如果不存在错误，则输出停止检查。"
                # instruction = "再次检查所生成的论元识别结果，如果存在错误则进行纠正，输出格式与问题所要求的格式保持一致。如果不存在错误，则输出停止检查。"
                messages.append({"role": "user", "content": instruction})
                prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)  
                # input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids.cuda()

                # output = model.generate(
                #     input_ids=input_ids,
                #     generation_config=generation_config
                # )
                # response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
                output = llm.generate(
                    [prompt],
                    sampling_params=sampling_params
                )
                response = output[0].outputs[0].text
                messages.append({"role": "assistant", "content": response})
                print("response", response)
                if "stop checking" in response.lower():
                    print(response)
                    if 'argument labeling result:' in response.lower():
                        print("process", response[response.lower().find('argument labeling result:')+len('argument labeling result:'):])
                        response = response[response.lower().find('argument labeling result:')+len('argument labeling result:'):]
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
                if 'argument labeling result:' in response.lower():
                    response = response[response.lower().find('argument labeling result:')+len('argument labeling result:'):]
            # if i != args.arg_max_steps -1:
            #     messages.append({"role": "assistant", "content": response})
            pred_arg['arguments'] = response
            preds.append(pred_arg)
        datas[text] = preds
        json_string = json.dumps(
            {'text': text, 'srl': datas[text], 'token':token},
            ensure_ascii=False
        )
        fout.write(json_string + "\n")
      