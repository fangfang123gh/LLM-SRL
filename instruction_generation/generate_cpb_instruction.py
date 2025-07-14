

import json
import copy
import pickle
from tqdm import tqdm
import re

# 与trainer中的处理保持一致
def process_pred(response):
    pred_pattern = r"@@(.*?)##"
    matches1 = re.finditer(pred_pattern, response)
    start_pos = None
    count = 0
    all_pred = []
    
    for match in matches1:
        word = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        blank_num = 0
        for t in response[:end_pos]:
            if t == ' ':
                blank_num += 1
        temp_count = count
        count += 4
        start_pos = start_pos - temp_count-blank_num + 1
        pred_arg = {'pred': word, 'position': [start_pos, start_pos + len(word) - 1]}
        all_pred.append(pred_arg)
    return all_pred

def process_arg(response):
    pred_pattern = r"@@(.*?)##"
    pattern = r"<([^<>]+)>([^<>]+)</\1>"
    matches1 = re.finditer(pred_pattern, response)
    start_pos = None
    count = 0
    pred_arg = {}
    for match in matches1:
        word = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        blank_num = 0
        for t in response[:end_pos]:
            if t == ' ':
                blank_num += 1
        temp_count = count
        count += 4
        start_pos = start_pos - temp_count-blank_num + 1
        pred_arg = {'pred': word, 'position': [start_pos, start_pos + len(word) - 1], 'arguments': []}
    
    if len(pred_arg) ==0 :
        return pred_arg
    start_pos = None
    for match in matches1:
        start_pos = match.start()
    
    
    matches = re.finditer(pattern, response)
    arg_count = 0
    new_args = []
    for match in matches:
        role = match.group(1)
        value = match.group(2)
        start_index = match.start()
        end_index = match.end()
        # print("标签：", role)
        # print("文本：", value)
        
        blank_num = 0
        for t in response[:end_index]:
            if t == ' ':
                blank_num += 1
        temp_count = arg_count
        arg_count += len(role) * 2 + 5
        if start_pos is not None:
            if start_index > start_pos:
                start_index -= 4
        start_index = start_index - temp_count-blank_num + 1
        arg = {'value': value, 'role': role, 'position': [start_index, start_index + len(value) - 1]}

        new_args.append(arg)
    pred_arg['arguments'] = new_args
    return pred_arg

def get_instruction_data(predicate_base_path, agent_path, data_path, save_path, require_pred=True, require_rl=True):
    datas = []
    with open(predicate_base_path, 'rb') as f:
        preds_dict = pickle.load(f)

    with open(agent_path, 'rb') as f:
        pred_agent = pickle.load(f)

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            preds = data['srl']
            key = data['text'] 
            token = data['token']
            text = key
            
            pr_str = key
            sorted_result = sorted(preds, key=lambda x: x['position'][0])
            count = 0
            for a in sorted_result:
                b, e = a['position']
                if e == len(key):
                    pr_str = pr_str[0: b-1 + count] + ' @@' + a['pred'] + '##'
                    count += 5
                elif b == 1:
                    pr_str = '@@' + a['pred'] + '## ' + pr_str[e+count:]
                    count += 5
                else:
                    pr_str = pr_str[0: b-1 + count] + ' @@' + a['pred'] + '## ' + pr_str[e+count:]
                    count += 6
            if require_pred:
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
                    # ar = "\"" + a['value'] + '[{},{}]'.format(a['position'][0],a['position'][1]) + "\""
                    # arg_answer.append(ar)
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
        

            for r in preds:
                conversations = []
                task_exp = '语义角色标注（SRL）旨在识别句子中的谓词并为它们的论元分配角色。\n'
                pred_exp = '谓词指的是句子中的核心词或短语，表示动作、事件或状态，并作为句子中其他成分的焦点。它通常是动词或形容词。\n'
                conversations.append({'from': 'human', "value": task_exp + pred_exp})
                conversations.append({"from": "gpt", "value": '我已经理解了这个任务。'})
                if require_pred:
                    conversations.append({'from': 'human', "value": f'Text: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？该文本中可能的谓词结果为: {all_maybe_pr_str}\n其中谓词由@@和##给定。\n结合给定的可能的谓词结果，请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。注意没有出现在谓词结果的词也可能是谓词。\n'+pred_agent_des})
                else:
                    conversations.append({'from': 'human', "value": f'Text: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。'})
                conversations.append({"from": "gpt", "value": pr_str})

                instruction = "在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。"
                instruction += "论元分为核心论元和附加论元。\n"
                instruction += "所有附加论元的标签如下：\n"
                instruction += "ARGM-ADV：adverbial\nARGM-BNF：beneficiary\nARGM-CND：conditional\nARGM-DIR：direction\nARGM-EXT：extent\nARGM-FRQ：frequency\nARGM-LOC：location\nARGM-MNR：manner\nARGM-PRP：purpose\nARGM-TMP：temporal\nARGM-TPC：topic\nARGM-DGR：degree\nARGM-DIS：discourse marker\nARGM-CRD：coordinator\nARGM-PRD：predicate\n"
                pred = r['pred']
                
                instruction += "核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n"
                conversations.append({'from': 'human', "value": instruction})
                conversations.append({"from": "gpt", "value": '我已经理解了这个任务。'})
                instruction = ''

                position = r['position']
                args = r['arguments']
                text = copy.deepcopy(token)
                if position[0] == 1:
                    text[position[0]-1] = '@@' + text[position[0]-1]
                else:
                    text[position[0]-1] = ' @@' + text[position[0]-1]
                if position[1] == len(text):
                    text[position[1]-1] =text[position[1]-1]+ '##'  
                else:
                    text[position[1]-1] = text[position[1]-1]+ '## ' 
                question = f"Text: {text}\n给定谓词的论元及其对应的角色是什么？谓词由@@和##给定。\n"
                instruction += question
                
                
                if require_rl:
                    frameset_str = ''
                    if pred in preds_dict:
                        framesets = preds_dict[pred]

                        for idx, frameset in enumerate(framesets, 1):
                            frameset_str += f'框架 {idx}: \n它具有的核心论元是：\n'
                            for frame_role, frame_exp in frameset.items():
                                frameset_str += f"ARG{frame_role}: {frame_exp}\n"
                    if len(frameset_str) != 0:
                        instruction += f'对于谓词"{pred}", 它具有以下框架：\n'
                        instruction += frameset_str
                        instruction += "通过参考提供的框架，确定谓词所属的框架，以确定其核心论元。\n"
                        
                    else:
                        instruction += "所有核心论元的标签如下：\n"
                        instruction += "ARG0：执行动作或事件的实体\nARG1：承受动作或事件的实体\nARG2：根据谓词的不同，通常是动作或事件的目标或对象\nARG3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nARG4：根据谓词的不同，通常是动作或事件的来源或起源\n"
                else:
                    instruction += "所有核心论元的标签如下：\n"
                    instruction += "ARG0：执行动作或事件的实体\nARG1：承受动作或事件的实体\nARG2：根据谓词的不同，通常是动作或事件的目标或对象\nARG3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nARG4：根据谓词的不同，通常是动作或事件的来源或起源\n"
                instruction += "请重新编写给定文本，并使用相应的<label>和</label>标签将论元的开头和结尾括起来。\n"
                arg_str = copy.deepcopy(token)
                count = 0
                flag = True
                sorted_args = sorted(args, key=lambda x: x['position'][0])
                # if position[0] == position[1]:
                if position[0] == 1:
                    arg_str[position[0]-1] = '@@' + arg_str[position[0]-1]
                else:
                    arg_str[position[0]-1] = ' @@' + arg_str[position[0]-1]
                if position[1] == len(arg_str):
                    arg_str[position[1]-1] =arg_str[position[1]-1]+ '##'  
                else:
                    arg_str[position[1]-1] = arg_str[position[1]-1]+ '## '  
                for a in sorted_args:
                    # if a['role'] == 'ARGM-PRX':
                    #     continue
                    role = a['role']
                    start, end = a['position']
                    if start==1:
                        arg_str[start - 1] = f'<{role}>' + arg_str[start - 1]
                    else:

                        arg_str[start - 1] = f' <{role}>' + arg_str[start - 1]
                    if end==len(arg_str):
                        arg_str[end - 1] = arg_str[end - 1] + f'</{role}>'
                    else:
                        arg_str[end - 1] = arg_str[end - 1] + f'</{role}> '

                conversations.append({"from": "human", "value": instruction})
                gpt_template = {"from": "gpt", "value": ''.join(arg_str) }
                print("arg_str", ''.join(arg_str) )
                conversations.append(gpt_template)

                datas.append({'conversations': conversations, 'system': "你是一个有语言学背景并且善于理解文本，特别是在语义角色标注方面熟练的有帮助的助手。", 'text': key, 'pred_text': text, "gold_pred": json.dumps(process_pred(pr_str), ensure_ascii=False), "gold_rl": json.dumps(process_arg(''.join(arg_str)), ensure_ascii=False)})
                
    json_data = json.dumps(datas,ensure_ascii=False)
    with open(save_path, "w", encoding='utf-8') as file:
        file.write(json_data)

if __name__ == '__main__':
    predicate_base_path = '/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/LLM_SRL/blsp-main/data_process/predicate_framset.pkl' # the predicate database path
    agent_path = '/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/LLM_SRL/srl/chinese_predict_frameset.pkl' # the predicate agent path
    data_path = '/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/LLM_SRL/universal_sp/data/cpb1.0/cpb_train_filter_final_all_token.jsonl' # the training data
    save_path = '/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/LLM_SRL/srl_data/test.json'
    get_instruction_data(predicate_base_path, agent_path, data_path, save_path)


