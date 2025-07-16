

import json
import copy
import pickle
from tqdm import tqdm
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
            
            pr_str = token[:]
            sorted_result = sorted(preds, key=lambda x: x['position'][0])
            count = 0
            for a in sorted_result:
                position = a['position']
                if position[0] == 1:
                    pr_str[position[0]-1] = '@@' + pr_str[position[0]-1]
                else:
                    pr_str[position[0]-1] = ' @@' + pr_str[position[0]-1]
                if position[1] == len(pr_str):
                    pr_str[position[1]-1] =pr_str[position[1]-1]+ '##'  
                else:
                    pr_str[position[1]-1] = pr_str[position[1]-1]+ '## ' 
            pr_str = ''.join(pr_str)
            i = 0
            maybe_pred_pos = []
            maybe_pred_token = []
            while i < len(token):
                if token[i] in preds_dict:
                    maybe_pred_token.append(token[i])
                    maybe_pred_pos.append((len(''.join(token[:i])) +1, len(''.join(token[:i])) +len(token[i])))
                        
                i += 1
            maybe_pred_pos = sorted(maybe_pred_pos, key=lambda x: x[0])
            pred_agent_des = ''
            all_maybe_pr_str = ''
            if require_pred:
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

                instruction = "在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。识别的论元应该是"
                instruction += "论元分为核心论元和附加论元。\n"
                instruction += "所有附加论元的标签如下：\n"
                instruction += "ADV: Adverbial argument\nBNF: Beneficiary\nCND: Condition\nDIR: Direction\nEXT: Degree\nFRQ: Frequency\nLOC: Location\nMNR: Manner of action or event execution\nPRP: Purpose\nTMP: Time\nTPC: Topic\nDGR: Degree\nDIS: Discourse marker\nPRD: Sub-predicate\nVOC: Particle\n"
                instruction += '对于复合短语，第一个短语的类型是论元的类型，例如A0，而后面的短语的类型是C-A0。'
                pred = r['pred']
                
                instruction += "核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n"
                conversations.append({'from': 'human', "value": instruction})
                conversations.append({"from": "gpt", "value": '我已经理解了这个任务。'})
                instruction = ''

                position = r['position']
                args = r['arguments']
                text = key
                if position[0] == 1:
                    text = '@@' + pred + '## ' + key[position[1]:] 
                elif position[1] == len(key):
                    text = key[0:position[0]-1] + ' @@' + pred + '##'
                else:
                    text = key[0:position[0]-1] + ' @@' + pred + '## ' + key[position[1]:] 
                question = f"Text: {text}\n给定谓词的论元及其对应的角色是什么？谓词由@@和##给定。\n"
                instruction += question
                if require_rl:
                    frameset_str = ''
                    if pred in preds_dict:
                        framesets = preds_dict[pred]
                        for idx, frameset in enumerate(framesets, 1):
                            frameset_str += f'框架 {idx}: \n它具有的核心论元是：\n'
                            for frame_role, frame_exp in frameset.items():
                                frameset_str += f"A{frame_role}: {frame_exp}\n"
                    if len(frameset_str) != 0:
                        instruction += f'对于谓词"{pred}", 它具有以下框架：\n'
                        instruction += frameset_str
                        instruction += "通过参考提供的框架，确定谓词所属的框架，以确定其核心论元。\n"
                        
                    else:
                        instruction += "所有核心论元的标签如下：\n"
                        instruction += "A0：执行动作或事件的实体\nA1：承受动作或事件的实体\nA2：根据谓词的不同，通常是动作或事件的目标或对象\nA3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nA4：根据谓词的不同，通常是动作或事件的来源或起源\n"
                else:
                    instruction += "所有核心论元的标签如下：\n"
                    instruction += "A0：执行动作或事件的实体\nA1：承受动作或事件的实体\nA2：根据谓词的不同，通常是动作或事件的目标或对象\nA3：根据谓词的不同，通常是动作或事件的间接接受者或受益者\nA4：根据谓词的不同，通常是动作或事件的来源或起源\n"
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
                conversations.append(gpt_template)
                datas.append({'conversations': conversations, 'system': "你是一个有语言学背景并且善于理解文本，特别是在语义角色标注方面熟练的有帮助的助手。"})

    json_data = json.dumps(datas,ensure_ascii=False)
    with open(save_path, "w", encoding='utf-8') as file:
        file.write(json_data)

if __name__ == '__main__':
    predicate_base_path = '' # the predicate database path
    agent_path = '' # the predicate agent path
    data_path = '' # the training data
    save_path = ''
    get_instruction_data(predicate_base_path, agent_path, data_path, save_path)


    