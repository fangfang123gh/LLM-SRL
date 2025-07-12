# Please install OpenAI SDK first: `pip3 install openai`

import json
import re
import pickle
import time
from tqdm import tqdm

def process_text(text):
    def replace_with_placeholder(match):
        return f"__PLACEHOLDER_{match.group(0)}__"
    text = re.sub(r'@@([^#@]*?)##', replace_with_placeholder, text)
    patterns = [
        (r'__PLACEHOLDER_@@stock## specialist firms##__', r'@@stock## specialist firms'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    text = text.replace(r'__PLACEHOLDER_', '')
    text = text.replace(r'__', '')
    return text

def run_srl_infer(
    api_key,
    input_file,
    output_file,
    preds_database_path,
    pred_agent_path,
    base_url="https://api.deepseek.com",
    start_line=-300,
    end_line=-200,
    max_retries=5,
    delay=10
):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    pred_pattern = r"@@(.*?)##"
    rl_pattern = r"<([^<>]+)>([^<>]+)</\1>"

    with open(preds_database_path, 'rb') as f:
        preds_dict = pickle.load(f)
    with open(pred_agent_path, 'rb') as f:
        pred_agent = pickle.load(f)

    datas = {}
    with open(input_file, "r", encoding='utf-8') as fin, open(output_file, "w", encoding='utf-8') as fout:
        lines = fin.readlines()[start_line:end_line]
        system_prompt = '你是一个有语言学背景并且善于理解文本，特别是在语义角色标注方面熟练的有帮助的助手。'
        for line in tqdm(lines):
            data = json.loads(line.strip())
            preds = data['srl']
            key = data['text']
            text = key
            token = data['token']
            datas[text] = []

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
                    pred_agent_des += f'当{t}作为谓词是，它的意思为：{", ".join(pred_agent[t])}\n'

            task_exp = '语义角色标注（SRL）旨在识别句子中的谓词并为它们的论元分配角色。\n'
            pred_exp = '谓词指的是句子中的核心词或短语，表示动作、事件或状态，并作为句子中其他成分的焦点。它通常是动词或形容词。\n'
            task_exp += pred_exp

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_exp}
            ]
            messages.append({'role': 'assistant', 'content': "我已经理解了这个任务。"})

            question = f'文本: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？该文本中可能的谓词结果为: {all_maybe_pr_str}\n其中谓词由@@和##给定。\n结合给定的可能的谓词结果，请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。注意没有出现在谓词结果的词也可能是谓词。\n'+pred_agent_des
            question += '无需提供任何理由或引导语句。只需直接输出改写后的文本，且结果必须按照所要求的格式给出，否则视为识别错误。'
            question += '请参考以下输出格式：我 @@去## 学校'
            messages.append({"role": "user", "content": question})

            retries = 0
            response = ''
            while retries < max_retries:
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=False
                    )
                    if resp:
                        response = resp.choices[0].message.content
                        break
                except Exception as e:
                    response = ''
                    time.sleep(delay)
                    retries += 1

            if response == "":
                json_string = json.dumps(
                    {key: datas[key]},
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
                start_pos_real_start = None
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
                    print("提取错误 或者文本错误")
                    pattern = r"\b\w+\b"
                    match_index = [m.start() for m in re.finditer(pattern, key)]
                    if len(match_index) == 1:
                        start_pos_real_start = match_index[0]
                    elif len(match_index) == 0:
                        print("生成了新的词")
                        continue
                    else:
                        start_pos_real_start = match_index[0]
                        min_dis = abs(len(''.join(token[:match_index[0]])) - start_pos)
                        for match_i in match_index:
                            temp_dis = abs(len(''.join(token[:match_i])) - start_pos)
                            if temp_dis < min_dis:
                                min_dis = temp_dis
                                start_pos_real_start = match_i

                pred = word
                if start_pos_real_start is not None:
                    new_key = process_text(pred_response)
                    end_pos_real_end = start_pos_real_start + len(word)
                    if start_pos_real_start == 0:
                        text = '@@' + pred + '## ' + new_key[start_pos_real_start+1:]
                    elif end_pos_real_end == len(text):
                        text = new_key[0:start_pos_real_start] + ' @@' + pred + '##'
                    else:
                        text = new_key[0:start_pos_real_start] + ' @@' + pred + '## ' + new_key[end_pos_real_end:]
                else:
                    text = key
                    position = pred_arg['position']
                    if position[0] == 1:
                        text = '@@' + pred + '## ' + key[position[1]:]
                    elif position[1] == len(key):
                        text = key[0:position[0]-1] + ' @@' + pred + '##'
                    else:
                        text = key[0:position[0]-1] + ' @@' + pred + '## ' + key[position[1]:]

                instruction = "在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。"
                instruction += "论元分为核心论元和附加论元。\n"
                instruction += "所有附加论元的标签如下：\n"
                instruction += "ARGM-ADV：adverbial\nARGM-BNF：beneficiary\nARGM-CND：conditional\nARGM-DIR：direction\nARGM-EXT：extent\nARGM-FRQ：frequency\nARGM-LOC：location\nARGM-MNR：manner\nARGM-PRP：purpose\nARGM-TMP：temporal\nARGM-TPC：topic\nARGM-DGR：degree\nARGM-DIS：discourse marker\nARGM-CRD：coordinator\nARGM-PRD：predicate\n"
                instruction += "Core arguments depend on the predicate, and a predicate may have different core argument frames. Within these frames, core arguments will have different interpretations.\n"
                instruction += "核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n"
                messages.append({"role": "user", "content": instruction})
                messages.append({"role": "assistant", "content": "I have understood this task."})
                messages = messages[:8]
                instruction = ''
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
                instruction += f"\n\n给定谓词的文本: {text}\n给定谓词的论元及其对应的角色是什么？论元是基于跨度的。谓词由@@和##给定。\n"
                instruction += "请重新编写给定谓词的文本，这个谓词是给定不需要变化的，并使用相应的<角色标签>和</角色标签>标签将论元的开头和结尾括起来。其中，ARGN类型（比如ARG0）的标签只会出现一次。\n"
                instruction += '无需提供任何理由或引导语句。只需直接输出改写后的文本，且结果必须按照所要求的格式给出，否则视为识别错误。'
                instruction += '对于当前的给定谓词的文本，请参考以下输出格式：<ARG0>我</ARG0> @@去## <ARG1>学校</ARG1>'
                messages.append({"role": "user", "content": instruction})

                retries = 0
                response2 = ''
                while retries < max_retries:
                    try:
                        resp2 = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            stream=False
                        )
                        if resp2:
                            response2 = resp2.choices[0].message.content
                            break
                    except Exception as e:
                        response2 = ''
                        time.sleep(delay)
                        retries += 1

                print("response", response2)
                messages.append({"role": "assistant", "content": response2})
                pred_arg['arguments'] = response2
                preds.append(pred_arg)
            datas[key] = preds
            json_string = json.dumps(
                {key: datas[key]},
                ensure_ascii=False
            )
            fout.write(json_string + "\n")

if __name__ == "__main__":
    # 示例调用，实际使用时请替换为你的路径和key
    run_srl_infer(
        api_key='your api key',
        input_file='your_input.jsonl',
        output_file='your_output.jsonl',
        preds_database_path='your_preds_database.pkl',
        pred_agent_path='your_pred_agent.pkl',
        base_url="https://api.deepseek.com",
        start_line=-300,
        end_line=-200,
        max_retries=5,
        delay=10
    )
