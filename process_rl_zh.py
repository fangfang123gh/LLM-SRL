import json
from tqdm import tqdm
import re

def find_all_occurrences(key, word):
    positions = []
    start = 0
    while True:
        idx = key.find(word, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1 
    return positions

predpath = ''
savepath=''

with  open(predpath, 'r', encoding='utf-8') as f1, open(savepath, 'w', encoding='utf-8') as fout:
    lines1 = f1.readlines()
    for line1 in tqdm(lines1):
        
        data = json.loads(line1.strip())
        text = data['text']
        token = data['token']
        srl = data['srl']
        pattern = r"@@(.*?)##"
        preds = []
        counts = 0
        new_srl =[]
        for pred in srl:
            arg_response = pred['arguments']
            if arg_response is None:
                pred['arguments'] = []
                new_srl.append(pred)
                continue
            
            arg_response = arg_response.strip()
            pred_pattern = r"@@(.*?)##"
            matches = re.finditer(pred_pattern, arg_response)
            start_pos = None
            
            pred_count = 0
            for match in matches:
                start_pos = match.start()
            
            pattern = r"<([^<>]+)>([^<>]+)</\1>"
            matches = re.finditer(pattern, arg_response)
            count = 0
            args = []
            for match in matches:
                role = match.group(1)
                value = match.group(2)
                start_index = match.start()
                end_index = match.end()
                blank_num = 0
                for t in arg_response[:end_index]:
                    if t == ' ':
                        blank_num += 1
                temp_count = count
                count += len(role) * 2 + 5
                if start_pos is not None:
                    if start_index > start_pos:
                        start_index -= 4
                start_index = start_index - temp_count-blank_num + 1
                
                arg = {'value': value, 'role': role, 'position': [start_index, start_index + len(value) - 1]}
                if text[start_index-1: start_index + len(value) - 1] != value:
                    if value not in text:
                        continue
                    match_index = find_all_occurrences(text, value)
                    # 选一个最近的
                    index = match_index[0]
                    if len(match_index) == 0:
                        continue
                    elif len(match_index) == 1:
                        index = match_index[0]
                    else:
                        min_dis = abs(match_index[0] - start_pos)
                        for match_i in match_index:
                            temp_dis = abs(match_i - start_pos)
                            if temp_dis < min_dis:
                                min_dis = temp_dis
                                index = match_i
                    arg = {'value': value, 'role': role, 'position': [index, index + len(value) - 1]}
                # 找到token的位置
                start_index, end_index = arg['position']
                token_start_index = 0
                if start_index == 1:
                    token_start_index = 0
                else:
                    c = 0
                    for i, t in enumerate(token):
                        if len(t) + c == start_index - 1:
                            token_start_index = i + 1
                            break
                        c = len(t) + c
                
                if end_index == len(text):
                    token_end_index = len(text) - 1
                else:
                    c = 0
                    for i, t in enumerate(token):
                        if len(t) + c == end_index:
                            token_end_index = i
                            break
                        c = len(t) + c
                arg['position'] = [token_start_index+1, token_end_index+1]
                args.append(arg)
            pred['arguments'] = args
            new_srl.append(pred)
        json_string = json.dumps(
            {'text': new_srl},
            ensure_ascii=False
        )
        # fout.write(json_string + "\n")
