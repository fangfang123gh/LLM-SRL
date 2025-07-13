import json
import re
from tqdm import tqdm
import re
pred_pattern = r"@@(.*?)##"
pattern = r"<([^<>]+)>([^<>]+)</\1>"




def extract_arguments_from_string(argument_str, token):
    """
    从带标注的 argument_str 字符串中提取语义角色标注结果，并映射回原始 token 索引。
    
    参数：
    - argument_str: str, 含有 <ARG>...</ARG> 和 @@...## 标记的预测字符串
    - token: List[str], 原始分词后的文本

    返回：
    - List[Dict], 每个元素是 {'value': str, 'role': str, 'position': [start, end]}
    """

    pred_pattern = r"@@(.*?)##"
    pattern = r"<([^<>]+)>([^<>]+)</\1>"

    matches1 = re.finditer(pred_pattern, argument_str)
    start_pos = None
    for match in matches1:
        start_pos = match.start()

    matches = re.finditer(pattern, argument_str)
    new_args = []
    arg_count = 0

    for match in matches:
        role = match.group(1)
        value = match.group(2)
        start_index = match.start()
        end_index = match.end()
        
        blank_space = argument_str[:start_index].count(' ')
        blank_space_end = argument_str[:end_index].count(' ')

        temp_count = arg_count
        arg_count += 5 + len(role) * 2

        if start_pos is not None and start_index > start_pos:
            start_index -= 4
            end_index -= 4

        start_index = start_index - temp_count - blank_space + 1
        end_index = end_index - arg_count - blank_space_end

        all_value = value.split()
        match_index = [i for i, tok in enumerate(token) if tok == all_value[0]]

        if not match_index:
            for sep in ["n't", "'", ',', '.', '?', '...']:
                if sep in all_value[0]:
                    base = all_value[0].split(sep)[0]
                    match_index = [i for i, tok in enumerate(token) if tok.startswith(base)]
                    if match_index:
                        break
                    
            if not match_index:
                print("找不到匹配的开头 token")
                continue

        arg_token_index = match_index[0]
        if len(match_index) > 1:
            new_match_index = []
            for match_i in match_index:
                if all(token[match_i + tmp_i] in value for tmp_i in range(len(all_value)) if match_i + tmp_i < len(token)):
                    new_match_index.append(match_i)
            if new_match_index:
                arg_token_index = min(new_match_index, key=lambda i: abs(len(''.join(token[:i])) - start_index))

        # 处理结束 token
        match_index = [i for i, tok in enumerate(token) if tok == all_value[-1] and i >= arg_token_index]

        if not match_index:
            for sep in ["n't", "'", ',', '?', '...', '.']:
                if sep in all_value[-1]:
                    parts = [all_value[-1][sep_i:] for sep_i in range(all_value[-1].find(sep), len(all_value[-1]))]
                    for part in parts:
                        match_index = [i for i, tok in enumerate(token) if tok == part and i >= arg_token_index]
                        if match_index:
                            break
                    if match_index:
                        break
            if not match_index:
                print("找不到匹配的结束 token")
                continue

        arg_token_index_end = match_index[0]
        if len(match_index) > 1:
            new_match_index = []
            for match_i in match_index:
                if all(token[tmp_i] in value for tmp_i in range(arg_token_index, match_i + 1)):
                    new_match_index.append(match_i)
            if new_match_index:
                arg_token_index_end = min(new_match_index, key=lambda i: abs(len(''.join(token[:i])) - (end_index - len(all_value[-1]))))

        value_reconstructed = ' '.join(token[arg_token_index: arg_token_index_end + 1])
        arg = {
            'value': value_reconstructed,
            'role': role,
            'position': [arg_token_index + 1, arg_token_index_end + 1]
        }
        new_args.append(arg)

    return new_args

if __name__ == '__main__':
    data_path = ''
    save_path = ''

    with open(data_path, 'r', encoding='utf-8') as f, open(save_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            for text, srl in data.items():
                token = text.split()
                # print("token", token)
                new_srl = []
                for pred in srl:

                    argument_str = pred['arguments']
                    new_args = extract_arguments_from_string(argument_str, token)
                    pred['arguments'] =new_args
                    new_srl.append(pred)
                fout.write(json.dumps({text:new_srl},ensure_ascii=False)+'\n')
                