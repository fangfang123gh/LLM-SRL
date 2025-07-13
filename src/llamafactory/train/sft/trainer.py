import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import numpy as np
import torch
from transformers import Seq2SeqTrainer,GenerationConfig
import re
from torch import nn
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,

)
import warnings
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")
import logging

# 设置 transformers 的日志级别为 ERROR（忽略 warning）
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

# from accelerate.utils import DistributedType
# from transformers.training_args import OptimizerNames

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


pred_pattern = r"@@(.*?)##"
rl_pattern = r"<([^<>]+)>([^<>]+)</\1>"
from collections import Counter

# 只看三元组的 不看位置信息
# 看位置信息的话 直接相同就可以了吧
def is_right(response, ground_truth):
    pred_matches = re.finditer(pred_pattern, response)
    all_triplets = []
    for match in pred_matches:
        pred_word = match.group(1).strip()
        
        matches = re.finditer(rl_pattern, response)

        for match_r in matches:
            role = match_r.group(1)
            value = match_r.group(2)
            all_triplets.append((pred_word, value, role))

    all_gold_triplets = []
    pred_matches = re.finditer(pred_pattern, ground_truth)
    for match in pred_matches:
        pred_word = match.group(1).strip()
        
        matches = re.finditer(rl_pattern, ground_truth)

        for match_r in matches:
            role = match_r.group(1)
            value = match_r.group(2)
            all_gold_triplets.append((pred_word, value, role))
    
    return Counter(all_triplets) == Counter(all_gold_triplets)




def intervals_intersect(a, b, c, d):
    return max(a, c) <= min(b, d)

# 如果中文也采用分词的话 跟这个处理方式是一致的
def process_pred_en(response, token):
    pred_pattern = r"@@(.*?)##"
    matches1 = re.finditer(pred_pattern, response)
    start_pos = None
    count = 0
    all_pred = []
    
    count = 0
    for match in matches1:
        word = match.group(1).strip()
        start_pos = match.start()
        # end_pos = match.end() - 2
        temp_count = count
        count += 4
        blank_space = 0
        for tok in response[:start_pos]:
            if tok == ' ':
                blank_space += 1
                
            
        start_pos = start_pos - temp_count - blank_space

        match_index = [i for i, tok in enumerate(token) if tok.lower() == word.lower()]  

        if len(match_index) == 1:
            token_index = match_index[0]
        elif len(match_index) == 0:
            return []
        else:
            # 选一个最近的
            token_index = match_index[0]
            min_dis = abs(len(''.join(token[:match_index[0]])) - start_pos)
            for match_i in match_index:
                temp_dis = abs(len(''.join(token[:match_i])) - start_pos)
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    token_index = match_i
        pred_arg = {'pred': word, 'position': [token_index + 1, token_index + 1]}
        all_pred.append(pred_arg)
    return all_pred

def select_closest_match(match_index, all_value, token, index_ref):
    if len(match_index) == 1:
        return match_index[0]
    
    valid = []
    for match_i in match_index:
        if all(i + match_i < len(token) and token[i + match_i] in all_value for i in range(len(all_value))):
            valid.append(match_i)

    if not valid:
        return match_index[0]  # fallback

    best_i = valid[0]
    min_dis = abs(len(''.join(token[:best_i])) - index_ref)
    for i in valid:
        dis = abs(len(''.join(token[:i])) - index_ref)
        if dis < min_dis:
            min_dis = dis
            best_i = i
    return best_i

def process_arg_en(response, token):
    pred_pattern = r"@@(.*?)##"
    pattern = r"<([^<>]+)>([^<>]+)</\1>"
    all_pred = process_pred_en(response, token)
    if len(all_pred) != 1:
        return {}
    pred_arg = all_pred[0]

    matches1 = re.finditer(pred_pattern, response)
    start_pos = None
    for match in matches1:
        start_pos = match.start()
    
    new_args = []
    matches1 = list(re.finditer(pred_pattern, response))
    start_pos = matches1[-1].start() if matches1 else None

    matches = list(re.finditer(pattern, response))
    arg_count = 0

    for match in matches:
        role = match.group(1)
        value = match.group(2)
        start_index = match.start()
        end_index = match.end()

        blank_space = response[:start_index].count(' ')
        blank_space_end = response[:end_index].count(' ')

        temp_count = arg_count
        arg_count += 5 + len(role) * 2  # 处理 role 标签影响

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
                    break
            if not match_index:
                continue  # 无法匹配则跳过

        arg_token_index = select_closest_match(match_index, all_value, token, start_index)

        # 处理结束索引
        match_index = [i for i, tok in enumerate(token) if tok == all_value[-1] and i >= arg_token_index]

        if not match_index:
            for sep in ["n't", "'", ',', '.', '?', '...']:
                if sep in all_value[-1]:
                    parts = all_value[-1].split(sep)
                    for part in reversed(parts):
                        match_index = [i for i, tok in enumerate(token) if tok == part and i >= arg_token_index]
                        if match_index:
                            break
            if not match_index:
                continue

        arg_token_index_end = select_closest_match(match_index, list(reversed(all_value)), token, end_index - len(all_value[-1]))

        arg = {
            'value': ' '.join(token[arg_token_index: arg_token_index_end + 1]),
            'role': role,
            'position': [arg_token_index + 1, arg_token_index_end + 1]
        }

        new_args.append(arg)
    pred_arg['arguments'] = new_args
    return pred_arg

def find_reason_en(pred, gold, token):
    pred = pred.strip()
    gold = gold.strip()

    if len(pred) == 0:
        return 'The generated text is inconsistent with the original text: the generated text is empty while the original text is not.\n'

    if not compare_char_frequencies(gold, pred):
        reason = 'The generated text is inconsistent with the original text.\n'

        pred_pred = process_arg_en(pred, token)
        gold_pred = process_arg_en(gold, token)
        if gold_pred == pred_pred:
            return reason + "The generated text is incorrect, but the argument labeling results are correct.\n"

        if len(pred_pred) == 0 and len(gold_pred) == 0:
            return reason + "The generated text is incorrect, but the argument labeing results are correct.\n"
        elif len(pred_pred) == 0 or gold_pred.get('pred') is None:
            return reason + 'The predicate in the text was not correctly identified.\n'

        if gold_pred['pred'] != pred_pred['pred'] or gold_pred['position'] != pred_pred['position']:
            return reason + 'The predicate in the text was not correctly identified.\n'

        new_reason = ''
        all_role = len(re.findall(r"<[^>/]+>", pred))
        if all_role != len(re.findall(r"</[^>]+>", pred)):
            new_reason += 'Mismatched opening and closing role tags.\n'
        if all_role != len(pred_pred['arguments']):
            new_reason += 'Mismatch between number of role tags and extracted arguments.\n'

        index = []
        for arg in pred_pred['arguments']:
            found = False
            for i, gold_arg in enumerate(gold_pred['arguments']):
                if arg['value'] == gold_arg['value'] and arg['role'] == gold_arg['role']:
                    found = True
                    if i not in index:
                        index.append(i)
                    break
            if not found:
                if 'Incorrectly identified argument(s).\n' not in new_reason:
                    new_reason += 'Incorrectly identified argument(s).\n'

        if len(index) < len(gold_pred['arguments']) and 'Missing argument(s).\n' not in new_reason:
            new_reason += 'Missing argument(s).\n'

        return reason + new_reason

    if pred == gold:
        return 'Stop checking.'

    pred_pred = process_arg_en(pred, token)
    gold_pred = process_arg_en(gold, token)
    if gold_pred == pred_pred:
        return "Stop checking."

    if len(pred_pred) == 0 or gold_pred.get('pred') is None:
        return 'The predicate in the text was not correctly identified.\n'

    if gold_pred['pred'] != pred_pred['pred'] or gold_pred['position'] != pred_pred['position']:
        return 'The predicate in the text was not correctly identified.\n'

    reason = ''
    all_role = len(re.findall(r"<[^>/]+>", pred))
    if all_role != len(re.findall(r"</[^>]+>", pred)):
        reason += 'Mismatched opening and closing role tags.\n'
    if all_role != len(pred_pred['arguments']):
        reason += 'Mismatch between number of role tags and extracted arguments.\n'

    index = []
    temp_reason = None
    for arg in pred_pred['arguments']:
        found = False
        appeared = False
        for i, gold_arg in enumerate(gold_pred['arguments']):
            if arg['value'] == gold_arg['value']:
                appeared = True
                if arg['position'] == gold_arg['position'] and arg['role'] == gold_arg['role']:
                    found = True
                    if i not in index:
                        index.append(i)
                    break
            elif arg['role'] == gold_arg['role']:
                if intervals_intersect(arg['position'][0], arg['position'][1], gold_arg['position'][0], gold_arg['position'][1]):
                    temp_reason = 'Incorrect argument span.\n'

        if not found:
            if temp_reason is not None and temp_reason not in reason:
                reason += temp_reason
            if appeared:
                if 'The role label does not match the relation between predicate and argument.\n' not in reason:
                    reason += 'The role label does not match the relation between predicate and argument.\n'
            else:
                if 'Incorrectly identified argument(s).\n' not in reason:
                    reason += 'Incorrectly identified argument(s).\n'

    if len(index) < len(gold_pred['arguments']) and 'Missing argument(s).\n' not in reason:
        reason += 'Missing argument(s).\n'

    return reason





def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or \
           '\u3400' <= char <= '\u4dbf' or \
           '\u20000' <= char <= '\u2a6df' or \
           '\u3000' <= char <= '\u303f':
            return True
    return False

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

def clean_text(text: str) -> str:
    # 移除指定的符号、标签和空格
    text = re.sub(r'[#@]', '', text)  # 移除 '#' 和 '@'
    text = re.sub(r'</?[^>]+>', '', text)  # 移除所有形如 <...> 和 </...> 的标签
    text = re.sub(r'\s+', '', text)  # 移除所有空格
    return text

def compare_char_frequencies(gold: str, pred: str):
    gold = clean_text(gold)
    pred = clean_text(pred)
    
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    
    if gold_count == pred_count:
        return True
    else:
        # 找出gold中有但pred中没有的字符
        # gold_only = gold_count - pred_count
        
        # # 找出pred中有但gold中没有的字符
        # pred_only = pred_count - gold_count
        # def format_counter(counter):
        #     return ''.join([f"'{char}'({count}) " for char, count in counter.items()])

        # s = ''
        # if pred_only:
        #     s += "生成的文本中多了以下字符：" + format_counter(pred_only) + '\n'
        # if gold_only:
        #     s += "生成的文本中少了以下字符：" + format_counter(gold_only) + '\n'

        
        return False
    

def find_reason_pred_en(pred, gold, token):
    pred = pred.strip()
    gold = gold.strip()

    if len(pred) == 0:
        return 'The generated text is inconsistent with the original text: the generated text is empty while the original text is not.\n'

    if pred == gold:
        return 'Stop checking.'

    if not compare_char_frequencies(gold, pred):
        reason = 'The generated text is inconsistent with the original text.\n'

        pred_pred = process_pred_en(pred, token)
        gold_pred = process_pred_en(gold, token)

        new_reason = ''
        if pred.count('##') != pred.count('@@'):
            new_reason += 'Incorrect output format.\n'

        index = []
        for arg in pred_pred:
            find_arg = False
            for i, gold_arg in enumerate(gold_pred):
                if arg['pred'] == gold_arg['pred']:
                    if i not in index:
                        index.append(i)
                    find_arg = True
                    break
            if not find_arg and 'Incorrectly predicted predicate.\n' not in new_reason:
                new_reason += 'Incorrectly predicted predicate.\n'

        if len(index) != len(gold_pred) and 'Missing predicted predicate.\n' not in new_reason:
            new_reason += 'Missing predicted predicate.\n'

        if len(new_reason) > 0:
            reason += f'The following errors exist: {new_reason}'
        else:
            reason += 'The generated text is incorrect, but the predicate recognition is correct.\n'
        return reason

    pred_pred = process_pred_en(pred, token)
    gold_pred = process_pred_en(gold, token)

    reason = ''
    if pred.count('##') != pred.count('@@'):
        reason += 'Incorrect output format.\n'

    index = []
    for arg in pred_pred:
        find_arg = False
        for i, gold_arg in enumerate(gold_pred):
            if arg['pred'] == gold_arg['pred'] and arg['position'] == gold_arg['position']:
                if i not in index:
                    index.append(i)
                find_arg = True
                break
        if not find_arg and 'Incorrectly predicted predicate.\n' not in reason:
            reason += 'Incorrectly predicted predicate.\n'

    if len(index) != len(gold_pred) and 'Missing predicted predicate.\n' not in reason:
        reason += 'Missing predicted predicate.\n'

    return reason


'''
| 错误类型               | 统一后的表达                                                                         |
| ------------------ | ------------------------------------------------------------------------------ |
| 输出格式错误             | `Incorrect output format.\n`                                                   |
| 生成文本与原始文本不一致       | `The generated text is inconsistent with the original text.\n`                 |
| 谓词识别为空             | `The predicate in the text was not correctly identified.\n`                    |
| 谓词识别错误（匹配不上）       | `Incorrectly predicted predicate.\n`                                           |
| 缺失谓词               | `Missing predicted predicate.\n`                                               |
| 谓词识别正确，但文本不一致      | `The generated text is incorrect, but the predicate recognition is correct.\n` |
| 谓词识别正确，但文本不一致（含论元） | `The generated text is incorrect, but the argument annotations are correct.\n` |
| 标签数量与角色数量不一致       | `Mismatch between number of role tags and extracted arguments.\n`              |
| 标签闭合错误             | `Mismatched opening and closing role tags.\n`                                  |
| 错误的论元              | `Incorrectly identified argument(s).\n`                                        |
| 缺失论元               | `Missing argument(s).\n`                                                       |
| 角色标签不匹配            | `The role label does not match the relation between predicate and argument.\n` |
| 论元位置不一致            | `Incorrect argument span.\n`                                                   |
| 完全正确               | `Stop checking.`                                                               |
'''


def find_reason_pred(pred, gold):
    pred = pred.strip()
    gold = gold.strip()
    if len(pred) == 0:
        return '所生成的文本与原始文本不一致：生产文本为空，而原文本不为空\n'

    if pred == gold:
        return '停止检查'

    if not compare_char_frequencies(gold, pred):
        reason = f'所生成的文本与原始文本不一致\n'

        pred_pred = process_pred(pred)
        gold_pred = process_pred(gold)

        # 不看位置
        new_reason = ''
        if pred.count('##') != pred.count('@@'):
            new_reason += '输出格式不正确\n'
        index = []
        for arg in pred_pred:
            find_arg = False
            
            for i,gold_arg in enumerate(gold_pred):
                if arg['pred'] == gold_arg['pred']:
                        if i not in index:
                            index.append(i)
                        find_arg = True
                        break
            
            if '有错误识别的谓词\n' not in new_reason and not find_arg:
                new_reason += '有错误识别的谓词\n'
        
        # 文本与原始文本一致才有这些错误原因的提出 所以位置信息应该是需要考虑的
        if len(index) != len(gold_pred) and '有未识别的谓词\n' not in new_reason:
            new_reason += '有未识别的谓词\n'
        if len(new_reason) != 0:
            reason += f'在当前文本下，还存在以下错误：{new_reason}'
        else:
            reason += f'生成文本不正确，但是谓词识别结果正确\n'
        return reason
    

    pred_pred = process_pred(pred)
    gold_pred = process_pred(gold)
    reason = ''

    if pred.count('##') != pred.count('@@'):
        reason += '输出格式不正确\n'
    index = []
    for arg in pred_pred:
        find_arg = False
        
        for i,gold_arg in enumerate(gold_pred):
            if arg['pred'] == gold_arg['pred']:
                if arg['position'] == gold_arg['position']:
                    if i not in index:
                        index.append(i)
                    find_arg = True
                    break
        
        if '有错误识别的谓词\n' not in reason and not find_arg:
            reason += '有错误识别的谓词\n'

    if len(index) != len(gold_pred) and '有未识别的谓词\n' not in reason:
        reason += '有未识别的谓词\n'
    return reason

def intervals_intersect(a, b, c, d):
    return max(a, c) <= min(b, d)


def find_reason(pred, gold):
    pred = pred.strip()
    gold = gold.strip()
    if len(pred) == 0:
        return '所生成的文本与原始文本不一致：生产文本为空，而原文本不为空\n'

    if not compare_char_frequencies(gold, pred):
        reason = f'所生成的文本与原始文本不一致\n'

        pred_pred = process_arg(pred)
        gold_pred = process_arg(gold)
        if gold_pred == pred_pred:
            return reason + "生成文本不正确，但是论元标注结果正确\n"

        if len(pred_pred) == 0 and len(gold_pred) == 0:
            return reason + "生成文本不正确，但是论元标注结果正确\n"
        elif len(pred_pred) == 0:
            return reason+'问题文本中的谓词未正确包括\n'
        elif len(gold_pred) == 0:
            return reason + '问题文本中的谓词未正确包括\n'
        # print("gold_pred", gold_pred)
        # print("pred_pred", pred_pred)
        if gold_pred['pred'] != pred_pred['pred']:
            return reason +'问题文本中的谓词未正确包括\n'
        if gold_pred['position'] != pred_pred['position']:
            return reason +'问题文本中的谓词未正确包括\n'
        
        new_reason = ''
        # 识别出计算evaluate的时候的格式、text
        all_role= len(re.findall(r"<[^>/]+>", pred))
        if  all_role != len(re.findall(r"</[^>]+>", pred)):
            new_reason += '输出格式错误\n'
        if all_role != len(pred_pred['arguments']):
            new_reason += '闭合的标签角色不一致\n'
        index = []
        for arg in pred_pred['arguments']:
            find_arg = False
           
            for i,gold_arg in enumerate(gold_pred['arguments']):
                if arg['value'] == gold_arg['value']:
                    if arg['role'] == gold_arg['role']:
                        find_arg = True
                        if i not in index:
                            index.append(i)
                        break 
            if not find_arg:
                    if  '有错误识别的论元\n' not in new_reason:
                        new_reason += '有错误识别的论元\n'

        if len(index) < len(gold_pred['arguments']) and '有未识别的论元\n' not in new_reason:
            new_reason += '有未识别的论元\n'
        return reason + new_reason
    if pred == gold:
        return '停止检查'
    

    pred_pred = process_arg(pred)
    gold_pred = process_arg(gold)
    if gold_pred == pred_pred:
        return "停止检查"
    
    
    if len(pred_pred) == 0 and len(gold_pred) == 0:
        return '停止检查'
    elif len(pred_pred) == 0:
        return '问题文本中的谓词未正确包括\n'
    elif len(gold_pred) == 0:
        return '问题文本中的谓词未正确包括\n'
    
    if gold_pred['pred'] != pred_pred['pred']:
        return '问题文本中的谓词未正确包括\n'
    if gold_pred['position'] != pred_pred['position']:
        return '问题文本中的谓词未正确包括\n'
    
    reason = ''
    # 识别出计算evaluate的时候的格式、text
    all_role= len(re.findall(r"<[^>/]+>", pred))
    if  all_role != len(re.findall(r"</[^>]+>", pred)):
        reason += '输出格式错误\n'
    new_reason = None
    if all_role != len(pred_pred['arguments']):
        reason += '闭合的标签角色不一致\n'
    index = []
    for arg in pred_pred['arguments']:
        find_arg = False
        appear = False
        
        for i,gold_arg in enumerate(gold_pred['arguments']):
            if arg['value'] == gold_arg['value']:
                appear = True
                if arg['position'] == gold_arg['position']:
                    if arg['role'] == gold_arg['role']:
                        find_arg = True
                        if i not in index:
                            index.append(i)
                        break
            elif arg['role'] == gold_arg['role']:
                if intervals_intersect(arg['position'][0],arg['position'][1], gold_arg['position'][0],gold_arg['position'][1]):
                    new_reason = '论元跨度不正确\n'

        
        if not find_arg:
            if new_reason is not None:
                if new_reason not in reason:
                    reason  += new_reason
            if appear:
                if '论元标签含义与谓词和该论元的关系不一致\n' not in reason:
                    reason += '论元标签含义与谓词和该论元的关系不一致\n'
            else:
                if  '有错误识别的论元\n' not in reason:
                    reason += '有错误识别的论元\n'

    if len(index) < len(gold_pred['arguments']) and '有未识别的论元\n' not in reason:
        reason += '有未识别的论元\n'
    return reason
                    

def extract_issue_segment(text):
    """
    提取输出中的“存在问题”部分的填充内容（不包含前缀“存在问题：”）
    如果没有匹配，返回空字符串
    """
    match = re.search(r"存在问题：([^\n]*)", text)
    return match.group(1).strip() if match else ""

def extract_predicate_result(text):
    """
    提取输出中的“谓词识别结果”部分的内容（不包含前缀“谓词识别结果：”）
    如果没有匹配，返回空字符串
    """
    match = re.search(r"谓词识别结果：([^\n]*)", text)
    return match.group(1).strip() if match else ""

def extract_argument_result(text):
    """
    提取输出中的“谓词识别结果”部分的内容（不包含前缀“谓词识别结果：”）
    如果没有匹配，返回空字符串
    """
    match = re.search(r"论元识别结果：([^\n]*)", text)
    return match.group(1).strip() if match else ""

def extract_issue_segment_en(text):
    """
    Extract the content of the 'Issue' section from the output (excluding the prefix 'Issue:').
    Returns an empty string if no match is found.
    """
    match = re.search(r"Issue:([^\n]*)", text)
    return match.group(1).strip() if match else ""

def extract_predicate_result_en(text):
    """
    Extract the content of the 'Predicate Recognition Result' section from the output
    (excluding the prefix 'Predicate Recognition Result:').
    Returns an empty string if no match is found.
    """
    match = re.search(r"Predicate Recognition Result:([^\n]*)", text)
    return match.group(1).strip() if match else ""

def extract_argument_result_en(text):
    """
    Extract the content of the 'Argument Recognition Result' section from the output
    (excluding the prefix 'Argument Recognition Result:').
    Returns an empty string if no match is found.
    """
    match = re.search(r"Argument Recognition Result:([^\n]*)", text)
    return match.group(1).strip() if match else ""


import re
def extract_pred_text(text):
    match = re.search(r'^Text:\s*(.*)', text, re.MULTILINE)

    return match.group(1).strip() if match else ""

def extract_pred_text_zh(text):
    match = re.search(r'^分词后的文本为：\s*(.*)', text, re.MULTILINE)

    return match.group(1).strip() if match else ""


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", data_args: "DataArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.data_args = data_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # for key, _ in inputs.items():
        #     print(key)
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # print(self.state.global_step)
        if self.finetuning_args.correct_step != -1 and self.state.global_step >= self.finetuning_args.correct_step:
            max_step = 3
            step = 0
            is_not_cal = False
            max_length = self.data_args.cutoff_len
            is_terminate = False
            attention_mask = inputs['attention_mask']
            label_ids = inputs['labels']
            input_ids = inputs['input_ids']
            loss = 0.0
            gold_output = None
            text = None
            
            if self.data_args.template == 'qwen':
                assistant_presuffix = '<|im_start|>assistant\n'
                user_presuffix = '<|im_start|>user\n'
                user_suffix = '<|im_end|>\n'
                signal = '<|im_start|>'
                lang = 'zh'
            elif self.data_args.template == 'llama3':
               
                assistant_presuffix = '<|start_header_id|>assistant<|end_header_id|>\n'
                user_presuffix = '<|start_header_id|>user<|end_header_id|>\n'
                user_suffix = '<|eot_id|>\n'
                signal = '<|start_header_id|>'
                lang = 'en'
            with torch.no_grad():
                
                batch_labels = []
                start = -1
                label = inputs['labels'][0]
                for i,l in enumerate(label):
                    l = l.item()
                    if l == -100 and start != -1:
            
                        batch_labels.append(label[start: i])
                        start = -1
                    elif l != -100 and start == -1:
                        start = i
                if start != -1:
                    batch_labels.append(label[start: i])
            if lang == 'zh':
                
                if self.finetuning_args.pred_correct:
                    if len(batch_labels) > 2:
                        ground_truth = self.tokenizer.decode(batch_labels[1], skip_special_tokens=True)
                        # print("ground_truth", ground_truth)
                        ground_truth = ground_truth.strip()
                        
                    else:
                        is_terminate = True
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                    
                        new_input_ids = input_ids[:, :possible_starts[5]]
                        new_attention_mask = attention_mask[:, :possible_starts[5]]
                        new_labels_ids = label_ids[:,:possible_starts[5]]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                        loss = outputs['loss']
                    
                    if not is_terminate:
                        
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                        new_input_ids = input_ids[:, :possible_starts[5]]
                        new_attention_mask = attention_mask[:, :possible_starts[5]]
                        new_labels_ids = label_ids[:,:possible_starts[5]]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        with torch.no_grad():
                            # 因为还要纠错 所以是还不进行forward的
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

                            text = self.tokenizer.decode(input_ids[:, possible_starts[3]:possible_starts[4]][0], skip_special_tokens=True)
                            # print("text", text)
                            text = extract_pred_text_zh(text)
                            logits = torch.argmax(outputs.logits, dim=-1)
                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                            # output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=False)
                            output_result_ori = output_result_ori[0]
                            assistant_idx = output_result_ori.lower().rfind("assistant")
                            user_idx = output_result_ori.lower().rfind("user")
                            # print("output_result_ori", output_result_ori)
                            if assistant_idx > user_idx:
                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                # print("提取 assistant 后的文本：\n", output_result)
                            else:
                                output_result = output_result_ori[user_idx + len("user"):]
                                # print("提取 user 后的文本：\n", output_result)
                            output_result = output_result.strip()
                            # if  '我已经理解' in output_result:
                            #     output_result_processed = output_result_ori[0].lower().split('user')
                            #     output_result = output_result_processed[-1]
                            if  '我已经理解' in output_result:
                                is_not_cal = True
                            if len(output_result) != 0:
                                if output_result[0] == ':':
                                    output_result = output_result[1:]
                            else:
                                is_not_cal = True
                            output_result = output_result.strip()
                            # if len(output_result) > 3*len(text):
                            #     # 如果很长的话 肯定不是好的回答
                            #     is_not_cal = True
                            if '语义角色标注' in output_result or '它的意思' in output_result:
                                is_not_cal = True
                        if is_not_cal or new_input_ids.shape[-1] > max_length:
                            # 这时候才需要进行forwrd
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            loss += outputs['loss']
                        else:
                        
                            another_prompt=None
                            argument_result = output_result
                            if argument_result in ground_truth:
                                # print("学会停止检查")
                                # print(pre_output_res)
                                is_terminate = True 
                                gold_output = '停止检查'
                            else:
                                gold_output = find_reason_pred(argument_result.lower(), ground_truth.lower())
                            if '所生成的文本与原始文本不一致' in gold_output:
                                gold_output += f'原始文本为： {text}\n'
                            if gold_output != '停止检查' or '停止检查'not in gold_output:
                                gold_output = '存在问题：'+gold_output+'\n'
                                gold_output += '谓词识别结果：\n'+ ground_truth
                            # else:
                            gold_output +=user_suffix
                            prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''


                            prompt = prompt.replace('<<text>>', text)
                            prompt = prompt.replace('<<output>>', argument_result)
                        
                            
                            if another_prompt is not None:
                                prompt = another_prompt + prompt
                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = extra_label_ids.unsqueeze(0)
                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)
            
                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                            # 还不进行forward计算
                            with torch.no_grad():
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            
                            # loss += outputs['loss']

                        
                            if not is_terminate:
                            
                                # 循环第二次
                                is_break = False
                                is_terminate = False
                                with torch.no_grad():
                                    logits = torch.argmax(outputs.logits, dim=-1)
                                    output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                    output_result_ori = output_result_ori[0]
                                    assistant_idx = output_result_ori.lower().rfind("assistant")
                                    user_idx = output_result_ori.lower().rfind("user")

                                    if assistant_idx > user_idx:
                                        output_result = output_result_ori[assistant_idx + len("assistant"):]
                                        # print("提取 assistant 后的文本：\n", output_result)
                                    else:
                                        output_result = output_result_ori[user_idx + len("user"):]
                                        # print("提取 user 后的文本：\n", output_result)
                                    output_result = output_result.strip()
                                    # print("new output_result", output_result)

                                # if len(output_result) > 3*len(text):
                                #     # 如果很长的话 肯定不是好的回答
                                #     is_break = True
                                # print("output_result", output_result)
                                if '语义角色标注' in output_result or '它的意思' in output_result:
                                    is_break = True

                                if '停止检查' in output_result or '停止存在问题检查'in output_result or ('停止' in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                    is_break = True
                                elif '谓词识别结果：' in output_result:
                                    argument_result = output_result[output_result.rfind('谓词识别结果：')+len('谓词识别结果：'):]
                                else:
                                    is_break = True
                                


                                if not is_break:
                                    argument_result = argument_result.strip()
                                    # print("argument_result", argument_result)
                                    if argument_result in ground_truth:
                                        is_terminate = True 
                                        gold_output = '停止检查'
                                    else:
                                        gold_output = find_reason_pred(argument_result.lower(), ground_truth.lower())
                                    
                                    if '所生成的文本与原始文本不一致' in gold_output:
                                        gold_output += f'原始文本为： {text}\n'

                                    if gold_output != '停止检查' or '停止检查'not in gold_output:
                                        gold_output = '存在问题：'+gold_output+'\n'
                                        gold_output += '谓词识别结果：\n'+ ground_truth
                                    # else:
                                    gold_output +=user_suffix
                                    prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''
                                    # print("2222222222222")

                                    prompt = prompt.replace('<<text>>', text)
                                    prompt = prompt.replace('<<output>>', argument_result)
                                
                                    
                                    if another_prompt is not None:
                                        prompt = another_prompt + prompt
                                    extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                    extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = extra_label_ids.unsqueeze(0)
                                    terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                    extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                    extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                    extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                    extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                    # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                    new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                    new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                    new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                    with torch.no_grad():

                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    
                                    # loss += outputs['loss']
                                    is_break =False
                                    if not is_terminate:
                                        is_terminate = False
                                        with torch.no_grad():
                                            logits = torch.argmax(outputs.logits, dim=-1)
                                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                            output_result_ori = output_result_ori[0]
                                            assistant_idx = output_result_ori.lower().rfind("assistant")
                                            user_idx = output_result_ori.lower().rfind("user")

                                            if assistant_idx > user_idx:
                                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                                # print("提取 assistant 后的文本：\n", output_result)
                                            else:
                                                output_result = output_result_ori[user_idx + len("user"):]
                                                # print("提取 user 后的文本：\n", output_result)
                                            output_result = output_result.strip()
                                            # print("new output_result", output_result)

                    
                                        # if len(output_result) > 3*len(text):
                                        #     # 如果很长的话 肯定不是好的回答
                                        #     is_break = True
                                        if '语义角色标注' in output_result or '它的意思' in output_result:
                                            is_break = True
                                        if '停止检查' in output_result or '停止存在问题检查'in output_result or ('停止' in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                            is_break = True
                                        elif '谓词识别结果：' in output_result:
                                            argument_result = output_result[output_result.rfind('谓词识别结果：')+len('谓词识别结果：'):]
                                        else:
                                            is_break = True


                                        if not is_break:
                                            argument_result = argument_result.strip()
                                            # print("argument_result", argument_result)
                                            if argument_result in ground_truth:
                                                is_terminate = True 
                                                gold_output = '停止检查'
                                            else:
                                                gold_output = find_reason_pred(argument_result.lower(), ground_truth.lower())

                                            if '所生成的文本与原始文本不一致' in gold_output:
                                                gold_output += f'原始文本为： {text}\n'

                                            if gold_output != '停止检查' or '停止检查'not in gold_output:
                                                gold_output = '存在问题：'+gold_output+'\n'
                                                gold_output += '谓词识别结果：\n'+ ground_truth
                                            # else:
                                            gold_output +=user_suffix
                                            prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''


                                            prompt = prompt.replace('<<text>>', text)
                                            prompt = prompt.replace('<<output>>', argument_result)
                                        
                                            
                                            if another_prompt is not None:
                                                prompt = another_prompt + prompt
                                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = extra_label_ids.unsqueeze(0)
                                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                            with torch.no_grad():
                                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            
                                            # loss += outputs['loss']
                                        else:
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            loss += outputs['loss']
                                    else:
                                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                        loss += outputs['loss']
                                else:
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    loss += outputs['loss']
                            else:
                                inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                loss += outputs['loss']
                else:
                    
                    signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                    possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                
                    new_input_ids = input_ids[:, :possible_starts[5]]
                    new_attention_mask = attention_mask[:, :possible_starts[5]]
                    new_labels_ids = label_ids[:,:possible_starts[5]]
                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    loss += outputs['loss']
            elif lang == 'en':
                if self.finetuning_args.pred_correct:
                    
                    if len(batch_labels) > 2:
                        ground_truth = self.tokenizer.decode(batch_labels[1], skip_special_tokens=True)
                        # print("ground_truth", ground_truth)
                        ground_truth = ground_truth.strip()
                        
                    else:
                        is_terminate = True
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                        loss = outputs['loss']
                    if not is_terminate:
                        
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                        
                        new_input_ids = input_ids[:, :possible_starts[5]]
                        new_attention_mask = attention_mask[:, :possible_starts[5]]
                        new_labels_ids = label_ids[:,:possible_starts[5]]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            text = self.tokenizer.decode(input_ids[:, possible_starts[3]:possible_starts[4]][0], skip_special_tokens=True)
                            # print("text", text)
                            text = extract_pred_text(text)
                            token = text.split()
                            logits = torch.argmax(outputs.logits, dim=-1)
                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                            # output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=False)
                            output_result_ori = output_result_ori[0]
                        
                            assistant_idx = output_result_ori.lower().rfind("assistant")
                            user_idx = output_result_ori.lower().rfind("user")

                            if assistant_idx > user_idx:
                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                # print("提取 assistant 后的文本：\n", output_result)
                            else:
                                output_result = output_result_ori[user_idx + len("user"):]
                                # print("提取 user 后的文本：\n", output_result)
                            output_result = output_result.strip()
                            # if  '我已经理解' in output_result:
                            #     output_result_processed = output_result_ori[0].lower().split('user')
                            #     output_result = output_result_processed[-1]
                            if  'i have understood' in output_result.lower():
                                is_not_cal = True
                            if len(output_result) != 0:
                                if output_result[0] == ':':
                                    output_result = output_result[1:]
                            else:
                                is_not_cal = True
                            output_result = output_result.strip()
                            # if len(output_result) > 3*len(text):
                            #     # 如果很长的话 肯定不是好的回答
                            #     is_not_cal = True
                            if 'semantic role labeling' in output_result.lower() or 'its interpretation is' in output_result.lower():
                                is_not_cal = True
                        if is_not_cal or new_input_ids.shape[-1] > max_length:
                            # print("output_result", output_result)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            loss += outputs['loss']
                            # 有点搞不懂while了 直接写三次把
                        else:
                            # 超过长度 就算再循环也没用
                            another_prompt=None
                            argument_result = output_result
                            if argument_result in ground_truth:
                                # print("学会停止检查")
                                # print(pre_output_res)
                                is_terminate = True 
                                gold_output = 'Stop checking.'
                            else:
                                gold_output = find_reason_pred_en(argument_result.lower(), ground_truth.lower(), token)
                            if 'The generated text is inconsistent with the original text' in gold_output:
                                gold_output += f'Original text: {text}\n'
                            if gold_output != 'Stop checking.' and 'Stop checking.' not in gold_output:
                                gold_output = 'Issue detected: ' + gold_output + '\n'
                                gold_output += 'Predicate recognition result:\n' + ground_truth
                            gold_output += user_suffix

                            
                            # prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。如果生成的文本与原文本不一致，只需要输出该错误，不需要判断其他错误。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''
                            prompt = '''Re-evaluate the generated predicate recognition result. Check whether the format of the predicted output is correct, whether each predicate is correctly identified, and whether any predicates are missing. Correct the output based on the identified issues. The format of the result should be consistent with the previous outputs. Use the format: "Issue detected: ...\nPredicate recognition result:\n". If no errors are found, output "Stop checking.".\nOriginal text: <<text>>\nGenerated output to review: <<output>>\nOutput:\n'''


                            prompt = prompt.replace('<<text>>', text)
                            prompt = prompt.replace('<<output>>', argument_result)
                        
                            
                            if another_prompt is not None:
                                prompt = another_prompt + prompt
                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = extra_label_ids.unsqueeze(0)
                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                            with torch.no_grad():
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            
                            # loss += outputs['loss']

                        
                            if not is_terminate:
                            
                                # 循环第二次
                                is_break = False
                                is_terminate = False
                                with torch.no_grad():
                                    logits = torch.argmax(outputs.logits, dim=-1)
                                    output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                    output_result_ori = output_result_ori[0]
                                    assistant_idx = output_result_ori.lower().rfind("assistant")
                                    user_idx = output_result_ori.lower().rfind("user")

                                    if assistant_idx > user_idx:
                                        output_result = output_result_ori[assistant_idx + len("assistant"):]
                                        # print("提取 assistant 后的文本：\n", output_result)
                                    else:
                                        output_result = output_result_ori[user_idx + len("user"):]
                                        # print("提取 user 后的文本：\n", output_result)
                                    output_result = output_result.strip()
                                    # print("new output_result", output_result)

                                # if len(output_result) > 3*len(text):
                                #     # 如果很长的话 肯定不是好的回答
                                #     is_break = True
                                if 'semantic role labeling' in output_result.lower() or 'its interpretation is' in output_result.lower():
                                    is_break = True

                                if 'stop checking.' in output_result.lower() or ('stop'.lower() in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                    is_break = True
                                elif 'predicate recognition result:' in output_result.lower():
                                    argument_result = output_result[output_result.rfind('predicate recognition result:')+len('predicate recognition result:'):]
                                else:
                                    is_break = True
                                # 严格按照格式输出
                                # elif '识别结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                # elif '结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                # else:
                                #     argument_result = output_result
                                


                                if not is_break:
                                    argument_result = argument_result.strip()
                                    # print("argument_result", argument_result)
                                    if argument_result in ground_truth:
                                        is_terminate = True 
                                        gold_output = 'Stop checking.'
                                    else:
                                        gold_output = find_reason_pred_en(argument_result.lower(), ground_truth.lower(), token)
                                    
                                    if 'The generated text is inconsistent with the original text' in gold_output:
                                        gold_output += f'Original text: {text}\n'
                                    if gold_output != 'Stop checking.' and 'Stop checking.' not in gold_output:
                                        gold_output = 'Issue detected: ' + gold_output + '\n'
                                        gold_output += 'Predicate recognition result:\n' + ground_truth
                                    gold_output += user_suffix

                                    
                                    # prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。如果生成的文本与原文本不一致，只需要输出该错误，不需要判断其他错误。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''
                                    prompt = '''Re-evaluate the generated predicate recognition result. Check whether the format of the predicted output is correct, whether each predicate is correctly identified, and whether any predicates are missing. Correct the output based on the identified issues. The format of the result should be consistent with the previous outputs. Use the format: "Issue detected: ...\nPredicate recognition result:\n". If no errors are found, output "Stop checking.".\nOriginal text: <<text>>\nGenerated output to review: <<output>>\nOutput:\n'''


                                    prompt = prompt.replace('<<text>>', text)
                                    prompt = prompt.replace('<<output>>', argument_result)
                                
                                    
                                    if another_prompt is not None:
                                        prompt = another_prompt + prompt
                                    extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                    extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = extra_label_ids.unsqueeze(0)
                                    terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                    extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                    extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                    extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                    extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                    # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                    new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                    new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                    new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                    with torch.no_grad():
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    
                                    # loss += outputs['loss']
                                    is_break =False
                                    if not is_terminate:
                                        is_terminate = False
                                        with torch.no_grad():
                                            logits = torch.argmax(outputs.logits, dim=-1)
                                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                            output_result_ori = output_result_ori[0]
                                            assistant_idx = output_result_ori.lower().rfind("assistant")
                                            user_idx = output_result_ori.lower().rfind("user")

                                            if assistant_idx > user_idx:
                                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                                # print("提取 assistant 后的文本：\n", output_result)
                                            else:
                                                output_result = output_result_ori[user_idx + len("user"):]
                                                # print("提取 user 后的文本：\n", output_result)
                                            output_result = output_result.strip()
                                            # print("new output_result", output_result)

                    
                                        # if len(output_result) > 3*len(text):
                                        #     # 如果很长的话 肯定不是好的回答
                                        #     is_break = True
                                        if 'semantic role labeling' in output_result.lower() or 'its interpretation is' in output_result.lower():
                                            is_break = True

                                        if 'stop checking.' in output_result.lower() or ('stop'.lower() in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                            is_break = True
                                        elif 'predicate recognition result:' in output_result.lower():
                                            argument_result = output_result[output_result.rfind('predicate recognition result:')+len('predicate recognition result:'):]
                                        else:
                                            is_break = True
                                        # elif '识别结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                        # elif '结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                        # else:
                                        #     argument_result = output_result
                                        


                                        if not is_break:
                                            argument_result = argument_result.strip()
                                            if argument_result in ground_truth:
                                                # print("学会停止检查")
                                                # print(pre_output_res)
                                                is_terminate = True 
                                                gold_output = 'Stop checking.'
                                            else:
                                                gold_output = find_reason_pred_en(argument_result.lower(), ground_truth.lower(), token)
                                            if 'The generated text is inconsistent with the original text' in gold_output:
                                                gold_output += f'Original text: {text}\n'
                                            if gold_output != 'Stop checking.' and 'Stop checking.' not in gold_output:
                                                gold_output = 'Issue detected: ' + gold_output + '\n'
                                                gold_output += 'Predicate recognition result:\n' + ground_truth
                                            gold_output += user_suffix

                                            
                                            # prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。如果生成的文本与原文本不一致，只需要输出该错误，不需要判断其他错误。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''
                                            prompt = '''Re-evaluate the generated predicate recognition result. Check whether the format of the predicted output is correct, whether each predicate is correctly identified, and whether any predicates are missing. Correct the output based on the identified issues. The format of the result should be consistent with the previous outputs. Use the format: "Issue detected: ...\nPredicate recognition result:\n". If no errors are found, output "Stop checking.".\nOriginal text: <<text>>\nGenerated output to review: <<output>>\nOutput:\n'''

                                            prompt = prompt.replace('<<text>>', text)
                                            prompt = prompt.replace('<<output>>', argument_result)
                                        
                                            
                                            if another_prompt is not None:
                                                prompt = another_prompt + prompt
                                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = extra_label_ids.unsqueeze(0)
                                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                            with torch.no_grad():
                                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            
                                            # loss += outputs['loss']
                                        else:
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            loss += outputs['loss']
                                    else:
                                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)   
                                        loss += outputs['loss']                         
                                else:
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    loss += outputs['loss']
                            else:
                                inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                loss += outputs['loss']
                else:
                    signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                    possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                
                    new_input_ids = input_ids[:, :possible_starts[5]]
                    new_attention_mask = attention_mask[:, :possible_starts[5]]
                    new_labels_ids = label_ids[:,:possible_starts[5]]
                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    loss += outputs['loss']
            if lang == 'zh':
                if self.finetuning_args.argument_correct:
                    is_terminate = False
                    is_not_cal = False
                    if len(batch_labels) >= 4:
                        ground_truth = self.tokenizer.decode(batch_labels[3], skip_special_tokens=True)
                        # print("ground_truth", ground_truth)
                        ground_truth = ground_truth.strip()
                        
                    else:
                        is_terminate = True
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                        
                        new_input_ids = input_ids[:, possible_starts[5]:]
                        new_attention_mask = attention_mask[:, possible_starts[5]:]
                        new_labels_ids = label_ids[:,possible_starts[5]:]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                        loss += outputs['loss']
                    
                    if not is_terminate:
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                        
                        new_input_ids = input_ids[:, possible_starts[5]:]
                        new_attention_mask = attention_mask[:, possible_starts[5]:]
                        new_labels_ids = label_ids[:,possible_starts[5]:]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            text = self.tokenizer.decode(input_ids[:, possible_starts[3]:possible_starts[4]][0], skip_special_tokens=True)
                            # print("text", text)
                            text = extract_pred_text_zh(text)

                            logits = torch.argmax(outputs.logits, dim=-1)
                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                            # output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=False)
                            output_result_ori = output_result_ori[0]
                            assistant_idx = output_result_ori.lower().rfind("assistant")
                            user_idx = output_result_ori.lower().rfind("user")

                            if assistant_idx > user_idx:
                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                # print("提取 assistant 后的文本：\n", output_result)
                            else:
                                output_result = output_result_ori[user_idx + len("user"):]
                                # print("提取 user 后的文本：\n", output_result)
                            output_result = output_result.strip()
                            # if  '我已经理解' in output_result:
                            #     output_result_processed = output_result_ori[0].lower().split('user')
                            #     output_result = output_result_processed[-1]
                            if  '我已经理解' in output_result:
                                is_not_cal = True
                            if len(output_result) != 0:
                                if output_result[0] == ':':
                                    output_result = output_result[1:]
                            else:
                                is_not_cal = True
                            output_result = output_result.strip()
                            # if len(output_result) > 3*len(text):
                            #     # 如果很长的话 肯定不是好的回答
                            #     is_not_cal = True
                            if '语义角色标注' in output_result or '核心论元' in output_result:
                                is_not_cal = True
                        if is_not_cal or inputs['input_ids'].shape[-1] > max_length:
                            # print("output_result", output_result)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            loss += outputs['loss']
                            # 有点搞不懂while了 直接写三次把
                        else:
                            # 超过长度 就算再循环也没用
                            another_prompt=None
                            argument_result = output_result
                            if argument_result in ground_truth:
                                # print("学会停止检查")
                                # print(pre_output_res)
                                is_terminate = True 
                                gold_output = '停止检查'
                            else:
                                gold_output = find_reason(argument_result.lower(), ground_truth.lower())
                            if '所生成的文本与原始文本不一致' in gold_output:
                                gold_output += f'原始文本为： {text}\n'
                            if gold_output != '停止检查' or '停止检查'not in gold_output:
                                gold_output = '存在问题：'+gold_output+'\n'
                                gold_output += '论元标注结果：\n'+ ground_truth
                            # else:
                            gold_output +=user_suffix
                            
                            # prompt ='''重新思考生成的谓词识别结果，判断输出谓词的格式是否正确、每一个谓词是否识别正确以及是否有忽略的谓词，结合检查错误进行纠正。如果生成的文本与原文本不一致，只需要输出该错误，不需要判断其他错误。谓词识别结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n谓词识别结果：\n"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n'''
            
                            prompt = "检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致；是否存在未识别的论元。输出该问题并进行纠正，论元标注结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n论元标注结果：\"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n"


                            prompt = prompt.replace('<<text>>', text)
                            prompt = prompt.replace('<<output>>', argument_result)
                        
                            
                            if another_prompt is not None:
                                prompt = another_prompt + prompt
                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = extra_label_ids.unsqueeze(0)
                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                            with torch.no_grad():
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            
                            # loss += outputs['loss']

                        
                            if not is_terminate:
                            
                                # 循环第二次
                                is_break = False
                                is_terminate = False
                                with torch.no_grad():
                                    logits = torch.argmax(outputs.logits, dim=-1)
                                    output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                    output_result_ori = output_result_ori[0]
                                    assistant_idx = output_result_ori.lower().rfind("assistant")
                                    user_idx = output_result_ori.lower().rfind("user")

                                    if assistant_idx > user_idx:
                                        output_result = output_result_ori[assistant_idx + len("assistant"):]
                                        # print("提取 assistant 后的文本：\n", output_result)
                                    else:
                                        output_result = output_result_ori[user_idx + len("user"):]
                                        # print("提取 user 后的文本：\n", output_result)
                                    output_result = output_result.strip()
                                    # print("new output_result", output_result)

                                # if len(output_result) > 3*len(text):
                                #     # 如果很长的话 肯定不是好的回答
                                #     is_break = True
                                if '语义角色标注' in output_result or '它的意思' in output_result:
                                    is_break = True

                                if '停止检查' in output_result or '停止存在问题检查'in output_result or ('停止' in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                    is_break = True
                                elif '论元标注结果：' in output_result:
                                    argument_result = output_result[output_result.rfind('论元标注结果：')+len('论元标注结果：'):]
                                else:
                                    is_break = True
                                # 严格按照格式输出
                                # elif '识别结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                # elif '结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                # else:
                                #     argument_result = output_result
                                


                                if not is_break:
                                    argument_result = argument_result.strip()
                                    # print("argument_result", argument_result)
                                    if argument_result in ground_truth:
                                        is_terminate = True 
                                        gold_output = '停止检查'
                                    else:
                                        gold_output = find_reason(argument_result.lower(), ground_truth.lower())
                                    
                                    if '所生成的文本与原始文本不一致' in gold_output:
                                        gold_output += f'原始文本为： {text}\n'

                                    if gold_output != '停止检查' or '停止检查'not in gold_output:
                                        gold_output = '存在问题：'+gold_output+'\n'
                                        gold_output += '论元标注结果：\n'+ ground_truth
                                    # else:
                                    gold_output +=user_suffix
                                    prompt = "检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致；是否存在未识别的论元。输出该问题并进行纠正，论元标注结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n论元标注结果：\"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n"


                                    prompt = prompt.replace('<<text>>', text)
                                    prompt = prompt.replace('<<output>>', argument_result)
                                
                                    
                                    if another_prompt is not None:
                                        prompt = another_prompt + prompt
                                    extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                    extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = extra_label_ids.unsqueeze(0)
                                    terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                    extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                    extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                    extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                    extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                    # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                    new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                    new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                    new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                    with torch.no_grad():
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    
                                    # loss += outputs['loss']
                                    is_break =False
                                    if not is_terminate:
                                        is_terminate = False
                                        with torch.no_grad():
                                            logits = torch.argmax(outputs.logits, dim=-1)
                                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                            output_result_ori = output_result_ori[0]
                                            assistant_idx = output_result_ori.lower().rfind("assistant")
                                            user_idx = output_result_ori.lower().rfind("user")

                                            if assistant_idx > user_idx:
                                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                                # print("提取 assistant 后的文本：\n", output_result)
                                            else:
                                                output_result = output_result_ori[user_idx + len("user"):]
                                                # print("提取 user 后的文本：\n", output_result)
                                            output_result = output_result.strip()
                                            # print("new output_result", output_result)

                    
                                        # if len(output_result) > 3*len(text):
                                        #     # 如果很长的话 肯定不是好的回答
                                        #     is_break = True
                                        if '语义角色标注' in output_result or '它的意思' in output_result:
                                            is_break = True
                                        if '停止检查' in output_result or '停止存在问题检查'in output_result or ('停止' in output_result and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                            is_break = True
                                        elif '论元标注结果：' in output_result:
                                            argument_result = output_result[output_result.rfind('论元标注结果：')+len('论元标注结果：'):]
                                        else:
                                            is_break = True
                                        # elif '识别结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                        # elif '结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                        # else:
                                        #     argument_result = output_result
                                        


                                        if not is_break:
                                            argument_result = argument_result.strip()
                                            # print("argument_result", argument_result)
                                            if argument_result in ground_truth:
                                                is_terminate = True 
                                                gold_output = '停止检查'
                                            else:
                                                gold_output = find_reason(argument_result.lower(), ground_truth.lower())

                                            if '所生成的文本与原始文本不一致' in gold_output:
                                                gold_output += f'原始文本为： {text}\n'

                                            if gold_output != '停止检查' or '停止检查'not in gold_output:
                                                gold_output = '存在问题：'+gold_output+'\n'
                                                gold_output += '论元标注结果：\n'+ ground_truth
                                            # else:
                                            gold_output +=user_suffix
                                            prompt = "检查生成的论元识别结果，如果存在以下问题：所生成的文本是否与原始文本一致；论元跨度是否正确；论元标签含义是否与谓词和该论元的关系保持一致；是否存在未识别的论元。输出该问题并进行纠正，论元标注结果的格式与之前保持一致，输出格式示例为：\"存在问题：...\n论元标注结果：\"。如果不存在错误，则输出停止检查。\n原文本: <<text>>\n需要思考的输出结果： <<output>>\n输出：\n"


                                            prompt = prompt.replace('<<text>>', text)
                                            prompt = prompt.replace('<<output>>', argument_result)
                                        
                                            
                                            if another_prompt is not None:
                                                prompt = another_prompt + prompt
                                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = extra_label_ids.unsqueeze(0)
                                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                            with torch.no_grad():
                                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            
                                            # loss += outputs['loss']
                                        else:
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            loss += outputs['loss']
                                    else:
                                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                        loss += outputs['loss']
                                else:
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    loss += outputs['loss']
                            else:
                                inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                loss += outputs['loss']
                else:
                    signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                    possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                    
                    new_input_ids = input_ids[:, possible_starts[5]:]
                    new_attention_mask = attention_mask[:, possible_starts[5]:]
                    new_labels_ids = label_ids[:,possible_starts[5]:]
                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    loss += outputs['loss']
            elif lang == 'en':
                if self.finetuning_args.argument_correct:
                    is_terminate = False
                    is_not_cal = False
                    if len(batch_labels) >= 4:
                        ground_truth = self.tokenizer.decode(batch_labels[3], skip_special_tokens=True)
                        # print("ground_truth", ground_truth)
                        ground_truth = ground_truth.strip()
                        
                    else:
                        is_terminate = True
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                        loss += outputs['loss']

                    if not is_terminate:
                        signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                        possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                    
                        new_input_ids = input_ids[:, possible_starts[5]:]
                        new_attention_mask = attention_mask[:, possible_starts[5]:]
                        new_labels_ids = label_ids[:,possible_starts[5]:]
                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

                        # loss += outputs['loss']
                            text = self.tokenizer.decode(input_ids[:, possible_starts[3]:possible_starts[4]][0], skip_special_tokens=True)
                            # print("text", text)
                            text = extract_pred_text(text)
                            token = text.split()
                            logits = torch.argmax(outputs.logits, dim=-1)
                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                            # output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=False)
                            output_result_ori = output_result_ori[0]

                            assistant_idx = output_result_ori.lower().rfind("assistant")
                            user_idx = output_result_ori.lower().rfind("user")

                            if assistant_idx > user_idx:
                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                # print("提取 assistant 后的文本：\n", output_result)
                            else:
                                output_result = output_result_ori[user_idx + len("user"):]
                                # print("提取 user 后的文本：\n", output_result)
                            output_result = output_result.strip()
                            if  'i have understood' in output_result.lower():
                                is_not_cal = True
                            if len(output_result) != 0:
                                if output_result[0] == ':':
                                    output_result = output_result[1:]
                            else:
                                is_not_cal = True
                            output_result = output_result.strip()
                            # if len(output_result) > 3*len(text):
                            #     # 如果很长的话 肯定不是好的回答
                            #     is_not_cal = True
                            if 'semantic role labeling' in output_result or 'core argument' in output_result:
                                is_not_cal = True
                        if is_not_cal or inputs['input_ids'].shape[-1] > max_length:
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            loss += outputs['loss']
                            # print("output_result", output_result)

                            # 有点搞不懂while了 直接写三次把
                        
                            # 超过长度 就算再循环也没用
                        else:
                            another_prompt = None
                            argument_result = output_result
                            if argument_result in ground_truth:
                                is_terminate = True
                                gold_output = 'Stop checking.'
                            else:
                                gold_output = find_reason_en(argument_result.lower(), ground_truth.lower(), token)

                            if 'The generated text is inconsistent with the original text' in gold_output:
                                gold_output += f'Original text: {text}\n'

                            if gold_output != 'Stop checking.' or 'Stop checking.' not in gold_output:
                                gold_output = 'There are problems: ' + gold_output + '\n'
                                gold_output += 'Argument labeling result:\n' + ground_truth

                            gold_output += user_suffix
            
                            prompt = '''Check the generated argument identification result. If any of the following issues exist:
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



                            prompt = prompt.replace('<<text>>', text)
                            prompt = prompt.replace('<<output>>', argument_result)
                        
                            
                            if another_prompt is not None:
                                prompt = another_prompt + prompt
                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids = extra_label_ids.unsqueeze(0)
                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                            with torch.no_grad():
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                            
                            # loss += outputs['loss']

                        
                            if not is_terminate:
                            
                                # 循环第二次
                                is_break = False
                                is_terminate = False
                                with torch.no_grad():
                                    logits = torch.argmax(outputs.logits, dim=-1)
                                    output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                    output_result_ori = output_result_ori[0]
                                    assistant_idx = output_result_ori.lower().rfind("assistant")
                                    user_idx = output_result_ori.lower().rfind("user")

                                    if assistant_idx > user_idx:
                                        output_result = output_result_ori[assistant_idx + len("assistant"):]
                                        # print("提取 assistant 后的文本：\n", output_result)
                                    else:
                                        output_result = output_result_ori[user_idx + len("user"):]
                                        # print("提取 user 后的文本：\n", output_result)
                                    output_result = output_result.strip()
                                    # print("new output_result", output_result)

                                # if len(output_result) > 3*len(text):
                                #     # 如果很长的话 肯定不是好的回答
                                #     is_break = True
                                if 'semantic role labeling' in output_result or 'its interpretation is' in output_result:
                                    is_break = True

                                if 'stop checking' in output_result.lower() or ('stop' in output_result.lower() and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                    is_break = True
                                elif 'argument labeling result:' in output_result:
                                    argument_result = output_result[output_result.rfind('argument labeling result:')+len('argument labeling result:'):]
                                else:
                                    is_break = True
                                # 严格按照格式输出
                                # elif '识别结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                # elif '结果：' in output_result:
                                #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                # else:
                                #     argument_result = output_result
                                


                                if not is_break:
                                    argument_result = argument_result.strip()
            
                                    if argument_result in ground_truth:
                                        is_terminate = True
                                        gold_output = 'Stop checking.'
                                    else:
                                        gold_output = find_reason_en(argument_result.lower(), ground_truth.lower(), token)

                                    if 'The generated text is inconsistent with the original text' in gold_output:
                                        gold_output += f'Original text: {text}\n'

                                    if gold_output != 'Stop checking.' or 'Stop checking.' not in gold_output:
                                        gold_output = 'There are problems: ' + gold_output + '\n'
                                        gold_output += 'Argument labeling result:\n' + ground_truth

                                    gold_output += user_suffix
                    
                                    prompt = '''Check the generated argument identification result. If any of the following issues exist:
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


                                    prompt = prompt.replace('<<text>>', text)
                                    prompt = prompt.replace('<<output>>', argument_result)
                                
                                    
                                    if another_prompt is not None:
                                        prompt = another_prompt + prompt
                                    extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                    extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids = extra_label_ids.unsqueeze(0)
                                    terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                    extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                    extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                    extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                    extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                    extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                    # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                    new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                    new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                    new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                    with torch.no_grad():
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    
                                    # loss += outputs['loss']
                                    is_break =False
                                    if not is_terminate:
                                        is_terminate = False
                                        with torch.no_grad():
                                            logits = torch.argmax(outputs.logits, dim=-1)
                                            output_result_ori = self.tokenizer.batch_decode(logits,skip_special_tokens=True)
                                            output_result_ori = output_result_ori[0]
                                            assistant_idx = output_result_ori.lower().rfind("assistant")
                                            user_idx = output_result_ori.lower().rfind("user")

                                            if assistant_idx > user_idx:
                                                output_result = output_result_ori[assistant_idx + len("assistant"):]
                                                # print("提取 assistant 后的文本：\n", output_result)
                                            else:
                                                output_result = output_result_ori[user_idx + len("user"):]
                                                # print("提取 user 后的文本：\n", output_result)
                                            output_result = output_result.strip()
                                            # print("new output_result", output_result)

                    
                                        # if len(output_result) > 3*len(text):
                                        #     # 如果很长的话 肯定不是好的回答
                                        #     is_break = True
                                        if 'semantic role labeling' in output_result or 'its interpretation is' in output_result:
                                            is_break = True

                                        if 'stop checking' in output_result.lower() or ('stop' in output_result.lower() and '##' not in output_result and '@@' not in output_result and '</' not in output_result):
                                            is_break = True
                                        elif 'argument labeling result:' in output_result:
                                            argument_result = output_result[output_result.rfind('argument labeling result:')+len('argument labeling result:'):]
                                        else:
                                            is_break = True
                                        # elif '识别结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('识别结果：')+len('识别结果：'):]
                                        # elif '结果：' in output_result:
                                        #     argument_result = output_result[output_result.rfind('结果：')+len('结果：'):]
                                        # else:
                                        #     argument_result = output_result
                                        


                                        if not is_break:
                                            argument_result = argument_result.strip()
                                            # print("argument_result", argument_result)
                                            if argument_result in ground_truth:
                                                is_terminate = True
                                                gold_output = 'Stop checking.'
                                            else:
                                                gold_output = find_reason_en(argument_result.lower(), ground_truth.lower(), token)

                                            if 'The generated text is inconsistent with the original text' in gold_output:
                                                gold_output += f'Original text: {text}\n'

                                            if gold_output != 'Stop checking.' or 'Stop checking.' not in gold_output:
                                                gold_output = 'There are problems: ' + gold_output + '\n'
                                                gold_output += 'Argument labeling result:\n' + ground_truth

                                            gold_output += user_suffix
                            
                                            prompt = '''Check the generated argument identification result. If any of the following issues exist:
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


                                            prompt = prompt.replace('<<text>>', text)
                                            prompt = prompt.replace('<<output>>', argument_result)
                                        
                                            
                                            if another_prompt is not None:
                                                prompt = another_prompt + prompt
                                            extra_input_ids = self.tokenizer(user_presuffix+prompt+user_suffix+assistant_presuffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = [-100] *extra_input_ids.shape[-1]
                                            extra_label_ids = torch.tensor(extra_label_ids).to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids = extra_label_ids.unsqueeze(0)
                                            terminate_tensor = self.tokenizer(gold_output , add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                                            extra_label_ids_new = torch.cat((extra_label_ids, terminate_tensor), dim=-1)
                                            extra_input_ids_new = torch.cat((extra_input_ids, terminate_tensor), dim=-1)

                                            extra_attention_mask = [1] * extra_label_ids_new.shape[-1]
                                            extra_attention_mask = torch.tensor(extra_attention_mask).to(attention_mask.device).to(attention_mask.dtype)
                                            extra_attention_mask = extra_attention_mask.unsqueeze(0)
                                            # print("prompt", user_presuffix+prompt+user_suffix+assistant_presuffix+gold_output)

                                            new_input_ids = torch.cat([new_input_ids, extra_input_ids_new], dim=-1)
                                            new_attention_mask = torch.cat([new_attention_mask, extra_attention_mask], dim=-1)
                                            new_labels_ids = torch.cat([new_labels_ids, extra_label_ids_new], dim=-1)
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            # inputs = {'attention_mask': extra_attention_mask, "input_ids":extra_input_ids_new, 'labels':extra_label_ids_new}
                                            with torch.no_grad():
                                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            
                                            # loss += outputs['loss']
                                        else:
                                            inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                            loss += outputs['loss']
                                    else:
                                        inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                        loss += outputs['loss']
                                else:
                                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                    loss += outputs['loss']
                            else:
                                inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                                loss += outputs['loss']
                else:
                    signal_tensor = self.tokenizer(signal, add_special_tokens=False, return_tensors='pt')['input_ids'].to(input_ids.device).to(input_ids.dtype)
                    possible_starts = torch.where(input_ids[0] == signal_tensor[0][0])[0]
                    
                    new_input_ids = input_ids[:, possible_starts[5]:]
                    new_attention_mask = attention_mask[:, possible_starts[5]:]
                    new_labels_ids = label_ids[:,possible_starts[5]:]
                    inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                    loss += outputs['loss']
            if isinstance(loss, float):
                inputs = {'attention_mask': new_attention_mask, "input_ids":new_input_ids, 'labels':new_labels_ids}
                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                loss = outputs['loss']
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            
            # print("final loss", loss)
            if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                loss *= self.accelerator.num_processes
            
            return (loss, outputs) if return_outputs else loss
        else:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                # User-defined compute_loss function
                if self.compute_loss_func is not None:
                    loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                loss *= self.accelerator.num_processes
            
            return (loss, outputs) if return_outputs else loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))