# 首先是读取frame 然后对每一个谓词生产prompt
# 生成prompt后输入gpt 让gpt回答
# 后处理gpt的回答
import pickle
from gpt_infer import infer

predicate_base_path = ''
save_path = ''
with open(predicate_base_path, 'rb') as f:
    preds_dict = pickle.load(f)

all_preds_dict = {}
for pred, _ in preds_dict.items():
    prompt = f'是一个总结谓词框架含义的专家。你的任务是根据给出的中文谓词和框架论元解释总结出谓词在当前框架下的含义。其中，ARGN是语义角色标注任务中论元的标签。\n以下是给英文的例子：\n谓词：abolish\n框架：ARG0：entity getting rid of, outlawing something；ARG1：thing abolished\n谓词含义：get rid of, make illegal\n\n谓词：act\n框架：ARG0：agent；ARG1：predicate\n谓词含义：play a role; behave\n\n谓词：act\n框架：ARG0：actor；ARG1：rounds for action\n谓词含义：do something\n\n谓词：act\n框架：ARG0：actor, performer；ARG1：role, scenario enacted\n谓词含义：perform a role\n\n谓词：{pred}\n框架：ARG1：goal\n谓词含义：\n直接输出谓词含义。\n'
    desc = infer(prompt)
    if '谓词含义：' in desc:
        index = desc.find('谓词含义：')
        desc = desc[index+len('谓词含义：')]
    elif '含义：' in desc:
        index = desc.find('含义：')
        desc = desc[index+len('含义：')]
    elif '：' in desc:
        index = desc.find('：')
        desc = desc[index+1:]
    
    if '没有输出' in desc:
        continue
    else:
        if '。' in desc[-1]:
            desc = desc[:-1]
        if pred in all_preds_dict:
            if desc not in all_preds_dict[pred]:
                all_preds_dict[pred].append(desc)
        else:
            all_preds_dict[pred] = [desc]

        
with open(save_path, 'wb') as f:
    pickle.dump(all_preds_dict, f)