import json
import random
random.seed(42)
datas = []
path = ''
save_path = ''
number = 800
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)

sample_datas = random.sample(datas, number)
with open(save_path, 'w', encoding='utf-8') as f:
    for data in sample_datas:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')