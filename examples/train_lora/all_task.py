cpb_path = '/data/lxx/blsp-main/data/cpb_train_srl_conv_all_rag_zh_token_second_confirm.json'
conll09_en_path = '/data/lxx/blsp-main/data/conll09_en_train_srl_rag_new.json'
conll09_zh_path = '/data/lxx/blsp-main/data/conll_train_srl_conv_all_rag_zh_token_second_confirm_sample.json'
conll12_en_path = '/data/lxx/blsp-main/data/conll12_train_srl_rag_second_confirm_new.json'
import json
path = [cpb_path, conll09_en_path, conll09_zh_path, conll12_en_path]
dataset =  ['Chinese propbank dataset', 'Conll09 English dataset', 'Conll09 Chinese dataset', 'Conll12 English dataset']
all_datas = []
for p, datas in zip(path, dataset):
    with open(p, 'r', encoding='utf-8') as f:

        data = json.load(f)
        for d in data:
            modify_data = d['conversations'][4]['value']
            modify_data = modify_data.split('\n')
            if 'Chinese' in datas:
                modify_data[1] ='对于'+ datas+'，'+ modify_data[1] 
            elif 'English' in datas:
                if modify_data[1][0].isupper():
                    modify_data[1] = modify_data[1][0].lower() + modify_data[1][1:]
                modify_data[1] ='For '+ datas+', '+ modify_data[1] 
            modify_data = '\n'.join(modify_data)
            d['conversations'][4]['value'] = modify_data
            all_datas.append(d)
            # '在语义角色标注中，论元指的是语义上与给定谓词相关的成分或短语。它进一步描述了与句子中谓词相关的实体、动作或概念。论元分为核心论元和附加论元。\n所有附加论元的标签如下：\nARGM-ADV：副词性论元\nARGM-BNF：受益者\nARGM-CND：条件\nARGM-DIR：方向\nARGM-EXT：表示动作或事件的程度\nARGM-FRQ：表示动作或事件的频率\nARGM-LOC：表示动作或事件发生的地点\nARGM-MNR：表示动作或事件的执行方式\nARGM-PRP：目的\nARGM-TMP：时间\nARGM-TPC：主题\nARGM-DGR：程度\nARGM-DIS：话语\nARGM-CRD：并列\nARGM-PRD：子谓词\n核心论元依赖于谓词，且谓词可能具有不同的核心论元框架。在这些框架内，核心论元会有不同的解释。\n'
    print("11111111")

json_data = json.dumps(all_datas,ensure_ascii=False)
with open(f'/data/lxx/blsp-main/data/all_srl.json', "w", encoding='utf-8') as file:
    file.write(json_data)