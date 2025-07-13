import glob
import pickle
from bs4 import BeautifulSoup
import os

def build_predicate_frameset_dict(nb_folder, pb_folder, save_path):
    def process_folder(folder_path, pos_tag):
        all_preds = {}
        file_pattern = os.path.join(folder_path, '*.xml')
        xml_files = glob.glob(file_pattern)

        for file in xml_files:
            annot_word = os.path.basename(file).replace('.xml', '')

            with open(file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            soup = BeautifulSoup(xml_content, 'xml')
            preds = soup.find_all('predicate')

            for pred in preds:
                pred_word = pred['lemma']
                pred_word = ' '.join(pred_word.split('_'))

                if annot_word not in all_preds:
                    all_preds[annot_word] = {'n': [], 'v': []}

                framesets = pred.find_all('roleset')
                for frameset in framesets:
                    name_des = frameset['name']
                    all_preds[annot_word][pos_tag].append(name_des)

            # 同时加入去除 "-" 的变体
            if '-' in annot_word:
                all_preds[''.join(annot_word.split('-'))] = all_preds[annot_word]

        return all_preds

    nb_dict = process_folder(nb_folder, 'n')
    pb_dict = process_folder(pb_folder, 'v')

    # 合并两个字典
    all_preds_dict = nb_dict
    for k, v in pb_dict.items():
        if k not in all_preds_dict:
            all_preds_dict[k] = v
        else:
            all_preds_dict[k]['v'].extend(v['v'])

    # 保存为 pickle 文件
    with open(save_path, 'wb') as f:
        pickle.dump(all_preds_dict, f)
    print(f"保存成功: {save_path}")


if __name__ == '__main__':
    nb_folder = '' 
    pb_folder = '' 
    save_path = ''
    build_predicate_frameset_dict(nb_folder, pb_folder, save_path)