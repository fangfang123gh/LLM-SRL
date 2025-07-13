import glob
import pickle
from bs4 import BeautifulSoup
import os

def build_propbank_predicate_dict(xml_folder, save_path):
    file_pattern = os.path.join(xml_folder, '*.xml')
    xml_files = glob.glob(file_pattern)

    all_preds_dict = {}

    for file in xml_files:
        filename = os.path.basename(file)
        pos = filename[-5]  # 倒数第5位是词性标签，如 'v'
        annot_word = filename.replace('.xml', '')[:-2]  # 去掉 .xml 和 -v / -n 后缀

        with open(file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        soup = BeautifulSoup(xml_content, 'xml')
        predicates = soup.find_all('predicate')

        for pred in predicates:
            pred_word = ' '.join(pred['lemma'].split('_'))

            if annot_word not in all_preds_dict:
                all_preds_dict[annot_word] = {'n': [], 'v': []}

            framesets = pred.find_all('roleset')
            for frameset in framesets:
                name_des = frameset.get('name')
                all_preds_dict[annot_word][pos].append(name_des)

    # 保存为 pickle
    with open(save_path, 'wb') as f:
        pickle.dump(all_preds_dict, f)
    print(f"保存成功：{save_path}")
    return all_preds_dict

if __name__ == '__main__':
    xml_folder = ''
    save_path = ''
    build_propbank_predicate_dict(xml_folder, save_path)