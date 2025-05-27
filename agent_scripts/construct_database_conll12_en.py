import glob
import os
import pickle
from bs4 import BeautifulSoup

def get_database(folder_path: str, save_path: str):
    file_pattern = os.path.join(folder_path, '*.xml')
    xml_files = glob.glob(file_pattern)

    all_preds_dict = {}

    for file in xml_files:
        filename = os.path.basename(file)
        pos = filename[-5]  # 'v' or 'n'
        annot_word = filename[:-6]  # remove '-v.xml' or '-n.xml'

        with open(file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'xml')

        preds = soup.find_all('predicate')
        for pred in preds:
            if annot_word not in all_preds_dict:
                all_preds_dict[annot_word] = {'n': [], 'v': []}

            for frameset in pred.find_all('roleset'):
                name_des = frameset['name']
                all_preds_dict[annot_word][pos].append(name_des)

    with open(save_path, 'wb') as f:
        pickle.dump(all_preds_dict, f)

# 主程序入口
if __name__ == '__main__':
    get_database(
        folder_path='',
        save_path=''
    )
