import glob
from bs4 import BeautifulSoup
import pickle
import os

def parse_predicates(xml_files, pos_tag, all_preds_dict):
    for file in xml_files:
        annot_word = os.path.basename(file).replace('.xml', '')
        with open(file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'xml')

        preds = soup.find_all('predicate')
        for pred in preds:
            pred_word = pred['lemma'].replace('_', ' ')
            all_preds_dict.setdefault(annot_word, {'n': [], 'v': []})
            for frameset in pred.find_all('roleset'):
                name_des = frameset['name']
                all_preds_dict[annot_word][pos_tag].append(name_des)

        # 处理带有 - 的形式，如 'take-1'
        if '-' in annot_word:
            new_annot_word = annot_word.replace('-', '')
            all_preds_dict[new_annot_word] = all_preds_dict[annot_word]

def get_database(nb_frames_dir, pb_frames_dir, save_path):
    all_preds_dict = {}

    nb_xml_files = glob.glob(os.path.join(nb_frames_dir, '*.xml'))
    pb_xml_files = glob.glob(os.path.join(pb_frames_dir, '*.xml'))

    parse_predicates(nb_xml_files, 'n', all_preds_dict)
    parse_predicates(pb_xml_files, 'v', all_preds_dict)

    with open(save_path, 'wb') as f:
        pickle.dump(all_preds_dict, f)

if __name__ == '__main__':
    nb_frames_dir = '' 
    pb_frames_dir = '' 
    save_path = ''
    get_database(nb_frames_dir, pb_frames_dir, save_path)