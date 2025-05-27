import glob
import pickle
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

def get_database(folder_path: str, save_path: str):
    file_pattern = os.path.join(folder_path, '*.xml')
    xml_files = glob.glob(file_pattern)

    all_preds_dict = {}

    for file in tqdm(xml_files, desc="Processing XML files"):
        with open(file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')

        pred = soup.find('id').text.strip()
        framesets = soup.find_all('frameset')

        process_framesets = []

        for frameset in framesets:
            roles = {}
            for role in frameset.find_all('role'):
                roles[role['argnum']] = role['argrole']
            process_framesets.append(roles)

        all_preds_dict[pred] = process_framesets

    with open(save_path, 'wb') as f:
        pickle.dump(all_preds_dict, f)

# 主程序入口
if __name__ == '__main__':
    get_database(
        folder_path='',
        save_path=''
    )
