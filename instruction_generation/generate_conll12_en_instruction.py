import json
import copy
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tqdm import tqdm
from lemminflect import getLemma
from nltk.corpus import wordnet

def penn_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def get_instruction_data(predicate_base_path, agent_path, data_path, save_path, require_pred=True, require_rl=True):
    datas = []
    with open(predicate_base_path, 'rb') as f:
        preds_dict = pickle.load(f)

    with open(agent_path, 'rb') as f:
        pred_agent = pickle.load(f)

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            preds = data['srl']
            text = ' '.join(data['text'])
            token = text.split()
            lemmas = data['lemmas']
            pos = data['pos']
            
            pr_str = copy.deepcopy(token)
            sorted_result = sorted(preds, key=lambda x: x['position'][0])

            # construct the gold-standard predicate recognition sequence
            for a in sorted_result:
                start, end = a['position']
                pr_str[start - 1] ='@@'+ pr_str[start - 1]
                pr_str[end - 1] =pr_str[end - 1] + '##'
            pr_str = ' '.join(pr_str)

            # construct the possible predicate recognition sequence
            i = 0
            maybe_pred_pos = []
            maybe_pred_token = []
            while i < len(token):
                t = token[i].lower()
                temp_lemma = lemmas[i]
                # 先直接看在不在里面
                if t in preds_dict:
                    maybe_pred_pos.append(i)
                    maybe_pred_token.append(t)
                elif temp_lemma != '-':
                    if temp_lemma in preds_dict:
                        maybe_pred_pos.append(i)
                        maybe_pred_token.append(temp_lemma)

                else:
                    
                    lemma = getLemma(t, 'NOUN')
                    for l in lemma:

                        if l in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(l)
                            break
                    if i not in maybe_pred_pos:
                        lemma = getLemma(t, 'VERB')
                        for l in lemma:

                            if l in preds_dict:
                                maybe_pred_token.append(l)
                                maybe_pred_pos.append(i)
                                break    
                i += 1
            pred_agent_des = ''
            all_maybe_pr_str = ''
            if require_pred:
                all_maybe_pr_str = copy.deepcopy(token)
                for a in maybe_pred_pos:
                    all_maybe_pr_str[a] ='@@'+ all_maybe_pr_str[a] + '##'
                all_maybe_pr_str = ' '.join(all_maybe_pr_str)

                
            
                maybe_pred_token = set(maybe_pred_token)
                maybe_pred_token = list(maybe_pred_token)
                for t in maybe_pred_token:
                    for key, value in pred_agent[t].items():
                        temp = ''
                        if len(value) != 0:
                            pos_name = 'noun' if key == 'n' else 'verb'
                            temp += f'When the {pos_name} "{t}" functions as a predicate, its interpretation is: '
                
                            unique_value = []
                            for i,v in enumerate(value):
                                find = False
                                for j,vv in enumerate(value):
                                    if i != j:
                                        if v in vv or v== t:
                                            find = True
                                            break
                                if not find:
                                    unique_value.append(v)
                            if len(unique_value) == 0:
                                temp = ''
                            else:
                                temp += ", ".join(unique_value) + '\n'
                        pred_agent_des += temp
        

            for r in preds:
                conversations = []
                task_exp = 'Semantic Role Labeling (SRL) aims to identify predicates in a sentence and assign roles to their arguments.\n'
                pred_exp = 'A predicate refers to the core word or phrase in a sentence that conveys an action, event, or state and serves as the focus for other elements in the sentence.\n'
                conversations.append({'from': 'human', "value": task_exp + pred_exp})
                conversations.append({"from": "gpt", "value": 'I have understood this task.'})
                
                if require_pred:
                    conversations.append({'from': 'human', "value": f'Text: {" ".join(token)}\nFor the predicate indentification task, what are the predicates in the given text? Possible predicate results in the text are: {all_maybe_pr_str}\nwhere predicates are specified by @@ and ##.\n'+pred_agent_des+'Based on the given possible predicate results and interpretations, please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively. Note that words not present in the predicate results may also be predicates.\n'})
                else:
                    conversations.append({'from': 'human', "value": f'Text: {" ".join(token)}\nFor the predicate indentification task, what are the predicates in the given text? Please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively.'})
                # conversations.append({'from': 'human', "value": f'Text: {key}\n在进行语义角色标注任务时，给定文本中的谓词是什么？请重新编写给定文本，并使用@@和##分别标记谓词的开头和结尾。'})
                conversations.append({"from": "gpt", "value": pr_str})

                instruction = "In SRL, arguments refer to the components or phrases semantically related to a given predicate. They further describe the entities, actions, or concepts associated with the predicate in the sentence."
                instruction += "Arguments are divided into core arguments and adjunct arguments.\n"
                instruction += "The labels for all adjunct arguments are as follows:\n"
                instruction += 'ARGM-EXT: extent\nARGM-LOC: location\nARGM-DIR: direction\nARGM-NEG: negation  (not in PREDITOR)\n'
                instruction += 'ARGM-MOD: general modification\nARGM-ADV: adverbial modification\nARGM-MNR: manner\nARGM-PRD: secondary predication\n'
                instruction += 'ARGM-REC: recipricol (eg herself, etc)\nARGM-TMP: temporal\nARGM-PRP: purpose\nARGM-PNC: purpose no cause\nARGM-CAU: cause\n'
                instruction += 'ARGM-ADJ: adjectival (nouns only)\nARGM-COM: comitative\nARGM-DIS: discourse\nARGM-DSP: direct speech\n'
                instruction += 'ARGM-GOL: goal\nARGM-LVB: light verb (for nouns only)\nARGA: secondary agent\nARGM-PRR: predicating relation\n'
                pred = r['pred']
                
                instruction += "Core arguments depend on the predicate, and a predicate may have different core argument frames. Within these frames, core arguments will have different interpretations.\n"
                conversations.append({'from': 'human', "value": instruction})
                conversations.append({"from": "gpt", "value": 'I have understood this task.'})
                instruction = ''

                
                args = r['arguments']
                # instruction = ''
                # instruction += conversations[0]['value']
                start, end = r['position']
                text = copy.deepcopy(token)

                text[start - 1] = '@@' + text[start - 1]
                text[end - 1] = text[end - 1] + '##'

                question = f"Text: {' '.join(text)}\nWhat are the arguments and their corresponding roles for the given predicate? The predicate is specified by @@ and ##.\n"
                instruction += question
                
                if require_rl:

                    # 框架的组织
                    frameset_str = ''
                    wn_tag = penn_to_wordnet(pos[start - 1])
                    v_or_n =  'v' if wn_tag == wordnet.VERB else None
                    if v_or_n is None:
                        v_or_n =  'n' if wn_tag == wordnet.NOUN else None
                    framesets = {}
                    if lemmas[start - 1] != '-':
                        lemma = lemmas[start - 1]
                    elif pred in ['\'s', '\'re', '\'m']:
                        lemma = 'be'
                    else:
                        
                        lemma = lemmatizer.lemmatize(pred.lower(), pos=wn_tag)
                    if lemma not in preds_dict:
                        lemma = lemmatizer.lemmatize(pred.lower(), pos=wn_tag)
                    if lemma in preds_dict:
                        framesets = {}
                        temp_word = lemma
                        for j in range(1, 5):
                            if start + j <= len(token):
                                if token[start + j - 1] == '-':
                                    continue
                                temp_word += ' ' + token[start + j - 1]
                                if temp_word in preds_dict[lemma]:
                                    framesets[temp_word] = preds_dict[lemma][temp_word]
                                    break
                        if len(framesets) == 0:
                            if lemma in preds_dict[lemma]:
                                framesets[lemma] = preds_dict[lemma][lemma]
                    if len(framesets) == 0:
                        if pred in preds_dict:
                            if pred in preds_dict[pred]:
                                framesets[pred] = preds_dict[pred][pred]
                            
                            else:
                                if len(preds_dict[pred]) != 0:
                                    for key, value in preds_dict[pred].items():
                                        framesets[key] = value
                        elif lemma in preds_dict:
                            if lemma in preds_dict[lemma]:
                                framesets[lemma] = preds_dict[lemma][lemma]
                            else:
                                if len(preds_dict[lemma]) != 0:
                                    for key, value in preds_dict[lemma].items():
                                        framesets[key] = value
                    for frame_name, frameset in framesets.items():
                        for key, value in frameset.items():
                            if len(value) != 0:
                                if v_or_n is not None and key == v_or_n:
                                    n = 'verb' if key == 'v' else 'noun'
                                    frameset_str += f'For {frame_name} as a {n}\n'
                            
                                    for fram_index, f in enumerate(value):
                                        if len(f) == 0:
                                            continue
                                        frameset_str += f'Frame {fram_index + 1}:\nThe core arguments it has are:\n'
                                        for frame_role, frame_exp in f.items():
                                            frameset_str += f"ARG{frame_role}: {frame_exp}\n"
                                else:
                                    n = 'verb' if key == 'v' else 'noun'
                                    frameset_str += f'For {frame_name} as a {n}\n'
                            
                                    for fram_index, f in enumerate(value):
                                        if len(f) == 0:
                                            continue
                                        frameset_str += f'Frame {fram_index + 1}:\nThe core arguments it has are:\n'
                                        for frame_role, frame_exp in f.items():
                                            frameset_str += f"ARG{frame_role}: {frame_exp}\n" 
                    if len(frameset_str) != 0:
                        instruction += f'For the predicate "{pred}" in this text, it has the following frames:\n'
                        instruction += frameset_str
                        instruction += "By referring to the provided frames, determine the frame to which the predicate belongs in order to identify its core arguments.\n"
                    
                    else:
                        instruction += "The labels for all core arguments are as follows:\n"
                        instruction += "ARG0: agent\nARG1: patient\nARG2: instrument, benefactive, attribute\nARG3: starting point, benefactive, attribute\nARG4: ending point\nARG5: depend on predicate"
                    
                else:
                    instruction += "The labels for all core arguments are as follows:\n"
                    instruction += "ARG0: agent\nARG1: patient\nARG2: instrument, benefactive, attribute\nARG3: starting point, benefactive, attribute\nARG4: ending point\nARG5: depend on predicate"
                instruction += '"R-" arguments are arguments that are referencing another argument in the sentence. "C-" arguments are discontinous spans that all refer to the same argument. Please rewrite the given text and enclose the beginning and end of the arguments with the corresponding <label> and </label> tags.\n'
                sorted_args = sorted(args, key=lambda x: x['position'][0])
                for a in sorted_args:
                    if a['role'] == 'ARGM-PRX':
                        continue
                    role = a['role']
                    start, end = a['position']
                    text[start - 1] = f'<{role}>' + text[start - 1]
                    text[end - 1] = text[end - 1] + f'</{role}>'

                conversations.append({"from": "human", "value": instruction})
                gpt_template = {"from": "gpt", "value": ' '.join(text)}
                conversations.append(gpt_template)
                datas.append({'conversations': conversations , 'system': 'You are a helpful assistant who has a background in linguistics and is good at understanding texts, especially skilled in semantic role labeling recognition.'})

    json_data = json.dumps(datas,ensure_ascii=False)
    with open(save_path, "w", encoding='utf-8') as file:
        file.write(json_data)

if __name__ == '__main__':
    predicate_base_path = '' # the predicate database path
    agent_path = '' # the predicate agent path
    data_path = '' # the training data
    save_path = ''
    get_instruction_data(predicate_base_path, agent_path, data_path, save_path)