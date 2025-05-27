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
    
def penn_to_lemma(tag):
    if tag.startswith('J'):
        return 'ADJ'
    elif tag.startswith('V'):
        return 'VERB'
    elif tag.startswith('N'):
        return 'NOUN'
    elif tag.startswith('R'):
        return 'ADV'
    elif tag.startswith('P'):
        return 'PROPN'
    else:
        return 'AUX'
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
            token = data['words']
            pos = data['all_pos']
            lemmas = data['lemmas']
            
            pr_str = copy.deepcopy(token)
            sorted_result = sorted(preds, key=lambda x: x['position'][0])

            for a in sorted_result:
                start, end = a['position']
                pr_str[start - 1] ='@@'+ pr_str[start - 1]
                pr_str[end - 1] =pr_str[end - 1] + '##'
            pr_str = ' '.join(pr_str)
            if require_pred:
                i = 0
                maybe_pred_pos = []
                maybe_pred_token = []
                while i < len(token):
                    t = token[i].lower()
                    temp_lemma = lemmas[i]
                    if t in preds_dict:
                        maybe_pred_pos.append(i)
                        maybe_pred_token.append(t)

                    elif temp_lemma != '-':
                        if temp_lemma in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(temp_lemma)
                        elif temp_lemma == 'liquefy':
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append('liquify')
                    elif i != len(token) - 1 and token[i + 1] == '-':
                        t = temp_lemma
                        temp_lemma = lemmas[i+2]
                        if ( t + temp_lemma ) in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(t + temp_lemma)
                        elif ( t +'-'+ temp_lemma ) in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(t +'-'+ temp_lemma)
                    elif i != 0 and token[i-1] == '-':
                        t = temp_lemma
                        temp_lemma = lemmas[i-2]
                        if ( temp_lemma+t) in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(temp_lemma+t)
                        elif ( temp_lemma +'-'+ t ) in preds_dict:
                            maybe_pred_pos.append(i)
                            maybe_pred_token.append(temp_lemma +'-'+ t)
                    else:
                        if pos[i] not in ['HYPH', ':']:
                            lemma = getLemma(t, penn_to_lemma(pos[i]))
                            for l in lemma:

                                if l in preds_dict:
                                    maybe_pred_pos.append(i)
                                    maybe_pred_token.append(l)
                                    break
                            if i not in preds_dict:
                                lemma = getLemma(t, 'VERB')
                                for l in lemma:

                                    if l in preds_dict:
                                        maybe_pred_token.append(l)
                                        maybe_pred_pos.append(i)
                                        break
            
                    i += 1

                all_maybe_pr_str = copy.deepcopy(token)
                pred_agent_des = ''
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


                
                

                for a in maybe_pred_pos:
                    all_maybe_pr_str[a] ='@@'+ all_maybe_pr_str[a] + '##'
                all_maybe_pr_str = ' '.join(all_maybe_pr_str)

        for r in preds:
            conversations = []
            task_exp = 'Semantic Role Labeling (SRL) aims to identify predicates in a sentence and assign roles to their arguments.\n'
            pred_exp = 'A predicate refers to the core word or phrase in a sentence that conveys an action, event, or state and serves as the focus for other elements in the sentence.\n'
            conversations.append({'from': 'human', "value": task_exp + pred_exp})
            conversations.append({"from": "gpt", "value": 'I have understood this task.'})
            if require_pred:
                conversations.append({'from': 'human', "value": f'Text: {" ".join(token)}\nFor the SRL task, what are the predicates in the given text? Possible predicate results in the text are: {all_maybe_pr_str}\nwhere predicates are specified by @@ and ##.\nBased on the given possible predicate results, please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively. Note that words not present in the predicate results may also be predicates.\n'+pred_agent_des})
            else:
                conversations.append({'from': 'human', "value": f'Text: {" ".join(token)}\nFor the SRL task, what are the predicates in the given text? Please rewrite the given text, marking the beginning and end of predicates with @@ and ## respectively.'})
            conversations.append({"from": "gpt", "value": pr_str})

            instruction = "In SRL, arguments refer to the components or phrases semantically related to a given predicate. They further describe the entities, actions, or concepts associated with the predicate in the sentence. For the argument, only the head of it needs to be identified. "
            instruction += "Arguments are divided into core arguments and adjunct arguments.\n"
            instruction += "The labels for all adjunct arguments are as follows:\n"
            instruction += 'AM-TMP: temporal\nAM-LOC: location\nAM-MNR: manner\nAM-NEG: negation\n'
            instruction += 'AM-MOD: general modification\nAM-DIS: discourse\nAM-EXT: extent\nAM-ADV: adverbial modification\n'
            instruction += 'AM-PNC: purpose no cause\nAM-DIR: direction\nAM-PRD: secondary predication\nAM-CAU: cause\nAM: argument modification'
            instruction += 'AM-REC: recipricol (eg herself, etc)\nAM-PRT: particle\n'
            instruction+= 'AA: secondary agent\n'
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
                wn_tag = penn_to_wordnet(pos[start - 1])
                v_or_n =  'v' if wn_tag == wordnet.VERB else None
                if v_or_n is None:
                    v_or_n =  'n' if wn_tag == wordnet.NOUN else None
                frameset_str = ''
                framesets = {}
                lemma = ''
                if lemma not in preds_dict:
                    lemma = pred
                if lemma not in preds_dict:
                    lemma = pred.lower()
                if lemma not in preds_dict:
                    if lemmas[start - 1] != '-':
                        lemma = lemmas[start - 1]
                if lemma not in preds_dict:
                    lemma_list = getLemma(pred.lower(), penn_to_lemma(pos[start - 1]))
                    for lemma in lemma_list:
                        if lemma in preds_dict:
                            break
                
                if lemma not in preds_dict:
                    lemma_list = getLemma(pred.lower(), 'VERB')
                    for lemma in lemma_list:
                        if lemma in preds_dict:
                            break
                if lemma== 'liquefy':
                    lemma = 'liquify'
                if lemma not in preds_dict:
                    t= lemma
                    i = start - 1
                    if i != 0 and token[i-1] == '-':
                        temp_lemma = lemmas[i-2]
                        if ( temp_lemma+t) in preds_dict:
                            lemma = temp_lemma+t
                        elif ( temp_lemma +'-'+ t ) in preds_dict:
                            lemma = temp_lemma +'-'+ t 
                    elif i != len(token) - 1 and token[i + 1] == '-':
                        temp_lemma = lemmas[i+2]
                        if ( t + temp_lemma ) in preds_dict:
                            lemma = t + temp_lemma
                        elif ( t +'-'+ temp_lemma ) in preds_dict:
                            lemma=  t +'-'+ temp_lemma
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
                
                if lemma in preds_dict:
                    for k, _ in preds_dict[lemma].items():
                        framesets[k] = preds_dict[lemma][k]   
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
                                    framesets.append(value)

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
                                        frameset_str += f"A{frame_role}: {frame_exp}\n" 
                            else:
                                n = 'verb' if key == 'v' else 'noun'
                                frameset_str += f'For {frame_name} as a {n}\n'
                        
                                for fram_index, f in enumerate(value):
                                    if len(f) == 0:
                                        continue
                                    frameset_str += f'Frame {fram_index + 1}:\nThe core arguments it has are:\n'
                                    for frame_role, frame_exp in f.items():
                                        frameset_str += f"A{frame_role}: {frame_exp}\n" 
                if len(frameset_str) != 0:
                    instruction += f'For the predicate "{pred}" in this text, it has the following frames:\n'
                    instruction += frameset_str
                    instruction += "By referring to the provided frames, determine the frame to which the predicate belongs in order to identify its core arguments.\n"
                else:
                    instruction += "The labels for all core arguments are as follows:\n"
                    instruction += "A0: agent\nA1: patient\nA2: instrument, benefactive, attribute\nA3: starting point, benefactive, attribute\nA4: ending point\nA5: depend on predicate"
            else:
                instruction += "The labels for all core arguments are as follows:\n"
                instruction += "A0: agent\nA1: patient\nA2: instrument, benefactive, attribute\nA3: starting point, benefactive, attribute\nA4: ending point\nA5: depend on predicate"
            instruction += '"R-" arguments are arguments that are referencing another argument in the sentence. "C-" arguments are discontinous spans that all refer to the same argument. Please rewrite the given text and enclose the beginning and end of the arguments with the corresponding <label> and </label> tags.\n'
            sorted_args = sorted(args, key=lambda x: x['position'][0])
            for a in sorted_args:
                role = a['role']
                start, end = a['position']
                text[start - 1] = f'<{role}>' + text[start - 1]
                text[end - 1] = text[end - 1] + f'</{role}>'

            conversations.append({"from": "human", "value": instruction})
            gpt_template = {"from": "gpt", "value": ' '.join(text)}
            conversations.append(gpt_template)
            datas.append({'conversations': conversations ,'system': 'You are a helpful assistant who has a background in linguistics and is good at understanding texts, especially skilled in semantic role labeling recognition.'})

    json_data = json.dumps(datas,ensure_ascii=False)
    with open(save_path, "w", encoding='utf-8') as file:
        file.write(json_data)

if __name__ == '__main__':
    predicate_base_path = '' # the predicate database path
    agent_path = '' # the predicate agent path
    data_path = '' # the training data
    get_instruction_data(predicate_base_path, agent_path, data_path)
