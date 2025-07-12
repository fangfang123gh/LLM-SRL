import subprocess
import os
import json
import random
_SRL_CONLL_EVAL_SCRIPT = 'eval.sh'
#  修改自supar

def test_to_BES_graph(file_name, out_file_name):
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
    new_sentence_lsts = []
    
    for data in datas:
        new_sentence_lst = []
        text = data['text']
        srl = data['srl']
        sorted_srl = sorted(srl, key=lambda x: x['position'][0])
        words = text
        for num, word in enumerate(words, 1):
            new_line_lst = [str(num), word, '_', '_', '_', '_', '_', '_']
            # 还剩下倒数第二列放边，最后一列放'_'
            new_sentence_lst.append(new_line_lst)
        
        arc_lsts = [[] for i in range(len(new_sentence_lst))]
        for pred in sorted_srl:
            arcs = [[] for i in range(len(new_sentence_lst))]

            pred_begin, pred_end = pred['position']
            arcs[pred_begin - 1].append((0, '[prd]'))

            # 谓词span也要识别
            # for i in range(pred_begin, pred_end + 1):
            #     if pred_begin == pred_end:
            #         arcs[i - 1].append((pred_begin, 'S-V'))
            #     else:
            #         if i == pred_begin:
            #             arcs[i - 1].append((pred_begin, 'B-V'))
            #         elif i == pred_end:
            #             arcs[i - 1].append((pred_begin, 'E-V'))
            # 另一种谓词打标签的方式
            for i in range(pred_begin + 1, pred_end + 1):
                if pred_begin + 1 == pred_end:
                    arcs[i - 1].append((pred_begin, 'S-V'))
                else:
                    if i == pred_begin + 1:
                        arcs[i - 1].append((pred_begin, 'B-V'))
                    elif i == pred_end:
                        arcs[i - 1].append((pred_begin, 'E-V'))
            
            for arg in pred['arguments']:
                arg_begin, arg_end = arg['position']
                for i in range(arg_begin, arg_end + 1):
                    if arg_begin == arg_end:
                        arcs[i - 1].append((pred_begin, 'S-'+arg['role']))
                    else:
                        if i == arg_begin:
                            arcs[i - 1].append((pred_begin, 'B-'+arg['role']))
                        elif i == arg_end:
                            arcs[i - 1].append((pred_begin, 'E-'+arg['role']))

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]
        
        for i in range(len(arc_lsts)):
            arc_values = []
            for arc in arc_lsts[i]:
                head_idx = arc[0]
                label = arc[1]
                arc_values.append(str(head_idx)+':'+label)
            if(len(arc_values) > 0):
                new_sentence_lst[i] += ['|'.join(arc_values), '_']
            else:
                new_sentence_lst[i] += ['_', '_']
        new_sentence_lsts.append(new_sentence_lst)

    with open(out_file_name, 'w', encoding='utf-8') as f:
        for sentence_lst in new_sentence_lsts:
            for line_lst in sentence_lst:
                f.write('\t'.join(line_lst)+'\n')
            f.write('\n')


def produce_column_BES(relas, prd_idx):
    # used for BES
    flag = 0
    count = 0
    count2 = 0
    column = ['*'] * len(relas)
    column[prd_idx-1] = '(V*)'
    args = []
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            # column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            # column.append('*')
            i += 1
        elif (len(rel) == 0):
            # column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  
            if(position_tag == 'E'):
                column.append('*')   
                i += 1
                count += 1
            elif position_tag == 'S':
                args.append([i, i, label])
                i += 1
            else:
                span_start = i
                i += 1
                if i>=len(relas):
                    # column.append('(' + label + '*' + ')')
                    i += 1
                elif len(relas[i]) == 0:
                    while i < len(relas) and len(relas[i]) == 0:
                        i += 1
                    if i < len(relas):
                        if relas[i][0].startswith('E-'):
                            new_label = relas[i][0][2:]
                            if label != new_label:
                                count2 += 1
                            else:
                                args.append([span_start, i, label])
                        else:
                            count += 1
                        i += 1
                elif relas[i][0].startswith('B-'):
                    count += 1
                    continue
                elif relas[i][0].startswith('E-'):
                    new_label = relas[i][0][2:]
                    args.append([span_start, i, label])
                    if label != new_label:
                        count2 += 1
                    i += 1
                else:
                    # relas[i][0].startswith('S-')
                    new_label = relas[i][0][2:]
                    args.append([i, i, new_label])
                    i += 1
                    count += 1

    for st, ed, role in args:
        length = ed-st+1
        if length == 1:
            column[st] = '(' + role + '*' + ')'
        else:
            column[st] = '(' + role + '*'
            column[ed] = '*' + ')'

    return column, count, count2

def change_BES(source_file, tgt_file, task):
    '''
    for BES
    '''
    sent_idx = 0
    sum_conf1_count = 0
    sum_conf2_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sent_idx += 1
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        arc_value[prd_map[head_idx] - 1].append(rel)
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2 = produce_column_BES(this_prd_arc, this_prd_idx)
            sum_conf1_count += con1
            sum_conf2_count += con2
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    # print('conflict I-:'+str(sum_conf1_count))
    # print('conflict label:'+str(sum_conf2_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')


def get_results(gold_path, pred_path, file_seed, task):
    
    tgt_temp_file = 'tgt_temp_file' + file_seed

    change_BES(pred_path, tgt_temp_file, task)

    gold_tgt_temp_file = 'gold_tgt_temp_file' + file_seed

    change_BES(gold_path, gold_tgt_temp_file, task)

    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_tgt_temp_file, tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]
    print("eval_info", eval_info)
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, tgt_temp_file, gold_tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    print("eval_info2", eval_info2)
    os.remove(tgt_temp_file)
    os.remove(gold_tgt_temp_file)
    conll_recall = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_precision = float(str(eval_info2).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[5])
    print(conll_recall, conll_precision, conll_f1)
    return conll_recall, conll_precision, conll_f1, lisa_f1

if __name__ =='__main__':
    pred_path = ''
    tmp_pred_path = ''
    test_to_BES_graph(pred_path, tmp_pred_path)
    gold_path = ''
    tmp_gold_path= ''
    test_to_BES_graph(gold_path, tmp_gold_path)
    get_results(tmp_gold_path, tmp_pred_path, str(random.randint(1,100)), '12')
    os.remove(tmp_pred_path)
    os.remove(tmp_gold_path)