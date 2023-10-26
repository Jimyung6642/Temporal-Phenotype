import tqdm as td
import os, glob
from datetime import date
import configparser

import pandas as pd

import xml.etree.ElementTree as ET
import re

def eval_ner(output_dir: str, execute_date: str, few_shot: bool = True):
    '''
    Get performane of the task.
    By comparing gold standard and GPT-generated data, calculate performance.
    
    execute_date = %y%m%d (string)
    
    Using start, text and type. (in addition, end, if necessary)
    TP - correctly match all start, text and type.
    FP - Model identifies the entity not in gold standard (wrong text or correct text with different start point)
    FN - Entity exists in gold standard, but not in model result OR text is correct but type is different
    '''
    # Read gold standard data
    original_files = glob.glob(os.path.join(output_dir, 'eval/ner', '*.xml'))
    # Read GPT-generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    model = config['openai']['model']
    
    if execute_date is None:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
        else:
            date_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
    else:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + execute_date + "/ner"
        else:
            date_path = "output_zero_" + model + "_" + execute_date + "/ner"
    path = os.path.join(output_dir, date_path)
    output_files = glob.glob(os.path.join(path, ".xml"))
    
    ### Calculate metrics
    # preprocess
    df_original_list = []
    df_output_list = []
    for original, output in td.tqdm(zip(original_files, output_files), total = len(original_files), desc = "Evaluating NER performance", unit = "files"):
        with open(original, 'r') as f:
            gold = f.read()
        with open(output, 'r') as f:
            gpt_output = f.read()
        
        # Replace unescaped special characters
        gpt_output = gpt_output.replace('&', '&amp;')
        # Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        for event in original_root.findall("EVENT"):
            row = {
                'noteID': os.path.splitext(os.path.base),
                'id': event.get('id'),
                'start': event.get('start'),
                'end': event.get('end'),
                'text': event.get('text'),
                'type': event.get('type')
            }
            original_rows_by_note.append(row)
        # Append by whole corpus
        df_original_list.append(pd.DataFrame(original_rows_by_note))
        
        ## Process output list
        output_root = ET.fromstring(gpt_output)
        output_rows_by_note = []
        # Append by individual note
        for event in output_root.findall("EVENT"):
            row = {
                'noteID': os.path.splitext(os.path.basename(output))[0],
                'id': event.get('id'),
                'start': event.get('start'),
                'end': event.get('end'),
                'text': event.get('text'),
                'type': event.get('type')
            }
            output_rows_by_note.append(row)
        # Append by whoel corpus
        df_output_list.append(pd.DataFrame(output_rows_by_note))
        
    df_original = pd.concat(df_original_list, ignore_index = True)
    df_output = pd.concat(df_output_list, ignore_index = True)
    
    merged_df = df_original.merge(df_output, on = ['start','text','type'], how = 'outer', indicator = True)
    
    TP = len(merged_df[merged_df['_merge'] == 'both'])
    FP = len(merged_df[merged_df['_merge'] == 'right_only'])
    FN = len(merged_df[merged_df['_merge'] == 'left_only'])
    
    ### Calculate precision, recall, and f1-score
    # Micro-averaged metrics
    micro_precision = TP / (TP + FP) if TP + FP != 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN != 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0

    # For macro-average, we need to calculate metrics for each 'type' and then average them
    types = df_original['type'].unique().tolist() + df_output['type'].unique().tolist()
    types = list(set(types))  # get unique types

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    for t in types:
        TP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'both')])
        FP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'right_only')])
        FN_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'left_only')])
        
        precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
        recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
        f1_t = (2 * precision_t * recall_t) / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
        
        macro_precision_list.append(precision_t)
        macro_recall_list.append(recall_t)
        macro_f1_list.append(f1_t)

    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)
    
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1


def eval_re(output_dir, execute_date = None, few_shot: bool = True):
    '''
    Get performance of the task.
    By comparing gold standard and GPT-generated data, calculate performance.
    
    execute_date = %y%m%d (string)
    
    Using fromID, toID, and type. (in addition, fromText and toText, if necessary)
    TP - correctly match all IDs and type.
    FP - Model identifies the relation not in gold standard. 
    FN - Relation exists in gold standard, but not in model result OR IDs are identical but type is different.
    '''
    # Read gold standard data
    original_files = glob.glob(os.path.join(output_dir, 'eval/re', '*.xml'))
    # Read GPT generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    model = config['openai']['model']
    
    if execute_date is None:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/re"
        else:
            date_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/re"
    else:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + execute_date + "/re"
        else:
            date_path = "output_zero_" + model + "_" + execute_date + "/re"
    path = os.path.join(output_dir, date_path)
    output_files = glob.glob(os.path.join(path, '*.xml'))
        
    ### Calculate metrics
    # preprocess
    df_original_list = []
    df_output_list = []
    for original, output in td.tqdm(zip(original_files, output_files), total=len(original_files), desc="Evaluating tRE performance", unit="files"):
        with open(original, 'r') as f:
            gold = f.read()
        with open(output, 'r') as f:
            gpt_output = f.read()
            
        # Replace unescaped special characters
        gpt_output = gpt_output.replace('&', '&amp;')
        # Remove incomplete lines
        lines = gpt_output.strip().split('\n')
        lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
        gpt_output = '\n'.join(lines)
        gpt_output = '<TAGS>\n' + gpt_output + '\n</TAGS>'
        # Remove xml tags in text snippet
        gpt_output = re.sub(r'(<EVENT.*?/EVENT>)|(<TIMEX.*?/TIMEX>)|(<EVENT|<TIMEX|</EVENT>|</TIMEX>)', '', gpt_output)
        
        ## Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        for tlink in original_root.findall("TLINK"):
            row = {
                'noteID': os.path.splitext(os.path.basename(original))[0],
                'id': tlink.get('id'),
                'fromID': tlink.get('fromID'),
                'fromText': tlink.get('fromText'),
                'toID': tlink.get('toID'),
                'toText': tlink.get('toText'),
                'type': tlink.get('type')
            }
            original_rows_by_note.append(row)
        # Append by whole corpus
        df_original_list.append(pd.DataFrame(original_rows_by_note))
        
        ## Process output list
        output_root = ET.fromstring(gpt_output)
        output_rows_by_note = []
        # Append by individual note
        for tlink in output_root.findall("TLINK"):
            row = {
                'noteID': os.path.splitext(os.path.basename(output))[0],
                'id': tlink.get('id'),
                'fromID': tlink.get('fromID'),
                'fromText': tlink.get('fromText'),
                'toID': tlink.get('toID'),
                'toText': tlink.get('toText'),
                'type': tlink.get('type')
            }
            output_rows_by_note.append(row)
        # Append by whole corpus
        df_output_list.append(pd.DataFrame(output_rows_by_note))
        
    df_original = pd.concat(df_original_list, ignore_index = True) # (175,7)
    df_output = pd.concat(df_output_list, ignore_index = True) # (128,7)
    # pd.set_option('display.max_columns', None)
    # df_original.groupby('noteID').count()
    
    merged_df = df_original.merge(df_output, on=['fromID', 'toID', 'type'], how='outer', indicator=True)

    TP = len(merged_df[merged_df['_merge'] == 'both'])
    FP = len(merged_df[merged_df['_merge'] == 'right_only'])
    FN = len(merged_df[merged_df['_merge'] == 'left_only'])
    
    ### Calculate precision, recall, and f1-score
    # Micro-averaged metrics
    micro_precision = TP / (TP + FP) if TP + FP != 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN != 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0

    # print("Micro-Precision:", micro_precision)
    # print("Micro-Recall:", micro_recall)
    # print("Micro-F1:", micro_f1)

    # For macro-average, we need to calculate metrics for each 'type' and then average them
    types = df_original['type'].unique().tolist() + df_output['type'].unique().tolist()
    types = list(set(types))  # get unique types

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    for t in types:
        TP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'both')])
        FP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'right_only')])
        FN_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'left_only')])
        
        precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
        recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
        f1_t = (2 * precision_t * recall_t) / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
        
        macro_precision_list.append(precision_t)
        macro_recall_list.append(recall_t)
        macro_f1_list.append(f1_t)

    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)

    # print("\nMacro-Precision:", macro_precision)
    # print("Macro-Recall:", macro_recall)
    # print("Macro-F1:", macro_f1)

    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1
    
def eval_nerre(output_dir: str, execute_date = None, few_shot: bool = True):
    '''
    Get performance of end-to-end approach of the task
    By comparing gold standard and GPT-generated data, calculate performacne
    
    execute_date = %y%m%d (string)
    
    Evaluation on end-to-end approach is basically identical to relation extraction.
    However, the IDs are all different from gold standard data.
    Therefore, we need to use only text to match.
    Using fromText, toText, and type. 
    TP - correctly match all text and type.
    FP - Model identified the relation not in gold standard.
    FN - Reltaion exists in gold standard, but not in model result OR entities are identical but type is different.
    '''
    # Read gold standard data
    # Currently, evaluation is not differnt from relation-extraction.
    original_files = glob.glob(os.path.join(output_dir, 'eval/re', '*.xml')) 
    # Read GPT-generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    model = config['openai']['model']
    
    if execute_date is None:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/nerre"
        else:
            date_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/nerre"
    else:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + execute_date + "/nerre"
        else:
            date_path = "output_zero_" + model + "_" + execute_date + "/nerre"
    path = os.path.join(output_dir, date_path)
    output_files = glob.glob(os.path.join(path, "*.xml"))
    
    ### Calculate metrics
    # preprocess
    df_original_list = []
    df_output_list = []
    for original, output in td.tqdm(zip(original_files, output_files), total = len(original_files), desc = "Evaluting NERRE performance", unit = "files"):
        with open(original, 'r') as f:
            gold = f.read()
        with open(output, 'r') as f:
            gpt_output = f.read()
        # Replace unescpated special characters
        gpt_output = gpt_output.replaces('&', '&amp;')
        
        ## Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        for tlink in original_root.findall("TLINK"):
            row = {
                'noteID': os.path.splitext(os.path.basename(original))[0],
                'id': tlink.get('id'),
                'fromID': tlink.get('fromID'),
                'fromText': tlink.get('fromText'),
                'toID': tlink.get('toID'),
                'toText': tlink.get('toText'),
                'type': tlink.get('type')
            }
            original_rows_by_note.append(row)
        # Append by whole corpus
        df_original_list.append(pd.DataFrame(original_rows_by_note))
        
        ## Process output list
        output_root = ET.fromstring(gpt_output)
        output_rows_by_note = []
        # Append by individual notes
        for tlink in output_root.findall("TLINK"):
            row = {
                'noteID': os.path.splitext(os.path.basename(output))[0],
                'id': tlink.get('id'),
                'fromID': tlink.get('fromID'),
                'fromText': tlink.get('fromText'),
                'toID': tlink.get('toID'),
                'toText': tlink.get('toText'),
                'type': tlink.get('type')
            }
            output_rows_by_note.append(row)
        # Append by whole corpus
        df_output_list.append(pd.DataFrame(output_rows_by_note))
        
    df_original = pd.concat(df_original_list, ignore_index=True)
    df_output = pd.concat(df_output_list, ignore_index=True)
    
    merged_df = df_original.merge(df_output, on = ['fromText','toText','type'], how = 'outer', indicator=True)
    
    TP = len(merged_df[merged_df['_merge'] == 'both'])
    FP = len(merged_df[merged_df['_merge'] == 'right_only'])
    FN = len(merged_df[merged_df['_merge'] == 'left_only'])
    
    ### Calculate precision, recall, and f1-score
    # Micro-averaged metrics
    micro_precision = TP / (TP + FP) if TP + FP != 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN != 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0

    # print("Micro-Precision:", micro_precision)
    # print("Micro-Recall:", micro_recall)
    # print("Micro-F1:", micro_f1)

    # For macro-average, we need to calculate metrics for each 'type' and then average them
    types = df_original['type'].unique().tolist() + df_output['type'].unique().tolist()
    types = list(set(types))  # get unique types

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    for t in types:
        TP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'both')])
        FP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'right_only')])
        FN_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'left_only')])
        
        precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
        recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
        f1_t = (2 * precision_t * recall_t) / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
        
        macro_precision_list.append(precision_t)
        macro_recall_list.append(recall_t)
        macro_f1_list.append(f1_t)

    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)

    # print("\nMacro-Precision:", macro_precision)
    # print("Macro-Recall:", macro_recall)
    # print("Macro-F1:", macro_f1)

    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1