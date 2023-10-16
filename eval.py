import tqdm as td
import os, glob
from datetime import date

import pandas as pd

import xml.etree.ElementTree as ET

def eval_ner():
    pass

def eval_re(output_dir, execute_date = None):
    '''
    Get performance of the task.
    By comparing gold standard and GPT-generated data, calculate performance.
    
    execute_date = %y%m%d (string)
    
    Using fromID, toID, and type. (also, fromText and toText, if necessary)
    TP - correctly match all IDs and type.
    FP - Model identifies the relation not in gold standard. 
    FN - Relation exists in gold standard, but not in model result OR IDs are identical but type is different.
    '''
    # Read gold standard data
    original_files = glob.glob(os.path.join(output_dir, 'eval/re', '*.xml'))
    # Read GPT generated output
    if execute_date is None:
        date_path = "output_" + date.today().strftime("%y%m%d") + "/re"
        path = os.path.join(output_dir, date_path)
        output_files = glob.glob(os.path.join(path, '*.xml'))
    else:
        date_path = "output_" + execute_date + "/re"
        path = os.path.join(output_dir, date_path)
        output_files = glob.glob(os.path.join(path, '*.xml'))
        
    ### Calculate metrics
    # preprocess
    df_original_list = []
    df_output_list = []
    for original, output in td.tqdm(zip(original_files, output_files), total=len(original_files), desc="Preprocess RE output", unit="files"):
        with open(original, 'r') as f:
            gold = f.read()
        with open(output, 'r') as f:
            gpt_output = f.read()
            
        # Replace unescaped special characters
        gpt_output = gpt_output.replace('&', '&amp;')
        
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
    
def eval_nerre():
    pass