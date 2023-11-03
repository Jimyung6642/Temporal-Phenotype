import tqdm as td
import os, glob
from datetime import date
import configparser
import logging

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
    
    Relax match: found text span match
    TP - (gold standard start < model start < gold standard end) & (gold standard start < model end)
      |- (model start < gold standard start) & (gold standard < model end)
    FP & FN - same as strict match    
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
    output_files = glob.glob(os.path.join(path, "*.xml"))
    
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
        # Remove lines without text/type entities
        lines = gpt_output.strip().split('\n')
        lines = [line for line in lines if all(keyword in line for keyword in ('text', 'type'))]
        gpt_output = '\n'.join(lines)
        # Use regex to find the text attribute and then apply the replacement function
        gpt_output = re.sub(r'text="([^"]*)"', 
                            lambda match: 'text="' + match.group(1).replace('<', '&lt;').replace('>', '&gt;') + '"',
                            gpt_output)
        # Remove incomplete lines
        pattern = r'<EVENT[^>]*\/>'
        matches = re.findall(pattern, gpt_output) 
        gpt_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
        
        # Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        try:
            for event in original_root.findall("EVENT"):
                row = {
                    'noteID': os.path.splitext(os.path.basename(original))[0],
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
        except Exception as e:
            logging.error(f'Error merging output files for evaluation: \n{e}\n')
            logging.error(f'Error occurred file: {output}')
        
    df_original = pd.concat(df_original_list, ignore_index = True)
    df_output = pd.concat(df_output_list, ignore_index = True)
    
    ### Calculate precision, recall, and f1-score
    merged_df = df_original.merge(df_output, on = ['start', 'end', 'text', 'type'], how = 'outer', indicator = True)
    
    TP = len(merged_df[merged_df['_merge'] == 'both'])
    FP = len(merged_df[merged_df['_merge'] == 'right_only'])
    FN = len(merged_df[merged_df['_merge'] == 'left_only'])

    ## Exact micro metrics
    micro_precision = TP / (TP + FP) if TP + FP != 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN != 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0

    logging.info(f'Performance of named entity recognition:\n{path}')
    logging.info(f'================================')
    logging.info(f'Overall exact match micro metrics...')
    logging.info(f'exact match micro-precission: {round(micro_precision, 3)}')
    logging.info(f'exact match micro-recall: {round(micro_recall, 3)}')
    logging.info(f'exact match micro-f1: {round(micro_f1, 3)}')

    # Define a function to check for an exact match across all fields
    def is_exact_match(row, df_to_compare, type):
        # Filter the comparison dataframe for the current type
        df_filtered = df_to_compare[df_to_compare['type'] == type]
        # Check if there's any row in the filtered dataframe that matches all criteria
        return any(
            (df_filtered['start'] == row['start']) &
            (df_filtered['end'] == row['end']) &
            (df_filtered['text'] == row['text']) &
            (df_filtered['type'] == row['type'])
        )

    # For macro-average, we need to calculate metrics for each 'type' and then average them
    # types = df_original['type'].unique().tolist() + df_output['type'].unique().tolist()
    types = df_original['type'].unique().tolist()
    types = list(set(types))  # get unique types
    
    # Initialize lists to hold the precision, recall, and F1 for each type
    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    # Iterate over each unique type
    for t in types:
        # Calculate TP, FP, and FN for each type
        TP_t = sum(df_original[df_original['type'] == t].apply(is_exact_match, axis=1, df_to_compare=df_output, type=t))
        FP_t = sum(df_output[df_output['type'] == t].apply(is_exact_match, axis=1, df_to_compare=df_original, type=t))
        FN_t = len(df_original[(df_original['type'] == t)]) - TP_t
        
        # Calculate precision, recall, and F1 for each type
        precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
        recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
        
        # Append the metrics to their respective lists
        macro_precision_list.append(precision_t)
        macro_recall_list.append(recall_t)
        macro_f1_list.append(f1_t)
        
        logging.info(f'Exact macro relation type: {t}...')
        logging.info(f'exact match macro precision: {round(precision_t, 3)}')
        logging.info(f'exact match macro recall: {round(recall_t, 3)}')
        logging.info(f'exact match macro f1-score: {round(f1_t, 3)}')

    # Calculate the macro-averaged metrics
    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)
    
    logging.info(f'Overall exact macro performance...')
    logging.info(f'exact match macro precision: {round(macro_precision, 3)}')
    logging.info(f'exact match macro recall: {round(macro_recall, 3)}')
    logging.info(f'exact match macro f1-score: {round(macro_f1, 3)}')
    
    ## Relax match
    # macro metrics
    # Define a function to check for overlap and text match
    def is_relax_macro_match(gold_start, gold_end, gold_text, gold_type, model_start, model_end, model_text, model_type):
        # Check for the two overlap conditions you've described
        condition1 = (gold_start < model_start < gold_end) and (gold_start < model_end)
        condition2 = (model_start < gold_start) and (gold_start < model_end)
        # Check for text and type match
        text_type_match = (gold_text == model_text) and (gold_type == model_type)
        return (condition1 or condition2) and text_type_match

    # Initialize counts
    TP = 0
    FP = 0
    FN = 0

    # Iterate over each row in the gold standard dataframe
    # This approach might be too slow but better for memory-saving
    for index, gold_row in td.tqdm(df_original.iterrows(), desc='Calculating TP/FN for relax macro metrics...', total=df_original.shape[0]):
        # Find any matching annotations in the model's output
        matches = df_output.apply(lambda model_row: is_relax_macro_match(
            gold_row['start'], gold_row['end'], gold_row['text'], gold_row['type'],
            model_row['start'], model_row['end'], model_row['text'], model_row['type']
        ), axis=1)
        
        # If there's at least one match, it's a TP; otherwise, it's an FN
        if matches.any():
            TP += 1
        else:
            FN += 1

    # Any annotation in the model's output that doesn't match with the gold standard is a FP
    for index, model_row in td.tqdm(df_output.iterrows(), desc='Calculating FP for relax macro metrics...', total=df_output.shape[0]):
        matches = df_original.apply(lambda gold_row: is_relax_macro_match(
            gold_row['start'], gold_row['end'], gold_row['text'], gold_row['type'],
            model_row['start'], model_row['end'], model_row['text'], model_row['type']
        ), axis=1)
        
        # If there's no match, it's a FP
        if not matches.any():
            FP += 1

    # Micro-averaged metrics
    relax_micro_precision = TP / (TP + FP) if TP + FP != 0 else 0
    relax_micro_recall = TP / (TP + FN) if TP + FN != 0 else 0
    relax_micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0

    logging.info(f'Overall relax match micro metrics...')
    logging.info(f'relax match micro-precission: {round(relax_micro_precision, 3)}')
    logging.info(f'relax match micro-recall: {round(relax_micro_recall, 3)}')
    logging.info(f'relax match micro-f1: {round(relax_micro_f1, 3)}')
    
    # Define a function to check for a relaxed match
    def is_relaxed_micro_match(gold_row, model_row):
        # Check for the two overlap conditions you've described
        condition1 = (gold_row['start'] < model_row['start'] < gold_row['end']) and (gold_row['start'] < model_row['end'])
        condition2 = (model_row['start'] < gold_row['start']) and (gold_row['start'] < model_row['end'])
        # Check for text and type match
        text_type_match = (gold_row['text'] == model_row['text']) and (gold_row['type'] == model_row['type'])
        return (condition1 or condition2) and text_type_match

    # Initialize lists to hold the precision, recall, and F1 for each type
    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    # Iterate over each unique type
    types = df_original['type'].unique().tolist()
    types = list(set(types))  # get unique types
    for t in types:
        # Filter the dataframes for the current type
        df_original_type = df_original[df_original['type'] == t]
        df_output_type = df_output[df_output['type'] == t]

        # Calculate TP, FP, and FN for each type using relaxed match
        TP_t = sum(df_output_type.apply(lambda model_row: df_original_type.apply(lambda gold_row: is_relaxed_micro_match(gold_row, model_row), axis=1).any(), axis=1))
        FN_t = sum(df_original_type.apply(lambda gold_row: not df_output_type.apply(lambda model_row: is_relaxed_micro_match(gold_row, model_row), axis=1).any(), axis=1))
        FP_t = sum(df_output_type.apply(lambda model_row: not df_original_type.apply(lambda gold_row: is_relaxed_micro_match(gold_row, model_row), axis=1).any(), axis=1))

        if not TP_t == 0: 
            # Calculate precision, recall, and F1 for each type
            precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
            recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
            f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
            
            # Append the metrics to their respective lists
            macro_precision_list.append(precision_t)
            macro_recall_list.append(recall_t)
            macro_f1_list.append(f1_t)

            # Print the metrics for the current type
            logging.info(f"Relax macro relation type: {t}...")
            logging.info(f"relax match macro precision: {round(precision_t, 3)}")
            logging.info(f"relax match macro recall: {round(recall_t, 3)}")
            logging.info(f"relax match macro f1-score: {round(f1_t, 3)}\n")
        else:
            logging.info(f'passing type: {t} for not having TP')

    # Calculate the macro-averaged metrics
    relax_macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    relax_macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    relax_macro_f1 = sum(macro_f1_list) / len(macro_f1_list)

    # Output the macro-averaged metrics
    logging.info(f"Overall relax macro performance...")
    logging.info(f"relax match macro precision: {round(relax_macro_precision, 3)}")
    logging.info(f"relax match macro recall: {round(relax_macro_recall, 3)}")
    logging.info(f"relax match macro f1-score: {round(relax_macro_f1, 3)}")
    
    return merged_df


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
        # Remove xml tags in text snippet
        gpt_output = re.sub(r'(<EVENT.*?/EVENT>)|(<TIMEX.*?/TIMEX>)|(<EVENT|<TIMEX|</EVENT>|</TIMEX>)', '', gpt_output)
        # Remove complete lines
        pattern = r'<TLINK[^>]*\/>'
        matches = re.findall(pattern, gpt_output)
        gpt_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
        
        ## Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        try:
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
        except Exception as e:
            logging.error(f'Error merging output files for evaluation: \n{e}\n')
            logging.error(f'Error occrred file: {output}')
        
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

    logging.info(f'Performance of temporal relation extraction:\n{path}')
    logging.info(f'================================')
    logging.info(f'micro-precission: {micro_precision}')
    logging.info(f'micro-recall: {micro_recall}')
    logging.info(f'micro-f1: {micro_f1}')

    # For macro-average, we need to calculate metrics for each 'type' and then average them
    types = df_original['type'].unique().tolist() + df_output['type'].unique().tolist()
    types = list(set(types))  # get unique types

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []

    for t in types:
        TP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'both')])
        FP_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'right_only')])
        FN_t = len(merged_df[(merged_df['type'] == t) & (merged_df['_merge'] == 'left_only')])
        
        if not TP_t == 0: 
            precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t != 0 else 0
            recall_t = TP_t / (TP_t + FN_t) if TP_t + FN_t != 0 else 0
            f1_t = (2 * precision_t * recall_t) / (precision_t + recall_t) if precision_t + recall_t != 0 else 0
            
            macro_precision_list.append(precision_t)
            macro_recall_list.append(recall_t)
            macro_f1_list.append(f1_t)
            
            logging.info(f'Relation type: {t}...')
            logging.info(f'macro precision: {precision_t}...')
            logging.info(f'macro recall: {recall_t}...')
            logging.info(f'macro f1-score: {f1_t}')
        else:
            logging.info(f'pass type: {t} for not having TP')

    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)

    logging.info(f'Overall macro performance...')
    logging.info(f'macro precision: {macro_precision}')
    logging.info(f'macro recall: {macro_recall}')
    logging.info(f'macro f1: {macro_f1}')
    return merged_df
    
def eval_nerre(output_dir: str, execute_date = None, few_shot: bool = True):
    '''
    Get performance of end-to-end approach of the task
    By comparing gold standard and GPT-generated data, calculate performacne.
    GPT will generate both NER and RE, but we will evaluate the RE performance here since it uses NER information.
    
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
        
        # Replace unescaped special characters
        gpt_output = gpt_output.replace('&', '&amp;')
        # Remove incomplete lines
        lines = gpt_output.strip().split('\n')
        lines = [line for line in lines if all(keyword in line for keyword in ('toText', 'fromText', 'type'))]
        gpt_output = '\n'.join(lines)
        # Remove xml tags in text snippet
        gpt_output = re.sub(r'(<EVENT.*?/EVENT>)|(<TIMEX.*?/TIMEX>)|(<EVENT|<TIMEX|</EVENT>|</TIMEX>)', '', gpt_output)
        # Use regex to find the fromText/toText attributes and then apply the replacement function
        gpt_output = re.sub(r'fromText="([^"]*)"', 
                            lambda match: 'fromText="' + match.group(1).replace('<', '&lt;').replace('>', '&gt;') + '"',
                            gpt_output)
        gpt_output = re.sub(r'toText="([^"]*)"', 
                            lambda match: 'tpText="' + match.group(1).replace('<', '&lt;').replace('>', '&gt;') + '"', 
                            gpt_output)
        # Remove complete lines
        pattern = r'<TLINK[^>]*\/>'
        matches = re.findall(pattern, gpt_output)
        gpt_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
        
        ## Process original list
        original_root = ET.fromstring(gold)
        original_rows_by_note = []
        # Append by individual note
        try:
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
        except Exception as e:
            logging.error(f'error merging output files for evaluation: \n{e}\n')
            logging.error(f'error occurred file: {output}')
        
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

    logging.info(f'Performance of end-to-end temporal relation extraction:\n{path}')
    logging.info(f'================================')
    logging.info(f'micro-precission: {micro_precision}')
    logging.info(f'micro-recall: {micro_recall}')
    logging.info(f'micro-f1: {micro_f1}')

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
        
        logging.info(f'Relation type: {t}...')
        logging.info(f'macro precision: {precision_t}')
        logging.info(f'macro recall: {recall_t}')
        logging.info(f'macro f1-score: {f1_t}')

    macro_precision = sum(macro_precision_list) / len(macro_precision_list)
    macro_recall = sum(macro_recall_list) / len(macro_recall_list)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list)

    logging.info(f'Overall macro performance...')
    logging.info(f'macro precision: {macro_precision}')
    logging.info(f'macro recall: {macro_recall}')
    logging.info(f'macro f1: {macro_f1}')
    return merged_df