import pandas

import tqdm as td
import os, glob, chardet

import pandas as pd


def eval_ner():
    pass

def eval_re(output_dir, execute_date = None):
    '''
    Get performance of the task.
    By comparing gold standard and GPT-generated data, calculate performance.
    
    execute_date = %y%m%d
    
    Using fromID, toID, and type. (also, fromText and toText, if necessary)
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
        
    # Calculate metrics
    for original, output in td.tqdm(original_files, output_files, desc = "Generate RE performance", unit="files"):
        with open(original, 'rb') as f:
            gold = f.read()
        with open(output, 'rb') as f:
            gpt_output = f.read()
        
        print(gold)
        print(gpt_output)

    
def eval_nerre():
    pass