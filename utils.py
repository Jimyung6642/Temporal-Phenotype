import pandas as pd
import tqdm as td

import xml.etree.ElementTree as ET

import glob, os, logging, re
import configparser
from datetime import date

def normalize(output_dir: str, execute_date: str, few_shot: bool = True):
    '''
    Normalize extracted entities by their temporal order
    
    execute_date: %y%m%d
    '''    
    # Read GPT-generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), 'api.config'))
    model = config['openai']['model']
    
    if execute_date is None:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
        else:
            date_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
    else:
        if few_shot == True:
            date_path = "output_one_" + model + "_" + execute_date + "/re"
        else:
            date_path = "output_zero_" + model + "_" + execute_date + "/re"
    path = os.path.join(output_dir, date_path)
    output_files = glob.glob(os.path.join(path, "*.xml"))
    
    # process by individual file
    df = []
    try:
        for output in td.tqdm(output_files, desc="Normalizing the output", unit = "file"):
            with open(output, 'r') as f:
                gpt_output = f.read()
            
            ## post-process output
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
            
            # parse xml
            root = ET.fromstring(gpt_output)
            note = []
            for tlink in root.findall('TLINK'):
                note.append({
                    'noteID': os.path.splitext(os.path.basename(output))[0],
                    'fromID': tlink.attrib['fromID'],
                    'fromText': tlink.attrib['fromText'],                
                    'toID': tlink.attrib['toID'],
                    'toText': tlink.attrib['toText'],
                    'type': tlink.attrib['type']
                })
        df.append(pd.DataFrame(note))
    except Exception as e:
        logging.error(f'Error occured while parsing xml file: \n{e}')
    
    df_corpus = pd.concat(df, ignore_index=True)
    
    
    