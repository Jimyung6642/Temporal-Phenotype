import pandas as pd
import tqdm as td

import xml.etree.ElementTree as ET

import glob, os, logging
import re
import configparser
from datetime import date

def normalize(output_dir: str, execute_date: str, few_shot: bool = True):
    '''
    Normalize extracted entities by their temporal order
    
    Both NER & RE result is needed because temporal information is in RE, and entity type & date is in NER.
    Normalization will be performed per each entity in NER output file.
    
    execute_date: %y%m%d
    '''    
    # Read GPT-generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), 'api.config'))
    model = config['openai']['model']
    
    if execute_date is None:
        if few_shot == True:
            ner_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
            re_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/re"            
        else:
            ner_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
            re_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/re"            
    else:
        if few_shot == True:
            ner_path = "output_one_" + model + "_" + execute_date + "/ner"
            re_path = "output_one_" + model + "_" + execute_date + "/re"
        else:
            ner_path = "output_zero_" + model + "_" + execute_date + "/ner"
            re_path = "output_zero_" + model + "_" + execute_date + "/re"            
    ner = os.path.join(output_dir, ner_path)
    re = os.path.join(output_dir, re_path)
    ner_files = glob.glob(os.path.join(ner, "*.xml"))
    re_files = glob.glob(os.path.join(re, "*.xml"))
    
    # process by individual file
    ner_df = []
    re_df = []
    try:
        for ner in ner_files:
            with open(ner, 'r') as f:
                ner_output = f.read()
            
            ## post-process output
            # Replace unescaped special characters
            ner_output = ner_output.replace('&', '&amp;')
            # Remove incomplete lines
            lines = ner_output.strip().split('\n')
            lines = [line for line in lines if all(keyword in line for keyword in ('text', 'type'))]
            ner_output = '\n'.join(lines)
            # Remove complete lines
            pattern = r'<EVENT[^>]*\/>'
            matches = re.findall(pattern, ner_output)
            ner_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
            
            # parse xml
            root = ET.fromstring(ner_output)
            ner_note = []
            for event in root.findall('EVENT'):
                ner_note.append({
                    'noteID': os.path.splitext(os.path.basename(ner))[0],
                    'text': event.attrib['text'],
                    'type': event.attrib['type']
                })
        ner_df.append(pd.DataFrame(ner_note))
    except Exception as e:
        logging.error(f'Error occured while parsing ner_xml file: \n{e}')
    
    try:
        for re in re_files:
            with open(re, 'r') as f:
                re_output = f.read()
            
            ## post-process output
            # Replace unescaped special characters
            re_output = re_output.replace('&', '&amp;')
            # Remove incomplete lines
            lines = re_output.strip().split('\n')
            lines = [line for line in lines if all(keyword in line for keyword in ('toText', 'fromText', 'type'))]
            re_output = '\n'.join(lines)
            # Remove xml tags in text snippet
            re_output = re.sub(r'(<EVENT.*?/EVENT>)|(<TIMEX.*?/TIMEX>)|(<EVENT|<TIMEX|</EVENT>|</TIMEX>)', '', re_output)
            # Remove complete lines
            pattern = r'<TLINK[^>]*\/>'
            matches = re.findall(pattern, re_output)
            re_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
            
            # parse xml
            root = ET.fromstring(re_output)
            re_note = []
            for tlink in root.findall('EVENT'):
                re_note.append({
                    'noteID': os.path.splitext(os.path.basename(re))[0],
                    'fromID': tlink.get('fromID'),
                    'toID': tlink.get('toID'),
                    'fromText': tlink.get('fromText'),
                    'toText': tlink.get('toText'),
                    'type': tlink.attrib['type']
                })
        re_df.append(pd.DataFrame(re_note))
    except Exception as e:
        logging.error(f'Error occured while parsing re_xml file: \n{e}')
        
        
    
    df_corpus = pd.concat(df, ignore_index=True)
    
    
    