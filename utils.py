import pandas as pd
import tqdm as td
import networkx as nx
from networkx.algorithms.dag import topological_sort

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import glob, os, logging
import re as reg
import configparser
from datetime import date

# Function to find the order of EVENT IDs with certainty based on relationship types
def find_event_order_with_certainty(df):
    # Dictionary to hold the relations with their type
    relations = {}
    for _, row in df.iterrows():
        fromID, toID, rel_type = row['fromID'], row['toID'], row['type']
        if fromID not in relations:
            relations[fromID] = []
        if toID not in relations:
            relations[toID] = []
        relations[fromID].append((toID, rel_type))
        relations[toID].append((fromID, rel_type))

    # Dictionary to store the certainty of each event
    certainty_dict = {}

    # Finding the order of EVENT IDs using DFS (Depth-First Search algorithm)
    visited = set()
    order = []

    # In DFS, note will be event IDs and edges will be relation types
    def dfs(node, is_certain):
        if node not in visited:
            visited.add(node)
            order.append(node)
            certainty_dict[node] = "Certain" if is_certain else "Uncertain"
            for neighbour, rel_type in relations.get(node, []):
                # If the relationship is BEFORE or AFTER, it adds certainty
                new_certainty = is_certain or rel_type in ["BEFORE", "AFTER"]
                dfs(neighbour, new_certainty)

    # Starting DFS from each unvisited node
    for node in df['fromID'].tolist() + df['toID'].tolist():
        if node not in visited:
            dfs(node, False)
            
    # Creating the DataFrame with EVENT and CERTAINTY
    event_certainty_df = pd.DataFrame({"id": order, "certainty": [certainty_dict[event] for event in order]})
    event_certainty_df

    return event_certainty_df

def normalize(output_dir: str, execute_date: str, few_shot: bool = True):
    '''
    Normalize extracted entities by their temporal order
    
    Both NER & RE result is needed because temporal information is in RE, and entity type & date is in NER.
    Normalization will be performed per each entity in NER output file.
    
    execute_date: %y%m%d
    '''    
    ## Temp for the dev
    os.chdir('./Temporal-Phenotype')
    output_dir = './result'
    ner = os.path.join(output_dir, 'eval/ner')
    re = os.path.join(output_dir, 'eval/re')
    
    # Read GPT-generated output
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), 'api.config'))
    model = config['openai']['model']
    
    # if execute_date is None:
    #     if few_shot == True:
    #         ner_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
    #         re_path = "output_one_" + model + "_" + date.today().strftime("%y%m%d") + "/re"            
    #     else:
    #         ner_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/ner"
    #         re_path = "output_zero_" + model + "_" + date.today().strftime("%y%m%d") + "/re"            
    # else:
    #     if few_shot == True:
    #         ner_path = "output_one_" + model + "_" + execute_date + "/ner"
    #         re_path = "output_one_" + model + "_" + execute_date + "/re"
    #     else:
    #         ner_path = "output_zero_" + model + "_" + execute_date + "/ner"
    #         re_path = "output_zero_" + model + "_" + execute_date + "/re"
    # ner = os.path.join(output_dir, ner_path)
    # re = os.path.join(output_dir, re_path)
    ner_files = glob.glob(os.path.join(ner, "*.xml"))
    re_files = glob.glob(os.path.join(re, "*.xml"))
    
    # ## For devel purpose, use eval files
    # ner_files = glob.glob('./result/eval/ner/*.xml')
    # re_files = glob.glob('./result/eval/re/*.xml')
    
    ## Process NER and RE results and transform as tabular form.
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
            matches = reg.findall(pattern, ner_output)
            ner_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
            
            # parse xml            
            root = ET.fromstring(ner_output)
            ner_note = []
            for event in root.findall('EVENT'):
                # Standardize time text to normalized value
                if any(type_attr in event.get("type") for type_attr in ['DATE', 'DURATION', 'FREQUENCY', 'TIME']):
                    event.set("text", event.get("val"))  # Replace text with val                
                    
                ner_note.append({
                    'noteID': os.path.splitext(os.path.basename(ner))[0],
                    'id': event.attrib['id'],
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
            re_output = reg.sub(r'(<EVENT.*?/EVENT>)|(<TIMEX.*?/TIMEX>)|(<EVENT|<TIMEX|</EVENT>|</TIMEX>)', '', re_output)
            # Remove complete lines
            pattern = r'<TLINK[^>]*\/>'
            matches = reg.findall(pattern, re_output)
            re_output = '<TAGS>\n' + '\n'.join(matches) + '\n</TAGS>'
            
            # parse xml
            root = ET.fromstring(re_output)
            re_note = []
            for tlink in root.findall('TLINK'):
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
        
    ner_df = pd.concat(ner_df, ignore_index=True)
    re_df = pd.concat(re_df, ignore_index=True)
    
    # Replace temporal expression in re_df using ner_df
    id_text_map = pd.Series(ner_df.text.values, index=ner_df.id).to_dict()
    re_df['fromText'] = re_df['fromID'].map(id_text_map)
    re_df['toText'] = re_df['toID'].map(id_text_map)
    
    # Extract Time entities from ner_df
    absolute_temp = ner_df[ner_df['type'].isin(['DATE','TIME'])]
    relative_temp = ner_df[ner_df['type'].isin(['DURATION','FREQUENCY'])]
    event = ner_df[ner_df['type'].isin(['TEST','TREATMENT','PROBLEM'])]
       
    # Identify DATE entity set as baseline temp
    date = ner_df[ner_df['type'] =='DATE']
    tmp_re = re_df[re_df['noteID']=='298']
    tmp_re = re_df[re_df['noteID']=='1']    
    tmp_ner = ner_df[ner_df['noteID']=='1']
    
    # Find the order of EVENT IDs with certainty
    event_certainty_df = find_event_order_with_certainty(tmp_re)
    event_certainty_df.merge(tmp_ner, on = "id", how='left')    

    
            
    ### Identify potential temporal expression from re_df, use DATE as baseline
    ## Process absolute temp relation
    # EVENT to TIME
    # TIME to EVENT
    # EVENT to EVENT
    ## Process relative temp relation
    # EVENT to TIME
    # TIME to EVENT
    # EVENT to EVENT
    ## Caculate date range of the events