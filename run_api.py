import openai
import configparser

import tqdm as td
import os, glob, chardet
from datetime import date
import re


def run_ner(input_dir, output_dir, few_shot = True):
    pass

def run_re(output_dir, few_shot = True):
    '''
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder store output
    date_path = "output_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/re"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    # Read NER input data
    notes = glob.glob(os.path.join(output_dir, 'data/re', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['RE']['few_prompt']
        few_user = config['RE']['few_user']
        few_assistant = config['RE']['few_assistant']
        
        for note in td.tqdm(notes, desc="Generate RE output from i2b2", unit="files"):
            with open(note, 'r') as f:
                context = f.read()
            
            try:
                # GPT API call
                completions = openai.ChatCompletion.create(
                    model = model,
                    temperature = temp,
                    n = 1,
                    messages = [
                        {'role':'system', 'content':system_msg},
                        {'role':'user', 'content':few_user},
                        {'role':'assistant', 'content':few_assistant},
                        {'role':'user', 'content':context}
                    ]
                )
                response = completions.choices[0]['message']['content']
            except Exception as e:
                print(e)
            
            # Add XML tags. May need to consider processing only the last TLINK sentence.
            if re.search(r'"\s?/>$', response[-25:]):
                response = '<TAGS>\n' + response + '\n</TAGS>'
            else: 
                response = '<TAGS>\n' + response + '" />' + '\n</TAGS>'
            
            output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
        
    else:
        system_msg = config['RE']['zero_prompt']
        
        for note in td.tqdm(notes, desc="Generate RE output from i2b2", unit="files"):
            with open(note, 'r') as f:
                context = f.read()
            
            # GPT API call
            completions = openai.ChatCompletion.create(
                model = model,
                temperature = temp,
                n = 1,
                messages = [
                    {'role':'system', 'content':system_msg},
                    {'role':'user', 'content':context}
                ]
            )
            response = completions.choices[0]['message']['content']
            # Add XML tags. May need to consider processing only the last TLINK sentence.
            if re.search(r'"\s?/>$', response[-25:]):
                response = '<TAGS>\n' + response + '\n</TAGS>'
            else: 
                response = '<TAGS>\n' + response + '" />' + '\n</TAGS>'
            
            output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)

def run_nerre():
    pass