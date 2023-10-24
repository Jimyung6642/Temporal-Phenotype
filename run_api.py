import openai
import configparser

import tqdm as td
import os, glob, chardet
from datetime import date
import re


def run_ner(output_dir: str, few_shot: bool = True):
    '''
    Do named entity recognition - problem, test, treatment
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + "one_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/ner"
    else:
        date_path = "output_" + "zero_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/ner"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read NER input data
    notes = glob.glob(os.path.join(output_dir, 'data/ner', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['NER']['few_prompt']
        few_user = config['RE']['few_user']
        few_assistant = config['NER']['few_assistant']
        
        for note in td.tqdm(notes, desc = "Generating NER output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                print('output exists %s' % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                # Call GPT API
                try:
                    completions = openai.ChatCompletion.create(
                        model = model,
                        temperature = temp,
                        n = 1,
                        messages = [
                            {'role':'system', 'content':system_msg},
                            {'role':'user', 'content':few_user},
                            {'role':'assistant', 'content':few_assistant},
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
                    
                # Remove incomplete reponse
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toText', 'fromText', 'type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '\n</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                    
    else: # zero_shot
        system_msg = config['NER']['zero_promt']
        
        for note in td.tqdm(notes, desc = "Generating NER output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                print('output exists: %s' % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                    
                try:
                    completions = openai.ChatCompletion.create(
                        model = model,
                        temperature = temp,
                        n = 1,
                        messages = [
                            {'role':'system', 'content': system_msg},
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
                
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toText', 'fromText', 'type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '\n</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding = 'utf-8') as f:
                    f.write(response)
    

def run_re(output_dir: str, few_shot: bool = True):
    '''
    Do temporal relation extraction
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + 'one_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/re"
    else:
        date_path = "output_" + 'zero_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/re"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    # Read RE input data
    notes = glob.glob(os.path.join(output_dir, 'data/re', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['RE']['few_prompt']
        few_user = config['RE']['few_user']
        few_assistant = config['RE']['few_assistant']
        
        for note in td.tqdm(notes, desc="Generating RE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                print('output exists: %s' % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
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
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
                    
                # Remove the last XML entity if it doesn't have toID, fromID, or type.
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '\n</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)
        
    else:
        system_msg = config['RE']['zero_prompt']
        
        for note in td.tqdm(notes, desc="Generating RE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                print('output exists: %s' % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                # GPT API call
                try:
                    completions = openai.ChatCompletion.create(
                        model = model,
                        temperature = temp,
                        n = 1,
                        messages = [
                            {'role':'system', 'content':system_msg},
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
                
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '\n</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)

def run_nerre(output_dir: str, few_shot: bool = True):
    '''
    Do end-to-end relation extraction
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + 'one_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/nerre"
    else:
        date_path = "output_" + 'zero_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/nerre"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    # Read NERRE input data
    notes = glob.glob(os.path.join(output_dir, 'data/nerre', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['NERRE']['few_prompt']
        few_user = config['NERRE']['few_user']
        few_assistant = config['NERRE']['few_assistant']
        
        for note in td.tqdm(notes, desc="Generating NERRE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + ".xml")):
                print("output exists: %s" % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                    
                try:
                    completions = openai.ChatCompletion.create(
                        model = model,
                        temperature = temp,
                        n = 1,
                        messages = [
                            {'role':'system', 'content':system_msg},
                            {'role':'user', 'content':few_user},
                            {'role':'assistant', 'content':few_assistant},
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
                    
                # Remove incomplete responses
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding = 'utf-8') as f:
                    f.write(response)
    
    else:
        system_msg = config['NERRE']['zero_prompt']
        
        for note in td.tqdm(notes, desc="Generating NERRE output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                print('output exists: %s' % os.path.splitext(os.path.basename(note))[0])
            else:
                with open(note, 'r') as f:
                    content = f.read()
                    
                try:
                    completions = openai.ChatCompletion.create(
                        model = model,
                        temperature = temp,
                        n = 1,
                        messages = [
                            {'role':'system', 'content':system_msg},
                            {'role':'user', 'content':content}
                        ]
                    )
                    response = completions.choices[0]['message']['content']
                except Exception as e:
                    print(e)
            
                lines = response.strip().split('\n')
                lines = [line for line in lines if all(keyword in line for keyword in ('toID','fromID','type'))]
                response = '\n'.join(lines)
                response = '<TAGS>\n' + response + '\n</TAGS>'
                
                output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)