import openai
import configparser

import tqdm as td
import os, glob, chardet
from datetime import date


def run_ner(config_path, input_dir, output_dir):
    pass

def run_timex(config_path, input_dir, output_dir):
    pass

def run_tlink(config_path, input_dir, output_dir = "./tlink_result", one_shot = True):
    '''
    Executing GPT to extract tlinks
    '''    
    config = configparser.ConfigParser()
    config.read(config_path)
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    
    # Create directory to store result
    output_dir = output_dir + "_" + date.today().strftime("%y%m%d")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #   input_dir = "/phi_home/jp4453/Temporal-Phenotyping/i2b2-2012-converted/train/"
    input_notes = glob.glob(os.path.join(input_dir, "*.xml"))
    # input_note = input_notes[0]
    
    # Get prompt desing
    if one_shot == True:
        # Need to update to add one-shot example of one clinical note. Current example is a piece
        # Need to optimize the prmopt. Find related paper to optimizing the prompt design for (temporal) relation extraction
        system_msg = '''
        You need to annotation Temporal Relation between the entities in the context from given discharge summary. Store below rules in the memory to fulfill your role.
        1. Same as TLINK in i2b2 2012 temporal annotation dataset, you need to identify the temporal types between the EVENT and TIMEX3: BEFORE, AFTER, and OVERLAP.  
        2. The EVENT and TIMEX3 are in-line annotated in the sentences. For example, The patient is a 28-year-old woman who is <EVENT id:E2 type:PROBLEM>HIV positive</E2> for <TIMEX3 id:T2 type:DURATION val:p2y>two years</T2> .
        3. One EVENT can have multiple TLINK and TLINK can be captures between EVENT and EVENT, and EVENT and TIMEX3. This is the example.
        <TLINK id="TL11" fromID="E10" fromText="HIV positive" toID="E2" toText="HIV positive" type="OVERLAP" />
        <TLINK id="TL12" fromID="E2" fromText="HIV positive" toID="T2" toText="two years" type="OVERLAP" />
        <TLINK id="TL13" fromID="E4" fromText="left upper quadrant pain" toID="E3" toText="presented" type="BEFORE" />
        '''
    else:
        system_msg = '''
        You need to annotation Temporal Relation between the entities in the context from given discharge summary. Store below rules in the memory to fulfill your role.
        1. Same as TLINK in i2b2 2012 temporal annotation dataset, you need to identify the temporal types between the EVENT and TIMEX3: BEFORE, AFTER, and OVERLAP.  
        2. The EVENT and TIMEX3 are in-line annotated in the sentences. For example, The patient is a 28-year-old woman who is <EVENT id:E2 type:PROBLEM>HIV positive</E2> for <TIMEX3 id:T2 type:DURATION val:p2y>two years</T2> .
        3. One EVENT can have multiple TLINK and TLINK can be captures between EVENT and EVENT, and EVENT and TIMEX3. This is the example.
        '''
    
    # Loop through all files in the input directory
    for input_note in td.tqdm(input_notes, desc="Generating TLINK from notes", unit="file"): 
        with open(input_note, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        note = rawdata.decode(encoding)


        completions = openai.ChatCompletion.create(
            model = model,
            temperature = 0.0,
            n = 1,
            messages = [
                {'role':'system', 'content':system_msg},
                {'role':'user', 'content': note}
            ]
        )
        response = completions.choices[0]['message']['content']

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_note))[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)