import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


import glob
import logging
import tqdm as td

import chardet
# Need to add merge_data for all function

def generate_input_data(input_dir, output_dir):
    '''
    input - i2b2 corpus with train/test folders
    result -- input data - NER; RE; NER+RE;
            |- output data - NER; RE; NRE+RE;
            |- eval data (gold standard) - NER; RE; NER+RE;
    
    Convert i2b2 corpus into input-format data:
    NER - text
    RE - text with in-line NER annotaiton
    NER+RE - text
    
    Only EVENTs (problem, test, and treatment) with positive, factual, and TIMEX3 will be extracted.
    '''
    # Create the data saving paths
    # input
    # input_dir = "/phi_home/jp4453/Temporal-Phenotype/i2b2-2012-original"    
    # output_dir = "/phi_home/jp4453/Temporal-Phenotype/result"
    subdirs = ['data/ner', 'data/re', 'data/nerre']
    for subdir in subdirs:
        path = os.path.join(output_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    train_files = glob.glob(os.path.join(input_dir, 'train/*.xml'))
    test_files = glob.glob(os.path.join(input_dir, 'test/*.xml'))
    xml_files = train_files + test_files
        
    ### Generate NER input data
    logging.info('Generating NER input data...')
    for xml_file in td.tqdm(xml_files, desc="Generating NER input data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)
        xml_data = xml_data.replace('&', '&amp;')
        
        root = ET.fromstring(xml_data)
        i2b2 = root.find('TEXT').text
        
        output_file = os.path.join(os.path.join(output_dir, 'data/ner'), os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(i2b2)
     
    ### Generate NER-RE input data
    logging.info('Generating NER-RE input data...')
    for xml_file in td.tqdm(xml_files, desc="Generating NER-RE input data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)
        xml_data = xml_data.replace('&', '&amp;')
        
        root = ET.fromstring(xml_data)
        i2b2 = root.find('TEXT').text
        
        output_file = os.path.join(os.path.join(output_dir, 'data/nerre'), os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(i2b2)

    ### Generate RE input data
    logging.info('Generating RE input data...')
    train_files = glob.glob(os.path.join(input_dir, 'train/*.xml'))
    test_files = glob.glob(os.path.join(input_dir, 'test/*.xml'))
    xml_files = train_files + test_files
    for xml_file in td.tqdm(xml_files, desc="Generating RE input data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)

        # Replace unescaped special characters
        xml_data = xml_data.replace('&', '&amp;')
         # Parse the XML string
        root = ET.fromstring(xml_data)
        # root = etree.fromstring
        i2b2 = root.find('TEXT').text
        
        # Extract EVENT and TIMEX3 annotations and sort by start offset
        annotations = []
        for event in root.findall(".//EVENT"): 
            event_modality = event.attrib.get('modality', '')
            event_polarity = event.attrib.get('polarity', '')
            event_type = event.attrib.get('type', '').lower()  # converting type to lowercase for comparison
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test', 'occurrence']:
                annotations.append((int(event.attrib['start']), int(event.attrib['end']), f'<EVENT id:"{event.attrib["id"]}" type:"{event.attrib["type"]}">', f'</EVENT>'))
        for timex in root.findall(".//TIMEX3"):
            annotations.append((int(timex.attrib['start']), int(timex.attrib['end']), f'<TIMEX3 id:"{timex.attrib["id"]}" type:"{timex.attrib["type"]}" val:"{timex.attrib["val"]}">', f'</TIMEX3>'))
        annotations.sort(key=lambda x: x[0])
        
        # Replace the portions of the main text with the annotations
        offset = 0    
        converted_text = i2b2            
        for start, end, type, closer in annotations:
            converted_text = converted_text[:+start+offset] + type + converted_text[+start+offset:]
            offset += len(type)
            converted_text = converted_text[:+end+offset] + closer + converted_text[+end+offset:]
            offset += len(closer)
            
        # Write the converted data to the output directory
        output_file = os.path.join(os.path.join(output_dir, 'data/re'), os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(converted_text)

def generate_eval_data(input_dir, output_dir):
    '''
    input - i2b2 corpus with train/test folders
    result -- input data - NER; RE; NER+RE;
            |- output data - NER; RE; NRE+RE;
            |- eval data (gold standard) - NER; RE; NER+RE;
    
    Only EVENTs (problem, test, and treatment) with positive, factual, and TIMEX3 will be extracted.
   
    Generate eval datasets for NER, RE, NER-RE
    For RE, only TLINK of PROBLEM, TEST, TREATMENT, and TIMEX3 will be extracted
    '''
    subdir = ['eval/ner', 'eval/re', 'eval/nerre']
    for subdir in subdir:
        path = os.path.join(output_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    train_files = glob.glob(os.path.join(input_dir, 'train/*.xml'))
    test_files = glob.glob(os.path.join(input_dir, 'test/*.xml'))
    xml_files = train_files + test_files        
            
    ### Generate NER eval data    
    logging.info('Generating NER eval data...')
    for xml_file in td.tqdm(xml_files, desc="Generating NER eval data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)
        
        # Replace unescaped special characters
        xml_data = xml_data.replace('&', '&amp;')
        # Parse the XML string
        root = ET.fromstring(xml_data)
        
        # Extract EVENT and TIMEX3 annotations and sort by start offset
        target_events = []
        for event in root.findall(".//EVENT"):
            event_modality = event.attrib.get('modality', '')
            event_polarity = event.attrib.get('polarity', '')
            event_type = event.attrib.get('type', '').lower()  # converting type to lowercase for comparison
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test', 'occurrence']:
                target_events.append(event.attrib)
        for timex in root.findall(".//TIMEX3"):
            target_events.append(timex.attrib)
        
        # Iterating through the target_tlink list and creating XML sub-elements
        root2 = ET.Element("TAGS")
        # for item in target_events:
        #     events = ET.SubElement(root2, 'EVENT', item)
        # Creating the XML tree with the root2 element
        tree = ET.ElementTree(root2)
        rough_string = ET.tostring(tree.getroot(), 'utf-8')
        reparsed = minidom.parseString(rough_string)
        reparsed = reparsed.toprettyxml(indent="  ")        
    
        output_file = os.path.join(os.path.join(output_dir, 'eval/ner'), os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reparsed)
        
    ### Generate RE eval data    
    for xml_file in td.tqdm(xml_files, desc="Generating RE eval data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)
        
        # Replace unescaped special characters
        xml_data = xml_data.replace('&', '&amp;')
        # Parse the XML string
        root = ET.fromstring(xml_data)                
        
        # Extract EVENT and TIMEX3 annotations and sort by start offset
        target_events = []
        for event in root.findall(".//EVENT"):
            event_modality = event.attrib.get('modality', '')
            event_polarity = event.attrib.get('polarity', '')
            event_type = event.attrib.get('type', '').lower()  # converting type to lowercase for comparison
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test', 'occurrence']:
                target_events.append(event.attrib['id'])
        for timex in root.findall(".//TIMEX3"):
            target_events.append(timex.attrib['id'])
        target_events.sort(key=lambda x: x[0])
                
        event = list(filter(lambda x:'E' in x, target_events))
        timex = list(filter(lambda x:'T' in x, target_events))
        
        target_tlink = []
        # Extract related TLINK of target event/timex
        for tlink in root.findall(".//TLINK"):
            # EVENT to EVENT
            if (tlink.attrib["fromID"] in event and tlink.attrib["toID"] in event) and tlink.attrib["id"].startswith('TL'):
                target_tlink.append(tlink.attrib)
            # EVENT to TIME X
            if (tlink.attrib["fromID"] in event and tlink.attrib["toID"] in timex) and tlink.attrib["id"].startswith('TL'):
                target_tlink.append(tlink.attrib)
            # TIMEX to EVENT
            if (tlink.attrib["fromID"] in timex and tlink.attrib["toID"] in event) and tlink.attrib["id"].startswith('TL'):
                target_tlink.append(tlink.attrib)
                        
        root2 = ET.Element("TAGS")
        
        # Iterating through the target_tlink list and creating XML sub-elements
        for item in target_tlink:
            tlink = ET.SubElement(root2, 'TLINK', item)
        
        # Creating the XML tree with the root2 element
        tree = ET.ElementTree(root2)
        rough_string = ET.tostring(tree.getroot(), 'utf-8')
        reparsed = minidom.parseString(rough_string)
        reparsed = reparsed.toprettyxml(indent="  ")
        
        # Writing the XML tree to the output file                    
        output_file = os.path.join(os.path.join(output_dir, 'eval/re'), os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reparsed)
    
    ### Generate NER-RE eval data
    for xml_file in td.tqdm(xml_files, desc="Generating NER-RE eval data", unit="file"):
        with open(xml_file, 'rb') as f:
            rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
        xml_data = rawdata.decode(encoding)
        
        # Replace unescaped special characters
        xml_data = xml_data.replace('&', '&amp;')
        # Parse the XML string
        root = ET.fromstring(xml_data)
        
        # Extract EVENT and TIMEX3 annotations and sort by start offset
        target = []
        target_list = []
        for event in root.findall(".//EVENT"):
            event_modality = event.attrib.get('modality', '')
            event_polarity = event.attrib.get('polarity', '')
            event_type = event.attrib.get('type', '').lower()  # converting type to lowercase for comparison
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test', 'occurrence']:
                target.append(event.attrib)
                target_list.append(event.attrib['id'])
        for timex in root.findall(".//TIMEX3"):
            target.append(timex.attrib)
            target_list.append(timex.attrib['id'])
            
        target_list.sort(key=lambda x: x[0])                
        event = list(filter(lambda x:'E' in x, target_list))
        timex = list(filter(lambda x:'T' in x, target_list))
            
        # Extract related TLINK of target event/timex
        for tlink in root.findall(".//TLINK"):
            # EVENT to EVENT
            if (tlink.attrib["fromID"] in event and tlink.attrib["toID"] in event) and tlink.attrib["id"].startswith('TL'):
                target.append(tlink.attrib)
            # EVENT to TIME X
            if (tlink.attrib["fromID"] in event and tlink.attrib["toID"] in timex) and tlink.attrib["id"].startswith('TL'):
                target.append(tlink.attrib)
            # TIMEX to EVENT
            if (tlink.attrib["fromID"] in timex and tlink.attrib["toID"] in event) and tlink.attrib["id"].startswith('TL'):
                target.append(tlink.attrib)
                
        root2 = ET.Element("TAGS")
        
        # Iterating through the target_tlink list and creating XML sub-elements
        for item in target:
            tlink = ET.SubElement(root2, 'EVENT', item)
        
        # Creating the XML tree with the root2 element
        tree = ET.ElementTree(root2)
        rough_string = ET.tostring(tree.getroot(), 'utf-8')
        reparsed = minidom.parseString(rough_string)
        reparsed = reparsed.toprettyxml(indent="  ")
        
        # Writing the XML tree to the output file                    
        output_file = os.path.join(os.path.join(output_dir, 'eval/nerre'), os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reparsed)