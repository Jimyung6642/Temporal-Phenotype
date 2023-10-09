import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import glob
import tqdm as td

import chardet
# Need to add merge_data for all function

def i2b2_to_sentence(input_dir, output_dir): 
    '''
    Convert I2B2 annotation form into in-line annotation format. 
    Only EVENTs (problem, test, and treatment) with positive, factual, and TIMEX3 will be extracted.
    '''
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of XML files
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    # xml_file = xml_files[0]
    # Loop through all XML files in the input directory
    for xml_file in td.tqdm(xml_files, desc="Converting annotation format", unit="file"):
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
        id = 1
        for event in root.findall(".//EVENT"):
            event_modality = event.attrib.get('modality', '')
            event_polarity = event.attrib.get('polarity', '')
            event_type = event.attrib.get('type', '').lower()  # converting type to lowercase for comparison
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test']:
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
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(converted_text)
   
    
def generate_tlink_eval_data(input_dir, output_dir):
    '''
    Only TLINK of PROBLEM, TEST, TREATMENT, and TIMEX3 will be extracted
    '''
    # Create directory to store evaluation dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    # Get the lists of XML files
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    # xml_file = xml_files[0]
    # Loop through all XML files in the input directory
    for xml_file in td.tqdm(xml_files, desc="Generating eval set", unit="file"):
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
            if event_modality == 'FACTUAL' and event_polarity == 'POS' and event_type in ['problem', 'treatment', 'test']:
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
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.xml')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reparsed)