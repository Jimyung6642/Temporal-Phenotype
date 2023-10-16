import logging, logging.handlers
from configparser import ConfigParser
import os

from generate_data import generate_eval_data, generate_input_data
from eval import eval_re
from run_api import run_re

os.chdir('/phi_home/jp4453/Temporal-Phenotype')
file_log = './tRE.log'

if __name__ == "__main__":
    try:
        # Setup logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.handlers.RotatingFileHandler(file_log, maxBytes=10000000, backupCount=10)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        logger.info('===============================================')
        logger.info('Begin temporal relation extraction')
        
        # Set input and output directories
        input_dir = os.path.join(os.getcwd(), 'i2b2-2012-original') 
        output_dir = os.path.join(os.getcwd(), 'result') 
        logger.info(f'i2b2 data path: {input_dir}')
        logger.info(f'output directory path: {output_dir}')
        
        # Generate and transform i2b2 data
        logger.info('Transform i2b2-2012 data for tasks')
        generate_input_data(input_dir, output_dir)
        
        # Generate eval datasets
        logger.info('Generate eval data from i2b2-2012')
        generate_eval_data(input_dir, output_dir)
        
        # Run GPT API function
        logger.info('Run GPT API for the temporal relation extraction')
        run_re(output_dir, few_shot = True)
        
        # Evaluate GPT output
        logger.info('Evaluate tRE output')
        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = eval_re(output_dir, execute_date = None)
        
        # Neeed to be implemented
        # Standardize extracted temporal related concepts
        
        logger.info(f'macro precision: {macro_precision}')
        logger.info(f'macro recall: {macro_recall}')
        logger.info(f'macro f1: {macro_f1}')
        logger.info(f'micro precesion: {micro_precision}')
        logger.info(f'micro recall: {micro_recall}')
        logger.info(f'micro f1: {micro_f1}')
        
        logger.info('Done!')

    except Exception as e:
        logger.exception(e)
        
