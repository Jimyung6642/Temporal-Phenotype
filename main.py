import logging, logging.handlers
import configparser
import os

from generate_data import generate_eval_data, generate_input_data
from eval import eval_ner, eval_re, eval_nerre
from run_api import run_ner, run_re, run_nerre

from datetime import datetime as date

if __name__ == "__main__":
    try:
        os.chdir('/Users/jimmypark/Documents/git/Temporal-Phenotype')
        file_log = './tRE_' + date.today().strftime("%y%m%d") + '.log' 
        
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
        
        logger.info('============================================================')
        logger.info('Start!')
        logger.info('============================================================')
        
        # Set input and output directories
        input_dir = os.path.join(os.getcwd(), 'i2b2-2012-original')
        if not os.path.exists(input_dir):
            logging.error(f'i2b2-2012-original directory does not exist. Please uploaed data or change directory name to i2b2-2012-original')
        output_dir = os.path.join(os.getcwd(), 'result') 
        
        logger.info(f'i2b2 data path: {input_dir}')
        logger.info(f'output directory path: {output_dir}\n')
        
        # Generate and transform i2b2 data        
        if not os.path.exists(os.path.join(output_dir, 'data')):
            logger.info('converting i2b2-2012 data for tasks...')
            generate_input_data(input_dir, output_dir)
            logger.info(f'done converting i2b2-2012\n')
        else:
            logger.info(f'skip convering i2b2 data. data directory already exists in {output_dir}')
        
        # Generate eval datasets
        if not os.path.exists(os.path.join(output_dir, 'eval')):
            logger.info(f'generate eval data from i2b2-2012...')
            generate_eval_data(input_dir, output_dir)
            logger.info(f'done generating eval data\n')
        else:
            logger.info(f'skip convering eval data. eval directory already exists in {output_dir}')
        
        ### Run GPT API function
        ## NER
        # few shot
        logger.info('============================================================')
        logger.info('start API request')
        logger.info('============================================================')        
        try:
            logger.info(f'============================================================')
            logger.info(f'start one-shot ner...')
            logger.info(f'============================================================')
            run_ner(output_dir, few_shot = True, api_retry=6)
            logger.info(f'done one-shot ner\n')
        except Exception as e:
            logger.info(f'error occurred while running few-shot ner: {e}')
        # zero shot
        try:
            logger.info(f'============================================================')
            logger.info(f'start zero-shot ner...')
            logger.info(f'============================================================')
            run_ner(output_dir, few_shot = False, api_retry=6)
            logger.info(f'done zero-shot ner\n')
        except Exception as e:
            logger.info(f'error occurred while running one-shot ner: {e}')
        ## tRE
        # few shot
        try:
            logger.info(f'============================================================')
            logger.info(f'start one-shot re...')
            logger.info(f'============================================================')
            run_re(output_dir, few_shot = True, api_retry=6)
            logger.info(f'done one-shot tre\n')
        except Exception as e:
            logger.info(f'error occurred while running one-shot tre: {e}')
        # zero shot
        try:
            logger.info(f'============================================================')
            logger.info(f'start zero-shot re...')
            logger.info(f'============================================================')
            run_re(output_dir, few_shot = False, api_retry=6)
            logger.info(f'done zero-shot tre\n')
        except Exception as e:
            logger.info(f'error occurred while running zero-shot tre: {e}')
        ## NER-RE
        # few shot
        try:
            logger.info(f'============================================================')
            logger.info(f'start one-shot ner-re...')
            logger.info(f'============================================================')
            run_nerre(output_dir, few_shot = True, api_retry=6)
            logger.info(f'done one-shot ner-re\n')
        except Exception as e:
            logger.info(f'error occurred while running one-shot ner-re: {e}')
        # zero shot
        try:
            logger.info(f'============================================================')
            logger.info(f'start zero-shot ner-re...')
            logger.info(f'============================================================')
            run_nerre(output_dir, few_shot = False, api_retry=6)
            logger.info(f'done zero-shot ner-re\n')
        except Exception as e:
            logger.info(f'error occurred while running zero-shot ner-re: {e}')
        
        # Evaluate GPT output
        logger.info('============================================================')
        logger.info('start model evaluation')
        logger.info('============================================================')
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), "api.config"))
        model = config['openai']['model']
        # one_basic_path = "output_" + "one_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d")
        # zero_basic_path = "output_" + "zero_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d")
        one_basic_path = "output_" + "one_" + config['openai']['model'] + '_' + '231027'
        zero_basic_path = "output_" + "zero_" + config['openai']['model'] + '_' + '231027'        
        
        # one-shot ner
        if os.path.exists(os.path.join(output_dir, one_basic_path, 'ner')):
            logger.info(f'============================================================')
            logger.info('evaluate one-shot ner output...')
            logger.info(f'============================================================')
            try:
                eval_df = eval_ner(output_dir, few_shot=True, execute_date = None)   
                eval_df.to_csv(os.path.join(output_dir, one_basic_path, 'ner_one_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluating one-shot ner: \n{e}')
        else:
            logger.info(f'pass evaluating one-shot ner...\n')
        # zero-shot ner
        if os.path.exists(os.path.join(output_dir, zero_basic_path, 'ner')):
            logger.info(f'============================================================')
            logger.info('evaluate zero-shot ner output')
            logger.info(f'============================================================')
            try:
                eval_df = eval_ner(output_dir, few_shot=False, execute_date = None)
                eval_df.to_csv(os.path.join(output_dir, zero_basic_path, 'ner_zero_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluting zero-shot ner: \n{e}')
        else:
            logger.info(f'pass evaluating zero-shot ner output...\n')
            
        # one-shot re
        if os.path.exists(os.path.join(output_dir, one_basic_path, 're')):
            logger.info(f'============================================================')
            logger.info('evaluate one-shot tre output')
            logger.info(f'============================================================')
            try:
                eval_df = eval_re(output_dir, few_shot=True, execute_date = '231027')
                eval_df.to_csv(os.path.join(output_dir, one_basic_path, 're_one_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluating one-shot tre: \n{e}')
        else:
            logger.info(f'pass evaluating one-shot tre...\n')
        # zero-shot re
        if os.path.exists(os.path.join(output_dir, zero_basic_path, 're')):
            logger.info(f'============================================================')
            logger.info('evaluate zero-shot tre output')
            logger.info(f'============================================================')
            try:
                eval_df = eval_re(output_dir, few_shot=False, execute_date = None)
                eval_df.to_csv(os.path.join(output_dir, one_basic_path, 're_zero_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluating zero-shot tre: \n{e}')
        else:
            logger.info(f'pass evaluating zero-shot tre output...\n')
        
        # one-shot ner-re
        if os.path.exists(os.path.join(output_dir, one_basic_path, 'nerre')):
            logger.info(f'============================================================')
            logger.info('evaluate one-shot ner-re output')
            logger.info(f'============================================================')
            try:
                eval_df = eval_nerre(output_dir, few_shot=True, execute_date = None)
                eval_df.to_csv(os.path.join(output_dir, one_basic_path, 'nerre_one_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluating one-shot ner-re: \n{e}')
        else:
            logger.info(f'pass evaluating one-shot ner...\n')
        # zero-shot ner
        if os.path.exists(os.path.join(output_dir, zero_basic_path, 'ner')):
            logger.info(f'============================================================')
            logger.info('evaluate zero-shot ner-re output')
            logger.info(f'============================================================')
            try:
                eval_df = eval_nerre(output_dir, few_shot=False, execute_date = None)
                eval_df.to_csv(os.path.join(output_dir, one_basic_path, 'nerre_one_df.csv'), index=False)
            except Exception as e:
                logger.error(f'error occurred while evaluting zero-shot ner-re: \n{e}')
        else:
            logger.info(f'pass evaluating zero-shot ner-re output...\n')
        
        # Normalize temporal information into structured format
        
        logger.info('============================================================')
        logger.info('Done!')
        logger.info('============================================================')

    except Exception as e:
        logger.exception(e)       
