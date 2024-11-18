import sys
import yaml
from osl_dynamics.config_api.batch import IndexParser, BatchTrain, batch_check

def main(index,config_path):
    '''
    This is the main function of all the analysis.
    Parameters
    ----------
    index: int
        index == -1 represents initialisation of the training
    config_path: str
        where read the config path.
    analysis_config_path: str
        if analysis_config_path is not None, and index == -1
        then implement the analysis code.
    '''
    with open(config_path, 'r') as file:
        config_batch = yaml.safe_load(file)
    if index > -1:
        if isinstance(config_batch,list):
            with open(f'{config_batch[index]}batch_config.yaml','r') as file:
                config = yaml.safe_load(file)
        else:
            index_parser = IndexParser(config_batch)
            config = index_parser.parse(index)
        batch_train = BatchTrain(config)
        batch_train.model_train()
    else:
        index_parser = IndexParser(config_batch)

if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    index = 0
    config_path = './results_final/real/config_HCP.yaml'
    main(index,config_path)


