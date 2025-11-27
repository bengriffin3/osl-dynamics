import sys
import yaml
from osl_dynamics.config_api.batch import IndexParser, BatchTrain, batch_check

from tpm_init import sticky_uniform_tpm

def main(index, config_path):
    with open(config_path, 'r') as file:
        config_batch = yaml.safe_load(file)

    if index > -1:
        if isinstance(config_batch, list):
            with open(f'{config_batch[index]}batch_config.yaml', 'r') as file:
                config = yaml.safe_load(file)
        else:
            index_parser = IndexParser(config_batch)
            config = index_parser.parse(index)

        # --- Inject initial TPM from YAML (if p_stay exists) ---
        if "model" in config and "hmm" in config["model"]:
            hmm_kwargs = config["model"]["hmm"]["config_kwargs"]
            n_states = config["n_states"]  # scalar after IndexParser.parse(...)

            p_stay = hmm_kwargs.pop("p_stay", None)
            if p_stay is not None and "initial_trans_prob" not in hmm_kwargs:
                if n_states > 1:
                    hmm_kwargs["initial_trans_prob"] = sticky_uniform_tpm(n_states, float(p_stay)).tolist()
                    offdiag = (1.0 - float(p_stay)) / (n_states - 1)
                    print(f"[TPM init] n_states={n_states} p_stay={float(p_stay):.4f} offdiag={offdiag:.6f}")
                else:
                    print("[TPM init] n_states=1 → skipping custom TPM (not needed)")


        batch_train = BatchTrain(config)
        batch_train.model_train()
    else:
        index_parser = IndexParser(config_batch)

if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    main(index, config_path)

# def main(index,config_path):
#     '''
#     This is the main function of all the analysis.
#     Parameters
#     ----------
#     index: int
#         index == -1 represents initialisation of the training
#     config_path: str
#         where read the config path.
#     analysis_config_path: str
#         if analysis_config_path is not None, and index == -1
#         then implement the analysis code.
#     '''
#     with open(config_path, 'r') as file:
#         config_batch = yaml.safe_load(file)
#     if index > -1:
#         if isinstance(config_batch,list):
#             with open(f'{config_batch[index]}batch_config.yaml','r') as file:
#                 config = yaml.safe_load(file)
#         else:
#             index_parser = IndexParser(config_batch)
#             config = index_parser.parse(index)
#         batch_train = BatchTrain(config)
#         batch_train.model_train()
#     else:
#         index_parser = IndexParser(config_batch)

# if __name__ == '__main__':
#     index = int(sys.argv[1]) - 1
#     config_path = sys.argv[2]
#     main(index,config_path)
