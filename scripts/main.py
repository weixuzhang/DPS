import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import huggingface_hub
import hydra
from omegaconf import OmegaConf

from src.configs import RunnerConfigs, register_base_configs
from src.run import Run
from src.utils import common_utils


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: RunnerConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    print(configs)

    common_utils.setup_random_seed(configs.random_seed)

    # Check if we should run in offline mode
    offline_mode = os.getenv("HF_OFFLINE", "false").lower() in ["true", "1", "yes"]
    
    if offline_mode:
        print("Running in offline mode - using cached models only")
    else:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Running in online mode - logging in to Hugging Face")
            huggingface_hub.login(token=hf_token)
        else:
            print("Running in online mode - using anonymous Hugging Face access")

    runner = Run(configs)
    _ = runner.test()


if __name__ == "__main__":
    register_base_configs()
    main()
