from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.utils.setup_helper import setup_agent_and_workspace, parse_wandb_to_hydra_config
import hydra
import torch
import logging
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
from copy import deepcopy
import pprint

from multiprocessing.connection import Listener

from movement_primitive_diffusion.datasets.scalers import (
    denormalize,
    normalize,
)

from movement_primitive_diffusion.datasets.process_batch import ProcessBatchProDMP

scaler_config: DictConfig = DictConfig(
    {
        "normalize_keys": [],
        "standardize_keys": [],
        "normalize_symmetrically": True,
        "scaler_values": {},
    }
)

address = ('localhost', 6000)

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="../conf", config_name="test_trained_agent_in_env")
def main(cfg: DictConfig) -> float:
    # Load wandb config
    wandb_config = OmegaConf.load(cfg.config)

    # Parse wandb config to hydra config
    hydra_config = parse_wandb_to_hydra_config(wandb_config)

    # Seeds:
    if "seed" in hydra_config:
        torch.manual_seed(hydra_config.seed)
        np.random.seed(hydra_config.seed)
        random.seed(hydra_config.seed)

    # Update config with new values
    hydra_config = OmegaConf.merge(hydra_config, cfg.to_change)

    # Setup agent, and workspace
    agent, _ = setup_agent_and_workspace(hydra_config,)

    # print(OmegaConf.to_yaml(hydra_config))
    # print(hydra_config.dataset_config.keys)

    # SCALING AND NORMALIZATION
    scaler_config = OmegaConf.to_container(hydra_config, resolve=True)
    pprint.pprint(scaler_config)

    normalize_keys: list[str] = scaler_config["dataset_config"]["normalize_keys"]
    print("normalize_keys")
    print(normalize_keys)
    
    standardize_keys: list[str] = scaler_config["dataset_config"]["standardize_keys"]
    print("standardize_keys")
    print(standardize_keys)
    
    normalize_symmetrically: list[str] = scaler_config["dataset_config"]["normalize_symmetrically"]
    print("normalize_symmetrically")
    print(normalize_symmetrically)
    
    scaler_values: list[str] = scaler_config["dataset_config"]["scaler_values"]
    print("scaler_values")
    print(scaler_values)

    # for key in normalize_keys:
    #     assert key in scaler_values, f"Key {key} not found in scaler values."
    #     for metric in ["min", "max"]:
    #         assert (
    #             metric in scaler_values[key]
    #             and scaler_values[key][metric] is not None
    #         ), f"Key {key} does not have {metric} in scaler values."
    #         scaler_values[key][metric] = np.array(scaler_values[key][metric], dtype=np.float32)


    # def denormalize_destandardize_actions( action: torch.tensor) -> torch.tensor:
    #     action_copy = deepcopy(action)
    #     if (key := "action") in normalize_keys:
    #         action_copy = denormalize(action_copy, scaler_values[key], symmetric=normalize_symmetrically)
    #         print("denomalize!")
    #     # action_copy = np.clip(action_copy, self.action_space.low, self.action_space.high)
    #     return action_copy
    
    ProcessBatch_ = ProcessBatchProDMP(scaler_config["agent_config"]["process_batch_config"]["t_obs"], 
                                 scaler_config["agent_config"]["process_batch_config"]["t_pred"], 
                                 hydra_config["agent_config"]["process_batch_config"]["action_keys"],
                                 scaler_config["agent_config"]["process_batch_config"]["observation_keys"],
                                 ["agent_pos"],
                                 ["agent_vel"])

    print('waiting for a connection')
    with Listener(address, authkey=b'secret password') as listener:
        with listener.accept() as conn:
            print('connection accepted from', listener.last_accepted)
            while True:
                try:
                    obs = None
                    if conn.poll():  # Check if there is data to read
                        obs = conn.recv()
                        print("Receiving data...")
                        print(obs)
        
                    # Observations from Isaac Sim (self.observation_buffer["agent_pos"], self.observation_buffer["agent_vel"])
                    # Predict the next action sequence
                    if(obs is not None):
                        
                        # obs["agent_pos"] = normalize(obs["agent_pos"].to(device='cpu'), scaler_values["agent_pos"], normalize_symmetrically)
                        print("Normalized!")

                        for key in obs.keys():
                            obs[key] = obs[key].to(device='cpu')

                        print(obs)

                        extra = {}
                        obs, extra = ProcessBatch_.process_env_observation(observation = obs)
                        print("process_env_observation!")

                        for key in obs.keys():
                            obs[key] = obs[key].to(device='cuda')
                        for key in extra.keys():
                            extra[key] = extra[key][:, :-1].to(device='cpu')
                        

                        print("obs")
                        print(obs)
                        print("extra")
                        print(extra)

                        actions = agent.predict(obs, extra)
                        print("Predicted!")

                        # actions = denormalize_destandardize_actions(actions.to(device='cpu'))
                        actions = actions.to(device='cuda')

                        print("Denormalized actions:")
                        print(actions)

                        conn.send(actions)
                        print("Actions send!")
                except BrokenPipeError:
                        continue
                except ConnectionResetError:
                    continue
                except ConnectionRefusedError:
                    continue
                except Exception as e:
                    print(e)
                    continue

if __name__ == "__main__":
    main()