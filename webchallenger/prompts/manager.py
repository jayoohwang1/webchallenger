import os
from loguru import logger
from types import SimpleNamespace

import yaml

from webchallenger.utils import Singleton


class PromptManager(metaclass=Singleton):
    def __init__(self) -> None:
        self.prompts_dict = {}
        # self.prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.prompt_dir = os.path.dirname(__file__)
        for prompt_file in os.listdir(self.prompt_dir):
            if prompt_file.endswith('.yaml'):
                with open(os.path.join(self.prompt_dir, prompt_file)) as f:
                    p = yaml.safe_load(f)
                    for prompt in p['prompt']['prompts']:
                        current_version = prompt["current_version"] if "current_version" in prompt else 0
                        name = prompt["name"]
                        prompt_txt = prompt["versions"][current_version]
                        if name in self.prompts_dict:
                            logger.warning(f'Duplicate prompt name "{name}" in file {prompt_file}')
                            continue
                        self.prompts_dict[prompt["name"]] = prompt["versions"][current_version]
    
    def __getattr__(self, name: str):
        return self.prompts_dict[name]
    
promp_manager = PromptManager()
