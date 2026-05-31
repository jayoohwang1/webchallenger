""" Model loading logic
"""
import io
import base64
import sys
import threading
import time
import os
from contextlib import contextmanager
from types import SimpleNamespace

import requests
import survey
from loguru import logger
from PIL.Image import Image

from webchallenger.utils import Singleton

INIT_TIMEOUT = 30  # seconds
LOADING_TIMEOUT = 1200  # 20 minutes


# Function to encode the image
def encode_image(image: Image=None, image_path: str=""):
    if image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def deduplicate_output(output: str) -> str:
    """"""

    lines = output.splitlines()
    
    deduplicated_lines = []
    for line in lines:
        if (len(line) < 5) or (line not in deduplicated_lines):
            deduplicated_lines.append(line)
    
    truncated_output = '\n'.join(deduplicated_lines)

    return truncated_output


@contextmanager
def capture_stdouterr():
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = sys_stdout, sys_stderr


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.vlm_temp = None
        self.llm_temp = None
        self.verbose = None
        self.init_t = None
        self.server_host = None
        self.server_port = None
        self.ready = True

    def init(self, vision: str, planning: str, args: SimpleNamespace):
        self.vision = vision
        self.planning = planning
        self.vlm_temp = args.vlm_temp
        self.llm_temp = args.llm_temp
        self.do_sample = args.do_sample
        self.verbose = int(args.verbose)

        self.local_backup = args.local_backup
        self.llm_api_url = args.llm_api_url
        self.llm_api_model = args.llm_api_model
        self.vlm_api_url = args.vlm_api_url
        self.vlm_api_model = args.vlm_api_model

        self.llm_api_key = os.environ.get('LLM_API_KEY')

        self.n_llm_requests = 0
        self.n_vlm_requests = 0

        self.total_tokens = 0
        self.max_req_tokens = float('-inf')
        self.avg_req_tokens = None
        self.avg_task_tokens = None


    def vlm_call(
        self, 
        screenshot:Image=None, 
        vlm_prompt:str=None, 
        sys_prompt:str=None,
        verbose=None, 
        vlm_temp=None, 
        do_sample=None, 
        image_path=None
    ) -> str:
        if not self.ready:
            raise ValueError("Vision model not initialized")
        if not screenshot:
            logger.warning(f"screenshot=None")
            return ""

        if (screenshot.size[0] > 2000) or (screenshot.size[1] > 2000):
            # resize while keeping aspect ratio
            screenshot.thumbnail(size=(1280, 1280))
        if not image_path:
            image_path = f"{os.path.dirname(__file__)}/input_image.png"
            screenshot.save(image_path)

        if not verbose:
            verbose = self.verbose
        if vlm_temp == None:
            vlm_temp = self.vlm_temp
        if do_sample == None:
            do_sample = self.do_sample
        if self.vision in ['qwen2.5vl', 'holo1']:
            add_bos_token = False
        else:
            add_bos_token = True

        if not sys_prompt:
            sys_prompt = 'You are a helpful assistant.'
        

        # local OpenAI-compatible model server
        base64_image = encode_image(screenshot, image_path)
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type":"text","text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        },
                        {"type": "text", "text": vlm_prompt}
                    ]
                }
            ],
            "temperature": vlm_temp,
            "repetition_penalty": 1.01,
            "add_bos_token": add_bos_token,
            "skip_special_tokens": False
        }
        if self.vlm_api_model:
            payload['model'] = self.vlm_api_model
        vlm_api_key = os.environ.get('VLM_API_KEY')

        res = requests.post(
            f"{self.vlm_api_url}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {vlm_api_key}"},
            json=payload
        )
        self.n_vlm_requests += 1

        if res.status_code == 200:
            output = res.json()['choices'][0]['message']['content']
            request_tokens = res.json()['usage']['total_tokens']
            self.total_tokens += request_tokens
            if request_tokens > self.max_req_tokens:
                self.max_req_tokens = request_tokens

            if verbose >= 1:
                logger.debug(f"Prompt #{self.n_vlm_requests}: {vlm_prompt}")
                logger.info(f"VLM: {output}\n")
            if len(output) > 5000:
                logger.warning(f"Output too long: {len(output)}, truncating duplicate lines.")
                # likely model repetition loop
                output = deduplicate_output(output)
            return output
        else:
            logger.exception(
                f"Vision model call failed with status code {res.status_code}, {res.text}")
            return ""


    def multi_img_vlm_call(
        self, 
        images:list[Image]=[],
        vlm_prompt:str=None, 
        sys_prompt:str=None,
        verbose=None, 
        vlm_temp=None, 
        do_sample=None,
        image_paths:list[str]=[], 
    ) -> str:
        
        if not self.ready:
            raise ValueError("Vision model not initialized")
        if not verbose:
            verbose = self.verbose
        if vlm_temp == None:
            vlm_temp = self.vlm_temp
        if do_sample == None:
            do_sample = self.do_sample
        if self.vision in ['qwen2.5vl', 'holo1']:
            add_bos_token = False
        else:
            add_bos_token = True

        image_files = []
        if not image_paths:
            for i in range(len(images)):
                image = images[i]
                if (image.size[0] > 2000) or (image.size[1] > 2000):
                    image.thumbnail(size=(1280, 1280))
                image_path = f"{os.path.dirname(__file__)}/multi_img_input_{i}.png"
                image.save(image_path)
                image_files.append(image_path)
        else:
            image_files = image_paths
        logger.debug(f"Images: {[img.size for img in images]}")
        
        
        if not sys_prompt:
            sys_prompt = 'You are a helpful assistant.'
        
        message_content = []
        for i in range(len(images)):
            image = images[i]
            image_path = image_files[i]
            base64_image = encode_image(image, image_path=image_path)
            if image_path.endswith('.png'):
                url = f"data:image/png;base64,{base64_image}"
            elif image_path.endswith('.jpg'):
                url = f"data:image/jpeg;base64,{base64_image}"
            elif image_path.endswith('.webp'):
                url = f"data:image/webp;base64,{base64_image}"
            else:
                url = f"data:image/jpeg;base64,{base64_image}"
            content_image = {"type": "image_url", "image_url": {"url": url}}
            message_content.append(content_image)
        message_content.append({"type": "text", "text": vlm_prompt})

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type":"text","text": sys_prompt}]},
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "temperature": vlm_temp,
            "add_bos_token": add_bos_token,
            "skip_special_tokens": False
        }
        if self.vlm_api_model:
            payload['model'] = self.vlm_api_model
        vlm_api_key = os.environ.get('VLM_API_KEY')

        res = requests.post(
            f"{self.vlm_api_url}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {vlm_api_key}"},
            json=payload
        )
        self.n_vlm_requests += 1

        if res.status_code == 200:
            output = res.json()['choices'][0]['message']['content']
            request_tokens = res.json()['usage']['total_tokens']
            self.total_tokens += request_tokens
            if request_tokens > self.max_req_tokens:
                self.max_req_tokens = request_tokens
            if verbose >= 1:
                logger.debug(f"Prompt #{self.n_vlm_requests}: {vlm_prompt}")
                logger.info(f"VLM: {output}\n")
            if len(output) > 5000:
                logger.warning(f"Output too long: {len(output)}, truncating duplicate lines.")
                # likely model repetition loop
                output = deduplicate_output(output)
            return output
        else:
            logger.exception(
                f"Vision model call failed with status code {res.status_code}, {res.text}")
            return ""


    def cog_llm_call(self, vlm_prompt: str) -> str:  #deprecated
        if not self.ready:
            raise ValueError("Vision model not initialized")
        res = requests.post(
            f"{self.server_url}/cog_llm_call",
            data={"vlm_prompt": vlm_prompt})
        if res.status_code == 200:
            return res.json()["output"]
        else:
            logger.exception(
                f"Planning model call failed with status code {res.status_code}, {res.text}")
            raise ValueError("Planning model call failed")


    def llm_call(self, messages: list[dict[str, str]], verbose=None, show_prompt=True, llm_temp=None, do_sample=None, max_tokens=None) -> str:
        if not self.ready:
            raise ValueError("Planning model not initialized")
        
        if not verbose:
            verbose = self.verbose
        if llm_temp == None:
            llm_temp = self.llm_temp
        if do_sample == None:
            do_sample = self.do_sample
        if not max_tokens:
            # max_tokens = 4096
            # max_tokens = 3072
            max_tokens = 2048
        
        base_url = f"{self.llm_api_url}/v1/chat/completions"
        if self.llm_api_url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions":
            base_url = self.llm_api_url

        payload = {
            "messages": messages,
            "temperature": llm_temp,
        }
        if self.planning == 'qwen3':
            payload['temperature'] = 0.6
            payload['top_k'] = 20
            payload['top_p'] = 0.95
        if self.planning.startswith('qwen'):
            payload['add_bos_token'] = False
        if not self.planning == 'gemini':
            payload['repetition_penalty'] = 1.01
            payload['max_tokens'] = max_tokens
        elif self.planning == 'gemini':
            payload['maxOutputTokens'] = max_tokens
            # disable gemini-2.5 thinking
            # payload['reasoning_effort'] = "none"
        if self.llm_api_model:
            payload['model'] = self.llm_api_model
        # self.llm_api_key = os.environ.get('LLM_API_KEY')
        
        

        num_retries = 0
        delay = 10.0

        while num_retries < 3:
            res = requests.post(
                base_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.llm_api_key}"},
                json=payload
            )
            self.n_llm_requests += 1
            if (self.llm_api_model):
                if ('free' in self.llm_api_model) and (self.n_llm_requests >= 950):
                    logger.success(f"n_requests={self.n_llm_requests}, switch to local")
                    self.llm_api_model = None
                    self.llm_api_key = os.environ.get('OPENAI_API_KEY')
                    self.llm_api_url = "http://localhost:5000"
            
            if res.status_code == 200:
                try:
                    res_json = res.json()
                except:
                    logger.error(f"JSON decode error")
                    res_json = None
                if (not res_json) or ('choices' not in res_json) or (res_json['choices'][0]['message']['content']==''):
                    if self.local_backup:
                        logger.error(f"ERROR: response = {res}")
                        logger.success(f"Prompt local model")
                        local_key = os.environ.get('OPENAI_API_KEY')
                        local_url = "http://localhost:5000"
                        res = requests.post(
                            f"{local_url}/v1/chat/completions",
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {local_key}"},
                            json=payload
                        )
                    else:
                        num_retries += 1
                        time.sleep(delay)
                        continue

                output = res.json()['choices'][0]['message']['content']
                request_tokens = res_json['usage']['total_tokens']
                self.total_tokens += request_tokens
                if request_tokens > self.max_req_tokens:
                    self.max_req_tokens = request_tokens

                if show_prompt:
                    if verbose == 2:
                        system_prompt = messages[0]['content']
                        logger.debug(f"System prompt:\n{system_prompt}")
                    if verbose >= 1:
                        user_prompt = messages[1]['content']
                        logger.info(f"User prompt:\n{user_prompt}")
                if verbose >=1:
                    logger.info(f"LLM request #{self.n_llm_requests}, total_tokens={self.total_tokens}: \n{output}\n")
                return output
            else:
                logger.exception(
                    f"Planning model call failed with status code {res.status_code}, {res.text}")
                num_retries += 1
                time.sleep(delay)
    

    # deprecated
    @property
    def server_url(self):
        return f"http://{self.server_host}:{self.server_port}"


# Create a global model manager
model_manager = ModelManager()
