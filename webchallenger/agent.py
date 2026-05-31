"""
WebChallenger
"""

import ast
import bisect
import copy
import glob
import json
import math
import os
import pickle
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, List, TypedDict, Union

import numpy as np
import requests
import survey
from loguru import logger
from markdownify import markdownify as md
from PIL import Image

from playwright.sync_api import Locator, ElementHandle, FrameLocator, Page, Error, TimeoutError, ViewportSize

# os.environ["PLAYWRIGHT_NODE_OPTIONS"] = "--max-old-space-size=16384"
os.environ["NODE_OPTIONS"] = "--max-old-space-size=8192"

from webchallenger.env import WebChallengerEnv
from webchallenger.memory import (
    EnvMemory, 
    WebsiteMem, 
    PageMem, 
    PageSection, 
    Element, 
    AgentAction, 
    SystemState, 
    create_action,
    print_action,
    save_trajectory
)
from webchallenger.models import model_manager
from webchallenger.prompts import promp_manager
from webchallenger.utils import (
    ActionInfo, 
    BBox,
    crop_img,
    stack_images,
    images_identical,
    draw_bbox_on_image,
    draw_point_on_image,
    get_website_url,
    get_sim_url,
    revert_sim_url,
    site_name,
    convert_local_url,
    clean_text,
    is_state_class,
    remove_html_tags,
    remove_nested_html,
    process_aria,
    extract_url_snapshot,
    process_md,
    extract_file_url_md,
    parse_class_string,
    autocorrect_text,
    get_full_url,
    url_path
)
from webchallenger.visualwebarena import (
    Action,
    ActionTypes,
    StateInfo,
    Trajectory,
    create_goto_url_action,
    create_key_press_action,
    create_keyboard_type_action,
    create_mouse_click_action,
    create_mouse_hover_action,
    create_none_action,
    create_stop_action,
    create_scroll_action,
    evaluator_router,
    get_captioning_fn,
    get_image_ssim
)
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html


# Images with smaller width or height won't be described by VLM
MIN_IMG_X = 50
MIN_IMG_Y = 50

MAX_NEW_PAGES = 500  # max pages to explore for each website 
MAX_ELEMENTS = 75  # max number of elements to iterate in analyze_elements


# Variable to store the last dialog message
last_dialog_msg = ""

def handle_dialog(dialog):
    global last_dialog_msg  # Access the global variable
    last_dialog_msg = f"{dialog.type}: '{dialog.message}'"  # Update with the latest dialog message
    logger.warning(f"Dialog {dialog.type}: {dialog.message}")  # Print the dialog message
    if dialog.type in ["confirm", "beforeunload"]:
        dialog.accept()  # Accept confirm dialogs
    else:
        dialog.dismiss()  # Dismiss alert dialogs or others




def add_agent_args(parser: ArgumentParser):
    parser.add_argument("--manual", action="store_true", default=False,
                        help="This will allow user to give order")
    parser.add_argument("--model_test", action="store_true", default=False,
                        help="Enter interactive menu for testing llm prompts")
    parser.add_argument("--start_url", default="https://www.google.com/",
                        help="Run agent on the url")
    parser.add_argument("--save_history", action="store_true", default=True,
                        help="If true, saves system state and agent action at each time step.")
    parser.add_argument("--data_dir", default=f'{os.path.dirname(__file__)}/data_dir',
                        help='Directory to save agent trajectory data')
    parser.add_argument("--verbose", default=1,
                        help="0 = Agent function calls, 1 = log llm user prompts, 2 = log llm system prompts")
    parser.add_argument("--mem_dir", default=f'{os.path.dirname(__file__)}/memory/saved_files',
                        help="Directory where agent memory is stored.")
    parser.add_argument("--result_dir", default=f'{os.path.dirname(__file__)}/results',
                        help="Directory where agent episode histories are stored.")
    
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_page_actions", type=int, default=30,
                        help="Max number of times to execute page_action for a task.")
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=4,
    )
    parser.add_argument("--test_config_base_dir", type=str, default=f"{os.path.dirname(__file__)}/benchmarks/visualwebarena/config_files")
    parser.add_argument("--test_start_idx", type=int, default=None)
    parser.add_argument("--test_end_idx", type=int, default=None)




class Agent:
    def __init__(self, vision: str, planning: str, browser: str, load_mem: bool, args: SimpleNamespace, config):
        self.cfg: CFG = config
        self.vision = vision
        self.planning = planning
        self.manual = args.manual
        self.slow_mode = False
        self.model_manager = model_manager
        self.prompts = promp_manager
        self.verbose = int(args.verbose)

        self.headless = True
        self.browser = browser
        self.start_url = args.start_url
        self.env: WebChallengerEnv = None
        self.mem_dir = f'{args.mem_dir}'
        self.screenshot_dir = f'{os.path.dirname(__file__)}/memory/screenshots'
        self.save_mem = True
        self.confirm_mem_save = False
        self.memory = EnvMemory(self.mem_dir, load_mem)
        self.exploration_start_time = None
        self.browser_tabs: list[Page] = []  # all open tabs
        self.browser_tabs_obs: list[PageMem] = []
        self.browser_tab_index: int = 0  # focused tab
        self.page_obs: PageMem = None  # structured page observation
        self.website_mem: WebsiteMem = None
        self.prev_screenshot: Image = None  # screenshot before last action
        self.post_screenshot: Image = None  # screenshot after last action
        self.screen_change = ""
        self.prev_thought = ""
        self.prev_action = ""
        self.clipboard = ""
        self.use_bookmarks = True
        self.bookmarks: list[Page] = []
        self.website_bookmarks: dict[Any] = dict()
        self.task_websites: list[str] = []
        self.changed_website = False
        self.page_suggestions = ""

        self.data_dir = args.data_dir
        self.homepage_start = True
        self.input_images = []
        self.input_image_paths = []
        self.input_image_desc = dict()
        self.image_task_context = ""
        self.saved_info = {
            "files": {
                "input_files": [f for f in os.listdir(f"{args.data_dir}/files/input_files")],
                "saved_files": [f for f in os.listdir(f"{args.data_dir}/files/saved_files")],
                "default_files": [f for f in os.listdir(f"{args.data_dir}/files/default_files")]
            },
            "saved_links": [],
            "notes": [],
            "required_info": {}
        }
        self.init_intent = None
        self.intent = None  # original user instruction
        self.history_str = ""
        self.detailed_history = ""
        self.form_plan = ""
        self.task_checks = 0
        self.task_complete = False
        self.no_actions_left = False
        self.answer = ""
        
        self.completed_subgoals = []
        self.last_milestone = -1  # time of last completed subgoal
        self.current_subgoal = ""

        self.task_id = None
        self.result_dir = args.result_dir
        self.trajectory_log_dir = ""
        self.trajectory_screenshot_dir = ""
        self.save_history = args.save_history
        self.time_step = 0
        self.llm_log = []
        self.error_log = []
        self.invalid_elems = []
        self.failed_actions = []
        self.repeated_actions = []
        self.clicked_suggestions = []
        self.dropdown_level = 0
        self.action_stack = []
        self.trajectory: Trajectory = []
        self.prev_states = []
        self.last_state_i = 0
        self.visited_pages = dict()  # all (url, PageMem) pairs visited in episode

        # Eval
        self.max_steps = args.max_steps
        self.max_page_actions = args.max_page_actions
        self.max_parse_error = args.parsing_failure_th
        self.max_repeat = args.repeating_action_failure_th
        
        self.task_config_dir = args.test_config_base_dir
        self.start_idx = args.test_start_idx
        self.end_idx = args.test_end_idx
        self.allowed_websites: list[str] = []  # website urls agent is confined to

    
    def reset_agent(self):
        """Reset agent state"""

        self.memory.reset()
        
        self.browser_tabs: list[Page] = []
        self.browser_tabs_obs: list[PageMem] = []
        self.browser_tab_index: int = 0
        self.page_obs: PageMem = None
        self.website_mem: WebsiteMem = None
        self.prev_screenshot: Image = None
        self.post_screenshot: Image = None
        self.screen_change = ""
        self.prev_thought = ""
        self.prev_action = ""
        self.clipboard = ""
        self.bookmarks: list[Page] = []
        self.website_bookmarks: dict[Any] = dict()
        self.task_websites: list[str] = []

        # Clear agent files
        saved_folder = f"{self.data_dir}/files/saved_files"
        input_folder = f"{self.data_dir}/files/input_files"
        for folder_dir in [saved_folder, input_folder]:
            for file_name in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted agent file: {file_path}")

        self.homepage_start = True
        self.input_images = []
        self.input_image_paths = []
        self.input_image_desc = dict()
        self.image_task_context = ""
        self.saved_info = {
            "files": {
                "input_files": [],
                "saved_files": [],
                "default_files": [f for f in os.listdir(f"{self.data_dir}/files/default_files")]
            },
            "saved_links": [],
            "notes": [],
            "required_info": {}
        }
        self.intent = None  # original user instruction
        self.task_complete = False
        self.no_actions_left = False
        self.answer = ""
        self.form_plan = ""
        self.history_str = ""
        self.detailed_history = ""
        self.task_checks = 0

        self.completed_subgoals = []
        self.last_milestone = -1
        self.current_subgoal = ""

        self.time_step = 0
        self.llm_log = []
        self.error_log = []
        self.invalid_elems = []
        self.failed_actions = []
        self.repeated_actions = []
        self.clicked_suggestions = []
        self.action_stack = []
        self.trajectory: Trajectory = []
        self.prev_states = []
        self.last_state_i = 0
        self.visited_pages = dict()


    ### ----- LLM string parsing functions ---- ###

    def format_vlm_prompt(self, image_path: str, query: str):
        """Format messages with image and prompt for VLM."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        return messages


    def format_llm_prompt(self, system_prompt: str=None, user_prompt: str=None, thinking=True) -> list[dict[str, str]]:
        """Formats the system and user prompt into list of dictionaries expected by call_llm.
        
        Args:
            system_prompt (str): System prompt with instructions for the chat LLM
            user_prompt (str): The user's query

        Returns:
            messages (list[dict[str, str]]): chat messages
        """

        if not system_prompt and not user_prompt:
            logger.warning(f"No prompt provided")
        if not user_prompt:
            user_prompt = system_prompt
            system_prompt = None
        if not system_prompt:
            system_prompt = f"You are a helpful assistant."
        
        user_prompt = get_sim_url(user_prompt)  # replace localhost with website name


        if self.planning == 'llama3_8b':
            messages = [
                {"role": "system", "content": f"\n{system_prompt}"},
                {"role": "user", "content": f"{user_prompt}"}
            ]
        elif self.planning in ['mistral7b', 'mistral_small', 'solar_pro_preview']:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
        elif self.planning in ['qwen_14b', 'virtuoso_small']:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        elif self.planning == 'qwen3':
            if thinking == False:
                user_prompt += f' /nothink'
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        return messages
    

    def extract_llm_answer(self, llm_out: str, keyword: str, backups=[], line_only=True, ret_full=False) -> str:
        """Extracts the answer text following the keyword."""

        if not llm_out:
            return ""

        # Remove <thinking>
        if (self.planning == 'qwen3') and ('</think>' in llm_out):
            llm_out = llm_out.split('</think>')[1].strip()
        
        answer = ""

        if not line_only:
            # Get all text after the answer keyword
            if keyword in llm_out:
                answer = llm_out.split(keyword)[1].strip()
                return answer

        lines = llm_out.split('\n')
        for line in lines:
            if keyword in line:
                answer = line.split(keyword)[1].strip()
                return answer
        
        # Check for backup strings
        for line in lines:
            for keyword in backups:
                if keyword in line:
                    answer = line.split(keyword)[1].strip()
                    return answer
        
        logger.warning(f'keyword: "{keyword}" not in llm_out:\n{llm_out}')
        if ret_full:
            answer = llm_out

        return answer
    

    def extract_bullet_list(self, list_str: str) -> list[str]:
        """Extract markdown bullet point list items from VLM output."""

        item_list = []

        lines = list_str.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 2:
                continue

            if line[0] in ['*', '-', '+', '•']:
                # Extract item text from line using regex
                match = re.match(r'^[*\-+•]\s*(.*)', line)
                if match:
                    item_text = match.group(1).strip()
                    item_list.append(item_text)

        return item_list


    def vlm_click_prompt(self, elem_desc: str, action: False) -> str:
        """"""

        if self.vision == 'molmo_7b':
            if action:
                vlm_prompt = self.prompts.point_action_molmo.format(action=elem_desc)
            else:
                vlm_prompt = self.prompts.point_molmo.format(desc=elem_desc)
        elif self.vision == 'os_atlas_7b':
            if action:
                vlm_prompt = self.prompts.point_action_osatlas.format(action=elem_desc)
            else:
                vlm_prompt = self.prompts.point_osatlas.format(desc=elem_desc)
        elif self.vision == 'qwen2.5vl':
            if action:
                vlm_prompt = self.prompts.point_action_qwen.format(action=elem_desc)
            else:
                vlm_prompt = self.prompts.point_qwen.format(desc=elem_desc)
        elif self.vision == "holo1":
            vlm_prompt = self.prompts.point_holo1.format(desc=elem_desc)

        return vlm_prompt


    def parse_molmo_points(self, point_str: str) -> list[tuple[float, float]]:
        """Return x, y coordinates from molmo point output scaled to 1.0 range."""

        # Pattern for single point
        single_point_pattern = r'<point x="([\d.]+)" y="([\d.]+)"'
        
        # Pattern for multiple points
        multi_point_pattern = r'<points\s((?:x\d+="[\d.]+"\s*y\d+="[\d.]+"(?:\s|$))+)'
        
        # Check for single point
        single_match = re.search(single_point_pattern, point_str)
        if single_match:
            return [(float(single_match.group(1))/100, float(single_match.group(2))/100)]
        
        # Check for multiple points
        multi_match = re.search(multi_point_pattern, point_str)
        if multi_match:
            points_str = multi_match.group(1)
            point_pairs = re.findall(r'x(\d+)="([\d.]+)"\s*y\1="([\d.]+)"', points_str)
            return [(float(x)/100, float(y)/100) for _, x, y in sorted(point_pairs)]
        
        # If no match found
        return []


    def parse_os_atlas_point(self, point_str: str) -> tuple[float, float]:
        """Parses bounding box from os-atlas output and returns centre point."""

        # from: https://github.com/e2b-dev/secure-computer-use/blob/os-computer-use/os_computer_use/grounding.py
        match = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", point_str)
        inner_text = match.group(1) if match else point_str
        numbers = [float(num)/1000 for num in re.findall(r"\d+\.\d+|\d+", inner_text)]
        if len(numbers) == 2:
            return numbers[0], numbers[1]
        elif len(numbers) >= 4:
            return (numbers[0] + numbers[2]) / 2, (numbers[1] + numbers[3]) / 2

        # If no match found
        return []
    

    def parse_qwen_points(self, point_str, img_hw) -> list[tuple[float, float]]:
        """Parses xml format point output and returns normalized coordinates.

        from: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
        """

        point_str = point_str.replace('&', '&amp;')

        # TODO: make sure image is resized before vlm_call
        width, height = img_hw

        # Parsing out the markdown fencing
        lines = point_str.splitlines()
        for i, line in enumerate(lines):
            if line == "```xml":
                point_str = "\n".join(lines[i+1:])  # Remove everything before "```xml"
                point_str = point_str.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```xml" is found

        xml_text = point_str.replace('```xml', '')
        xml_text = xml_text.replace('```', '')
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_text)
            num_points = (len(root.attrib) - 1) // 2
            points = []
            for i in range(num_points):
                x = root.attrib.get(f'x{i+1}')
                y = root.attrib.get(f'y{i+1}')
                x = int(x) / width
                y = int(y) / height
                points.append([x, y])
            return points
        except Exception as e:
            logger.warning(str(e))
            return []


    def parse_holo1_points(self, point_str: str, img_hw) -> tuple[float, float]:
        """"""

        width, height = img_hw

        pattern = r"Click\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        match = re.fullmatch(pattern, point_str) # re.fullmatch ensures the entire string matches

        if match:
            x_str, y_str = match.groups() # groups() returns a tuple of captured strings
            try:
                x = int(x_str) / width
                y = int(y_str) / height
                return (x, y)
            except ValueError:
                raise ValueError(
                    f"Coordinates '{x_str}', '{y_str}' are not valid integers in '{point_str}'."
                )
        else:
            raise ValueError(
                f"Invalid point_str format: '{point_str}'. Expected 'Click(x, y)'."
            )

    
    def parse_vlm_coords(self, point_str: str, img_hw=None) -> list[tuple[float, float]]:
        """"""

        if self.vision == 'molmo_7b':
            return self.parse_molmo_points(point_str)
        elif self.vision == 'os_atlas_7b':
            return [self.parse_os_atlas_point(point_str)]
        elif self.vision == 'qwen2.5vl':
            return self.parse_qwen_points(point_str, img_hw)
        elif self.vision == "holo1":
            return [self.parse_holo1_points(point_str, img_hw)]

        return []
    

    def add_list_numbers(self, str_list: list[str], start_index: int=1, indent=True, letter=False) -> list[str]:
        """"""

        numbered_strs = []
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        for i in range(len(str_list)):
            string = str_list[i]
            if letter:
                n = letters[i]
            else:
                n = i + start_index
            numbered = f'{n}) {string}'
            if indent:
                numbered = f'	{numbered}'
            numbered_strs.append(numbered)

        return numbered_strs


    def format_list_to_str(self, str_list: list[str], numbered: bool=False, double_newline=False, indent=True) -> str:
        """Formats a list of strings into a bullet or numbered list as a string."""

        list_str = ""
        
        if double_newline:
            el = "\n\n"
        else:
            el = "\n"
        
        i = 1
        for str in str_list:
            if numbered:
                start = f"{i})"
            else:
                start = '*'
            line_str = f"{start} {str}"
            line_str += el
            if indent:
                line_str = f"	{line_str}"
            list_str += line_str
            i += 1

        return list_str.strip('\n')

    
    def indent_str(self, string: str, indent_lvl: int=1, indent_first=False, tab=False) -> str:
        """Add indent to lines in string."""

        lines = string.split('\n')
        indent = '  '  # 2 space
        if tab:
            indent = '	'  # \t
        indent *= indent_lvl

        indented = []
        for i in range(len(lines)):
            line = lines[i]
            if (i==0) and (not indent_first):
                indented.append(line)
            else:
                indented.append(f'{indent}{line}')

        return '\n'.join(indented)

    




    #### ---- Basic browser actions ---- ####

    def scroll_to_bbox(self, bbox: BBox, upper_margin=200, iframe_id=None) -> BBox:
        """Scrolls the page so that the top of the bbox is at the top of the screen, or to
        bottom of the page. Returns new bbox with px coordinates adjusted to match page scroll."""

        screen_height = self.env.page.viewport_size["height"]
        page_height = max(screen_height, self.env.page.evaluate("document.body.scrollHeight"))

        logger.info(f"scroll to bbox: {bbox.get_abs_px_coords()}")
        window = self.env.page
        if iframe_id:
            window = self.get_frame_locator(iframe_id).locator('body')
        current_scroll_y = window.evaluate("window.scrollY")

        try:
            if (page_height - screen_height) < bbox.y1_abs_px:
                # Scroll to bottom
                window.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                # Calc dist between top of screen and top of bbox.
                offset = bbox.y1_abs_px - (page_height - screen_height)
            else:
                # Scroll top of screen to match top of bbox
                top_y = max((bbox.y1_abs_px - upper_margin), 0)
                if top_y < 250:
                    top_y = 0
                # window.evaluate(f"window.scrollTo(0, {top_y})")
                self.env.page.mouse.move(bbox.x1_abs_px, 500)
                time.sleep(0.5)
                self.env.page.mouse.wheel(0, top_y-current_scroll_y)
                offset = 0
            time.sleep(1.0)
        except Error as e:
            logger.warning(f"Error: {str(e)}")
            offset = 0

        crop_bbox = bbox.copy()
        crop_bbox.y1_abs_px = offset
        crop_bbox.y2_abs_px = offset + bbox.get_abs_px_height()

        return crop_bbox
    

    def key_press_action(
        self,
        trajectory: Trajectory,
        key_comb: str
    ) -> tuple[Trajectory, bool]:
        """Press the key or key combination (e.g. 'Control+A')"""

        logger.info(f'Executing key press action: {key_comb}')
        
        try:
            self.env.page.keyboard.press(key_comb)
            # time.sleep(0.5)
        except Error as e:
            return [], False

        return trajectory, True
    

    def go_to_url_action(
        self, 
        trajectory: Trajectory, 
        url: str,
        check_load: bool=True,
        max_retries: int=3
    ) -> tuple[Trajectory, bool]:
        """Goes to the url on the current browser window."""

        if not url:
            logger.warning("FAIL go_to_url: action requires argument.\n")
            return trajectory, False
        
        if url.startswith('/'):
            site_url = get_website_url(self.env.page.url)
            url = site_url + url[1:]
        if not url.startswith('h'):
            return trajectory, False
        
        retries = 0
        success = False
        while (not success) and (retries < max_retries):
            logger.info(f"Executing go_to_url action: {url}")
            try:
                response = self.env.page.goto(url)
                if (not response) or (response.status>=400):
                    logger.error(f"failed navigation")
                else:
                    success = True
                    break
            except Error as e:
                logger.error("failed navigation")
            retries += 1
            time.sleep(1.0)
        
        time.sleep(3.0)
        self.env._wait_dom_loaded()
        if check_load:
            is_loading = self.check_load_screen()

        return trajectory, success


    def login_action(
            self, 
            trajectory: Trajectory
        ) -> tuple[Trajectory, bool]:
        """Sign in if we have login details for the current website."""

        # Get login details for site
        website_url = get_website_url(self.env.page.url)
        if (website_url in self.memory.accounts):
            account_details = self.memory.accounts[website_url]
        elif (self.env.page.url in self.memory.accounts):
            account_details = self.memory.accounts[self.env.page.url]
        else:
            return trajectory, True
        login_page = account_details["login_page"]
        username = account_details["username"]
        password = account_details["password"]
        user_field_id = account_details["user_field_id"]
        pass_field_id = account_details["pass_field_id"]
        logger.info(f"Logging in with credentials: user = {username}, password = {password}")

        _, _ = self.go_to_url_action(trajectory, login_page)
        if (login_page not in self.env.page.url):
            logger.debug(f"Redirected {login_page}, {self.env.page.url}")
            return trajectory, True
        
        # locate login fields
        user_locator = self.get_elem_locator(id=user_field_id).first
        pass_locator = self.get_elem_locator(id=pass_field_id).first
        if (not user_locator.count()) or (not pass_locator.count()):
            logger.error(f"Login failed, can't locate login fields")
            return trajectory, False

        # enter credentials
        user_locator.fill(username)
        pass_locator.fill(password)
        time.sleep(0.5)
        _, success = self.key_press_action([], 'Enter')
        time.sleep(2.0)
        self.env._wait_dom_loaded(networkidle=True)
        self.check_load_screen()

        return trajectory, success
    



    #### ---- Element locator functions ---- ####

    def get_element_label(self, element_loc: Locator, element: Element=None, iframe_id=None) -> str:
        """Return the label for the element if it exists, else return None."""

        try:
            base_locator = self.env.page.locator('body')
            if iframe_id:
                base_locator = self.get_frame_locator(iframe_id)

            # Get the label for the element
            label_text = None

            id = element.id
            if id:
                label_locator = base_locator.locator(f'label[for="{id}"]').last
                if not label_locator.count():
                    label_locator = element_loc.locator('xpath=ancestor::label') or element_loc.locator('xpath=following-sibling::label')
                if label_locator.count():
                    label_text = label_locator.evaluate("label => label.textContent.trim()")
            
            aria_labelledby = element.aria_labelledby
            if aria_labelledby:
                label_locator = base_locator.locator(f'id={aria_labelledby}').first
                try:
                    label_text = label_locator.text_content(timeout=100)
                except TimeoutError as e:
                    label_text = None
            
            if element.tag in ['input', 'textarea', 'select']:
                span_loc = element_loc.locator('xpath=following-sibling::span').filter(visible=True).first
                if (span_loc.count()) and (span_loc.get_attribute('aria-hidden') != 'true'):
                    try:
                        span_text = span_loc.text_content(timeout=500)
                    except TimeoutError as e:
                        logger.warning(f"TimeoutError: {str(e)}")
                        span_text = None
                    if (span_text) and (len(span_text.splitlines())==1):
                        if label_text:
                            label_text += f" - {span_text}"
                        else:
                            label_text = span_text
                
                header_locator = element_loc.locator('xpath=../preceding-sibling::h3[1]').filter(visible=True)
                if (header_locator.count()):
                    header_text = header_locator.evaluate("label => label.textContent.trim()")
                    if label_text:
                        label_text = f"{header_text} {label_text}"
                    else:
                        label_text = header_text
            
            if (not label_text) and (base_locator.locator('div.form-group')).count():
                label_locator = element_loc.locator(
                    "xpath=./ancestor::div[contains(@class, 'form-group')]/label"
                )
                if label_locator.count():
                    label_text = label_locator.evaluate("label => label.textContent.trim()")
            
            return label_text
        
        except Error as e:
            logger.warning(f"Error: {e}")
            return None

    
    def get_valid_attr(self, element_loc, attribute: str) -> str:
        """Get attribute value and check for Playwright selector Error."""

        try:
            if isinstance(element_loc, Locator):
                attr_value = element_loc.get_attribute(f'{attribute}', timeout=500)
            else:
                attr_value = element_loc.get_attribute(f'{attribute}')

            if not attr_value:
                return None
            if attribute == 'class':
                return attr_value

            test_loc = self.env.page.locator(f'[{attribute}="{attr_value}"]').first
            _ = test_loc.count()
        except Error as e:
            logger.warning(f'{str(e)} while getting attribute "{attribute}"')
            return None
        
        return attr_value


    def page_scroll_height(self, iframe_id=None):
        """Get scroll height of current page. If iframe_id, get scroll height of frame_locator."""

        try:
            scroll_height = self.env.page.evaluate("window.pageYOffset || document.documentElement.scrollTop")
            
            if self.cfg.benchmark == 'workarena':
                if (self.page_obs) and (self.page_obs.html_sections):
                    iframe_id = self.page_obs.html_sections[-1].iframe_id
            
            if (iframe_id):
                frame_loc = self.get_frame_locator(iframe_id)
                if frame_loc.count():
                    scroll_height = frame_loc.evaluate("window.document.documentElement.scrollTop")

        except Error as e:
            logger.warning(f"Error: {str(e)}")
            scroll_height = 0

        return scroll_height


    def contained_in(self, selected_element: Element, locator: Locator, section: PageSection=None) -> Element:
        """"""

        if (section) and (section.type=='form'):
            selected_element.contained_in = "form"
            return selected_element
        if (section) and (section.type in ['nav', 'header']):
            return selected_element

        if locator.count() != 1:
            return selected_element

        if (selected_element.tag in ['input', 'textarea', 'select']) and ('field' in str(selected_element.parent_class)):
            selected_element.contained_in = "form"
            return selected_element

        # If element is part of form
        form_locator = locator.locator('xpath=ancestor::form')
        if form_locator.count():
            selected_element.contained_in = "form"

        # Check if element is in fieldset
        fieldsets = locator.locator('xpath=ancestor::fieldset').all()
        if fieldsets:
            legend_list = []
            for fieldset_locator in fieldsets:
                legend_locator = fieldset_locator.locator('xpath=descendant::legend').first
                if legend_locator.count():
                    legend = legend_locator.text_content(timeout=500)
                    legend_list.append(legend.strip())
            selected_element.fieldset = legend_list
            selected_element.contained_in = "form"

        return selected_element

    
    def get_elem_dict(self, element_loc: Locator):
        """"""

        try:
            elem_dict = element_loc.evaluate("""
                el => {
                    const elem = {};
                    elem['tag'] = el.tagName.toLowerCase();
                    elem['text'] = "";
                    if (el.innerText) {
                        elem['text'] = el.innerText.trim();
                    }
                    elem['bounding_box'] = el.getBoundingClientRect();
                    elem['cursor'] = window.getComputedStyle(el).getPropertyValue('cursor');
                    
                    const attrs = {};
                    for (const a of el.attributes) {
                        attrs[a.name] = a.value;
                    }
                    elem['attributes'] = attrs;
                    
                    return elem;
                }
            """)

            return elem_dict

        except TimeoutError as e:
            # Return None if locator doesn't work
            logger.warning(f"TimeoutError: {str(e)}")
            # logger.warning(traceback.format_exc())
            return None


    def elem_from_loc(self, element_loc: Locator, elem_dict=None, section=None, iframe_id=None, trow=None):
        """"""

        if not element_loc:
            return None
        if not element_loc.count() == 1:
            logger.debug(f"elem_loc.count() = {element_loc.count()}\n{element_loc}")
            return None
        
        try:
            if not elem_dict:
                elem_dict = self.get_elem_dict(element_loc)
                if not elem_dict:
                    return None
            attributes = elem_dict['attributes']

            # Create Element memory object
            element = Element()
            element.tag = elem_dict.get('tag')
            element.text = elem_dict.get('text', '')
            element.attributes_dict = attributes

            element.id = attributes.get('id')
            element.data_testid = attributes.get('data-testid')
            element.data_qa = attributes.get('data-qa-selector')
            element.data_value = attributes.get('data-value')
            element.data_track_property = attributes.get('data-track-property')
            element.aria_label = attributes.get('aria-label')
            element.aria_labelledby = attributes.get('aria-labelledby')
            element.aria_controls = attributes.get('aria-controls')
            element.aria_haspopup = attributes.get('aria-haspopup')
            element.aria_autocomplete = attributes.get('aria-autocomplete')
            element.aria_hidden = attributes.get('aria-hidden')
            element.aria_expanded = attributes.get('aria-expanded')
            element.aria_selected = attributes.get('aria-selected')
            element.open = 'open' in attributes
            if ('aria-current' in attributes) and (attributes.get('aria-current') != 'false'):
                element.open = True

            element.is_disabled = 'disabled' in attributes
            element.role = attributes.get('role')
            element.name = attributes.get('name')
            element.class_name = attributes.get('class', '')
            element.title = attributes.get('title')
            element.href = attributes.get('href')
            element.input_type = attributes.get('type')


            scroll_height = self.env.page.evaluate("window.pageYOffset || document.documentElement.scrollTop")
            if iframe_id:
                element.iframe_id = iframe_id
                scroll_height = self.page_scroll_height(iframe_id)
            element.bbox = BBox.from_playwright_bbox(elem_dict.get('bounding_box'), scroll_height)

            element.label = self.get_element_label(element_loc, element, iframe_id)
            parent = element_loc.locator('xpath=parent::*')
            if parent.count() == 1:
                parent_tag_class = parent.evaluate("""
                    el => {
                        const tag = el.tagName.toLowerCase();
                        const classValue = el.className;
                        return {tag: tag, className: classValue};
                    }
                """)
                element.parent_tag = parent_tag_class.get('tag')
                element.parent_class = parent_tag_class.get('className', '')

            if attributes.get('aria-expanded') == 'true':
                element.collapse = True
            
            if element.tag in ["input", "textarea", "select"]:
                if (element.tag=="select"):
                    element.options = self.all_select_options(element)
                    if 'multiple' in attributes:
                        element.multiple = True
                if (attributes.get('aria-required')=='true') or ('required' in attributes) or ('required' in element.class_name):
                    if element.input_type != "checkbox":
                        element.required = True
                element.input_value = self.get_input_value(element, input_loc=element_loc)
                element.placeholder = attributes.get('placeholder')
            element = self.contained_in(element, element_loc, section)

            if trow != None:
                element.table_row = trow
            if element.tag == "button":
                elem_name = element.get_name().lower()
                if ('Submit' in elem_name) or (elem_name in ["post", "comment", "apply", "set"]):
                    element.input_type = "submit"

            return element

        except TimeoutError as e:
            # Return None if locator doesn't work
            logger.warning(f"TimeoutError: {str(e)}")
            # logger.warning(traceback.format_exc())
            return None


    

    def is_clickable(self, elem_dict: dict, get_any=False) -> bool:
        """"""

        # Check bounding_box
        if (not elem_dict['bounding_box']) or (elem_dict['bounding_box']['right'] <= 0):
            return False

        # Check tags
        interactive_tags = {
            'button',
            'a',
            'input',
            'select',
            'textarea',
            # 'label',
            'details',
            'summary',
            # 'option',
            # 'optgroup',
        }
        if elem_dict['tag'] in interactive_tags:
            return True
    
        if (elem_dict['tag'] == 'i') and (not elem_dict['text']):
            return True
        if elem_dict['tag'] == 'label':
            if elem_dict['cursor'] == 'pointer':
                return True
            else:
                return False
        
        # Check attributes
        attributes = elem_dict['attributes']
        if attributes:
            # Check for event handlers or interactive attributes
            interactive_attributes = {'onclick', 'onmousedown', 'onmouseup', 'onkeydown', 'onkeyup'}
            if any(attr in attributes for attr in interactive_attributes):
                return True
            if ('tabindex' in attributes) and (attributes.get('tabindex') != '-1'):
                return True

            # Check roles
            role = None
            if 'role' in attributes:
                role = attributes['role']
            elif 'data-role' in attributes:
                role = attributes.get('data-role', '').lower()

            if role:
                interactive_roles = {
                    'button',
                    'link',
                    'menuitem',
                    'option',
                    'radio',
                    'checkbox',
                    'tab',
                    'textbox',
                    'combobox',
                    'slider',
                    'spinbutton',
                    'search',
                    'searchbox',
                }
                if role in interactive_roles:
                    return True

        # Check cursor style
        if (elem_dict['cursor'] in ['pointer', 'auto']) and (elem_dict['text']):
            class_name = attributes.get('class', '')
            if any(c in class_name for c in ['button', 'btn', 'link']) or ('data-placeholder' in attributes):
                if not any(c in class_name for c in ['buttons', 'links']):
                    return True
        
        if (elem_dict['cursor'] in ['pointer', 'auto']) and (elem_dict['text']):
            if 'button' in elem_dict['tag']:
                return True
            if 'accordion' in attributes.get('class', ''):
                return True
        
        if (elem_dict['cursor'] == 'pointer'):
            if 'close' in attributes.get('class', ''):
                return True
        
        if (get_any) and (elem_dict['cursor'] == 'pointer'):
            return True

        return False
    

    def clickable_locators(self, section_loc: Locator, section: PageSection, get_any=False):
        """"""

        # Locate all visible elements
        visible_elements = section_loc.locator('*').filter(visible=True)

        # Extract attributes
        all_elem_dicts = visible_elements.evaluate_all("""
            els => els.map(el => {
                const elem = {};
                elem['tag'] = el.tagName.toLowerCase();
                elem['text'] = "";
                if (el.innerText) {
                    elem['text'] = el.innerText.trim();
                }
                elem['bounding_box'] = el.getBoundingClientRect();
                elem['cursor'] = window.getComputedStyle(el).getPropertyValue('cursor');
                
                const attrs = {};
                for (const a of el.attributes) {
                    attrs[a.name] = a.value;
                }
                elem['attributes'] = attrs;
                
                return elem;
            })
        """)
        all_locs = visible_elements.all()

        interactible_elements = []
        for i in range(len(all_elem_dicts)):
            elem_dict = all_elem_dicts[i]
            if i >= len(all_locs):
                logger.warning(f"elements={len(all_elem_dicts)} > locators={len(all_locs)}")
                break
            if (not get_any) and (elem_dict.get('tag') not in ['input', 'textarea', 'select', 'label']):
                if (elem_dict['attributes'].get('role') not in ['combobox', 'tab']):
                    if ((section.type=='form') and (elem_dict['attributes'].get('tabindex')=='-1')):
                        continue
            if self.is_clickable(elem_dict, get_any=get_any):
                elem_loc = all_locs[i]
                interactible_elements.append((elem_loc, elem_dict))

        return interactible_elements


    def all_section_elements(self, section_loc: Locator, section: PageSection, get_any=False):
        """"""
        
        interactible_elements = self.clickable_locators(section_loc, section, get_any)

        final_elements = []
        for elem_loc, elem_dict in interactible_elements:
            elem = self.elem_from_loc(elem_loc, elem_dict, section, iframe_id=section.iframe_id)
            # if elem:
            final_elements.append(elem)

        return final_elements
    


    
    def focused_elem_handle(self):
        """Return ElementHandle for the current focused element if it exists, else None."""

        focused_handle = self.env.page.evaluate_handle('''
            () => {
                // Recursively find the focused element in shadow DOM
                function getFocusedElementInShadow(dom = document) {
                    if (dom.activeElement && dom.activeElement.shadowRoot) {
                        return getFocusedElementInShadow(dom.activeElement.shadowRoot);
                    }
                    return dom.activeElement;
                }

                // Return the focused element
                return getFocusedElementInShadow();
            }
        ''')

        if not focused_handle:
            return None
        
        focused_element_handle = focused_handle.as_element()
        
        return focused_element_handle


    def get_foc_elem(self) -> Element:
        """Get the currently focused element on the page and save it's details in an Element object.
        Returns None if no elements are selected.
        """

        foc_elem = None

        if 'service-now' not in self.env.page.url:
            element_loc = self.env.page.locator('*:focus')
            if element_loc.count() > 1:
                element_loc = element_loc.last
            foc_elem = self.elem_from_loc(element_loc)

        if not foc_elem:
            focused_element_handle = self.focused_elem_handle()
            if not focused_element_handle:
                return None

            import uuid
            unique_id = f"pw-temp-id-{uuid.uuid4()}"
            focused_element_handle.evaluate(f"(element) => element.setAttribute('{unique_id}', '')")
            focused_locator = self.env.page.locator(f"[{unique_id}]")
            foc_elem = self.elem_from_loc(focused_locator)
            focused_locator.evaluate("(element) => element.removeAttribute('pw-temp-id')")
            
            if not foc_elem:
                return None
        
        
        if foc_elem.tag == 'iframe':
            all_iframes = self.env.page.locator('iframe').all()
            for iframe in all_iframes:
                try:
                    iframe_id = iframe.get_attribute('id')
                    if (iframe_id) and ('/' not in iframe_id) and ('.' not in iframe_id):
                        frame_loc = self.env.page.frame_locator(f'#{iframe_id}')
                    else:
                        iframe_id = iframe.get_attribute('title')
                        if not iframe_id:
                            continue
                        frame_loc = self.env.page.frame_locator(f'[title="{iframe_id}"]')
                    element_loc = frame_loc.locator('*:focus').last
                    if element_loc.count():
                        return self.elem_from_loc(element_loc, iframe_id=iframe_id)
                except Exception as e:
                    continue

            return None

        if foc_elem.tag == 'body':
            return None

        return foc_elem
    
    
    def element_class_str(self, class_str: str, ignore_state=True) -> str:
        """Escape any special characters in element class and join into selector string."""

        if not class_str:
            return ""
        class_parts = class_str.split()

        # Escape each class name individually
        escaped_class_parts = []
        for part in class_parts:
            if (is_state_class(part)) and (ignore_state):
                continue
            escaped_part = re.sub(r'([^\w-])', r'\\\1', part)
            escaped_class_parts.append(escaped_part)
        
        escaped_classes = []
        for class_name in escaped_class_parts:
            if class_name[0].isdigit():
                escaped_class = f"\\{ord(class_name[0])} {class_name[1:]}"
                escaped_classes.append(escaped_class)
            else:
                escaped_classes.append(class_name)

        if len(escaped_classes) == 0:
            return ""
        class_str = ".".join(escaped_classes)
        class_str = f".{class_str}"

        return class_str


    def get_frame_locator(self, iframe_id) -> Locator:
        """"""

        try:
            if self.env.page.locator(f'iframe#{iframe_id}').count():
                frame_locator = self.env.page.frame_locator(f'id={iframe_id}').locator('html')
            elif self.env.page.locator(f'iframe[title="{iframe_id}"]').count():
                frame_locator = self.env.page.frame_locator(f'[title="{iframe_id}"]').locator('html')
            else:
                frame_locator = self.env.page.locator(":not(*)")
        except Error as e:
            logger.warning(f"{str(e)}")
            frame_locator = self.env.page.locator(":not(*)")

        return frame_locator


    def get_base_locator(self, element: Element, section: PageSection=None, section_strict=False):
        """Return locator for the page, section, or iframe the element is contained in."""

        if not self.cfg.split_page_sections:
            base_locator = self.env.page.locator('html')

        base_locator = self.env.page.locator('body')
        if not element:
            return base_locator

        if (section) and (str(section.parent_class) != str(element.parent_class)):
            section_loc = self.get_section_locator(page_section=section, ret_unique=False)
            if section_loc.count() == 1:
                if section_strict:
                    base_locator = section_loc
                else:  # optional section_loc
                    if section_loc.locator(f'{element.tag}').filter(visible=True).count():
                        base_locator = section_loc

        if (element.iframe_id) and ((not section) or (not section.iframe_id)):
            frame_loc = self.get_frame_locator(element.iframe_id)
            if frame_loc.locator('body').count():
                base_locator = frame_loc

        if element.table_row != None:
            row_loc = base_locator.locator('tbody > tr').filter(visible=True).nth(element.table_row)
            if row_loc.count():
                base_locator = row_loc
            else:
                logger.warning(f"row_loc: {row_loc}, {row_loc.count()}")

        return base_locator


    def get_elem_locator(
        self,
        element: Element=None,
        section: PageSection=None,
        id=None,
        debug=False,
        click_select=False,
        base_loc: Locator = None,
        first_only: bool=False,
        section_strict: bool=False
    ) -> Locator:
        """Attempts to find the element on the current page and returns a Playwright locator."""

        try:
            if not element:
                return self.env.page.locator(f'id={id}')
            if element.section:
                section = element.section

            base_locator = self.get_base_locator(element, section, section_strict=section_strict)
            if base_loc != None:
                base_locator = base_loc
            if element.id:
                id_loc = base_locator.locator(f'id={element.id}').filter(visible=True)
                if debug: print(f"id_loc: {id_loc} {id_loc.count()}\n")
                if id_loc.count() == 1:
                    return id_loc
                if id_loc.count():
                    locator = id_loc

            if element.parent_tag:
                class_select = self.element_class_str(element.parent_class)
                parent_loc = base_locator.locator(f'{element.parent_tag}{class_select}')
                if debug: print(f"parent_loc: {parent_loc} {parent_loc.count()}\n")
                if parent_loc.count():
                    base_locator = parent_loc
            
            locator = base_locator.locator(f'{element.tag}').filter(visible=True)
            if debug:
                print(f"{locator}: {locator.count()}\n")
            if (not locator.count()) and (isinstance(base_locator, Locator)):
                if (base_locator.count()==1):
                    return base_locator

            class_select = self.element_class_str(element.class_name)
            if class_select:
                class_loc = locator.and_(base_locator.locator(f'{class_select}'))
                if debug: print(f"class_loc: {class_loc} {class_loc.count()}\n")
                if class_loc.count() == 1:
                    return class_loc
                if class_loc.count():
                    locator = class_loc


            if element.data_testid:
                data_testid_loc = locator.and_(base_locator.get_by_test_id(f'{element.data_testid}'))
                if debug: print(f"data_testid_loc: {data_testid_loc} {data_testid_loc.count()}\n")
                if data_testid_loc.count() == 1:
                    return data_testid_loc
                if data_testid_loc.count():
                    locator = data_testid_loc
            if element.data_qa:
                data_qa_loc = locator.and_(base_locator.locator(f'[data-qa-selector="{element.data_qa}"]'))
                if debug: print(f"data_qa_loc: {data_qa_loc} {data_qa_loc.count()}\n")
                if data_qa_loc.count() == 1:
                    return data_qa_loc
                if data_qa_loc.count():
                    locator = data_qa_loc
            if element.data_value:
                data_value_loc = locator.and_(base_locator.locator(f'[data-value="{element.data_value}"]'))
                if debug: print(f"data_value_loc: {data_value_loc} {data_value_loc.count()}\n")
                if data_value_loc.count() == 1:
                    return data_value_loc
                if data_value_loc.count():
                    locator = data_value_loc
            if element.data_track_property:
                data_track_loc = locator.and_(base_locator.locator(f'[data-track-property="{element.data_track_property}"]'))
                if debug: print(f"data_track_loc: {data_track_loc} {data_track_loc.count()}\n")
                if data_track_loc.count() == 1:
                    return data_track_loc
                if data_track_loc.count():
                    locator = data_track_loc
            
            if element.aria_label:
                aria_label_loc = locator.and_(base_locator.get_by_label(f'{element.aria_label}', exact=True))
                if debug: print(f"aria_label_loc: {aria_label_loc} {aria_label_loc.count()}\n")
                if aria_label_loc.count() == 1:
                    return aria_label_loc
                if aria_label_loc.count():
                    locator = aria_label_loc
            if element.title:
                title_loc = locator.and_(base_locator.locator(f'[title="{element.title}"]'))
                if debug: print(f"title_loc: {title_loc} {title_loc.count()}\n")
                if title_loc.count() == 1:
                    return title_loc
                if title_loc.count():
                    locator = title_loc
            if element.name:
                name_loc = locator.and_(base_locator.locator(f'[name="{element.name}"]'))
                if debug: print(f"name_loc: {name_loc} {name_loc.count()}\n")
                if name_loc.count() == 1:
                    return name_loc
                if name_loc.count():
                    locator = name_loc
            if element.placeholder:
                placeholder_loc = locator.and_(base_locator.locator(f'[placeholder="{element.placeholder}"]'))
                if debug: print(f"placeholder_loc: {placeholder_loc} {placeholder_loc.count()}\n")
                if placeholder_loc.count() == 1:
                    return placeholder_loc
                if placeholder_loc.count():
                    locator = placeholder_loc

            if element.text != "":
                text_loc = locator.filter(has_text=f'{element.text}')
                if text_loc.count() > 1:
                    text_loc = locator.and_(base_locator.get_by_text(f'{element.text}', exact=True))
                if debug: print(f"text_loc: {text_loc} {text_loc.count()}\n")
                if text_loc.count() == 1:
                    return text_loc
                if text_loc.count():
                    locator = text_loc
            
            if (element.href) and (element.href != '#'):
                href_loc = locator.and_(base_locator.locator(f'[href="{element.href}"]'))
                if debug: print(f"href_loc: {href_loc} {href_loc.count()}\n")
                if href_loc.count() == 1:
                    return href_loc
                if href_loc.count():
                    locator = href_loc


            if element.nth:
                if not section:
                    section = element.section
                section_loc = self.get_section_locator(section)
                interactible_elements = self.clickable_locators(section_loc, section)
                if element.nth < len(interactible_elements):
                    elem_loc, elem_dict = interactible_elements[element.nth]
                    if debug: print(f"nth_loc: {elem_loc}\n")
                    return elem_loc

            
            if first_only or (section and section.data_level) or ('Copy' in element.get_name()):
                locator = locator.first

            if (not locator.count()) and (debug):
                logger.warning(f'Cannot locate Element {element.get_name()} on the page.')
        
        except Error as e:
            logger.warning(f"{repr(e)}")
            locator = self.env.page.locator(":not(*)")

        return locator
    

    def get_section_locator(self, page_section: PageSection, debug=False, ret_unique=True):
        """Get locator for PageSection element."""

        try:
            base_locator = self.env.page.locator('xpath=//html')
            if page_section.iframe_id:
                frame_loc = self.get_frame_locator(page_section.iframe_id)
                if frame_loc.locator('body').count():
                    base_locator = frame_loc
                else:
                    return frame_loc

            # list item
            if (page_section.list_section_tag):
                list_class_str = self.element_class_str(page_section.list_section_class)
                list_locator = base_locator.locator(f'{page_section.list_section_tag}{list_class_str}')
                item_loc = list_locator.locator('xpath=child::li').filter(visible=True).nth(page_section.nth)
                return item_loc

            if page_section.parent_tag:
                parent_class_str = self.element_class_str(page_section.parent_class)
                parent_loc = base_locator.locator(f'{page_section.parent_tag}{parent_class_str}')
                if debug: 
                    print(f"parent_loc: {parent_loc} {parent_loc.count()}\n")
                if parent_loc.count():
                    base_locator = parent_loc
            locator = base_locator

            class_select = self.element_class_str(page_section.class_name)
            if page_section.type == 'list':
                locator = locator.locator(f'{page_section.list_item_tag}{class_select}')
            else:
                locator = locator.locator(f'{page_section.type}{class_select}')
            locator = locator.filter(visible=True)

            if page_section.id and (':' not in str(page_section.id)):
                id_loc = locator.and_(base_locator.locator(f"id={page_section.id}"))
                if debug: print(f"id_loc: {id_loc} {id_loc.count()}\n")
                if id_loc.count() == 1:
                    return id_loc
                if id_loc.count():
                    locator = id_loc
            if page_section.data_index:
                locator = locator.and_(base_locator.locator(f'[data-index="{page_section.data_index}"]'))
            if page_section.role:
                locator = locator.and_(base_locator.locator(f'[role="{page_section.role}"]'))

            if (page_section.nth != None) and (locator.count() != 1):
                locator = locator.nth(page_section.nth)
            if page_section.type=='list':
                return locator
            
            if (1 < locator.count() < 100) and (ret_unique):
                section_height = page_section.bbox.get_abs_px_height()
                closest_size = None
                min_diff = 999
                xy_match = None
                for loc in locator.all():
                    if not loc.bounding_box(): continue
                    loc_bbox = loc.bounding_box()
                    h_diff = abs(loc_bbox['height'] - section_height)
                    if h_diff < min_diff:
                        closest_size = loc
                        min_diff = h_diff
                    if (int(loc_bbox['y'])==page_section.bbox.y1_abs_px) and (int(loc_bbox['x'])==page_section.bbox.x1_abs_px):
                        xy_match = loc
                if xy_match:
                    locator = xy_match
                elif closest_size:
                    locator = closest_size
            if (locator.count() > 1) and (ret_unique) and (page_section.bbox.y1_abs_px < 2000):
                logger.warning(f"locator.count(): {locator.count()}, use first")
                locator = locator.first
            
            if (page_section.is_dialog) and (self.cfg.benchmark=='workarena'):
                if self.env.page.locator('sn-impersonation').count():
                    locator = self.env.page.locator('sn-impersonation').first

            if debug:
                print(f"{locator}: {locator.count()}")
        
        except Error as e:
            logger.warning(f"{repr(e)}")
            locator = self.env.page.locator(":not(*)")

        return locator



    
    #### ---- Split page into sections ---- ####
    
    def should_split(self, section: Locator) -> bool:
        """Return True if we should divide div into smaller section, False otherwise."""

        if isinstance(section, list):
            return False
        
        bounding_box = section.bounding_box()
        tag = section.evaluate('element => element.tagName.toLowerCase()')
        role = section.get_attribute('role')
        class_name = section.get_attribute('class')

        if (tag in ['html', 'body']):
            return True
        if (tag=='main' or role=='main'):
            if (bounding_box) and (bounding_box['width'] < 1200) and (bounding_box['height']<1280):
                return False
        if ('footer' in str(class_name)) and (bounding_box['y']>2000):
            return False

        if (role in ['code', 'group']):
            return False

        if (tag == 'form') and ((bounding_box['height'] > 4500) and (bounding_box['width']>1250)):
            return True
        if (tag == 'article') and ((bounding_box['height'] > 4500) and (bounding_box['width']>1250)):
            return True
        if (tag in ['ol', 'ul']) and (bounding_box['height']>500) and (1<section.locator('li').filter(visible=True).count()<4):
            return True
        
        if tag in ['ol', 'ul', 'table', 'form', 'fieldset', 'aside', 'article', 'details', 'p', 'img', 'embed', 'nav', 'header', 'footer']:
            return False
        
        visible_children = self.visible_sections(section_locator=section)
        if (len(visible_children) > 5) and ((bounding_box['height'] < 720) or (bounding_box['width']<500)):
            return False
        
        link_child = section.locator('xpath=child::a').filter(visible=True)
        if (link_child.count()) and (tag != "body") and ((bounding_box['width']<=1200) or (bounding_box['height']<720)):
            return False
        paragraph_child = section.locator('xpath=child::p').filter(visible=True)
        if (paragraph_child.count()) and (tag != "body") and ((bounding_box['width']<=1200)):
            return False
        for child in visible_children:
            child_box = child.bounding_box()
            if not child_box:
                continue
            if ((child_box['height'] > 1500) and (child_box['width'] > 320)) or (child.evaluate('element => element.tagName.toLowerCase()')=='table'):
                return True
        
        if ((tag=='section') or (role=='region')) and (bounding_box['width']<1200 or bounding_box['height']<720):
            return False
        
        section_id = section.get_attribute('id')
        data_testid = section.get_attribute('data-testid')
        if (tag == 'div'):
            if (not section.get_attribute('class')) and (not section_id):
                return True
            if (bounding_box['y']>960) and (bounding_box['height']<720) and (section_id):
                return False

        if bounding_box['height'] > 900 and bounding_box['width'] > 320:
            return True
        if bounding_box['height'] > 500 and bounding_box['width'] > 800:
            return True

        # split main section of short page
        if (bounding_box['y'] < 420) and (bounding_box['height'] > 360) and (bounding_box['width'] > 1200):
            return True
        
        if section.locator('xpath=descendant::main').count():
            return True
        if tag == 'main':
            return True

        return False

    
    def should_split_elem(self, section, split_p=False) -> bool:
        """"""

        tag = section.evaluate('element => element.tagName.toLowerCase()')
        if (split_p == False) and (tag == 'p'):
            return False
        if tag in ['a', 'button', 'input', 'img', 'svg', 'time']:
            return False
        
        if (tag in ['table']) and (section.bounding_box()['height'] > 900):
            return False
        
        child_loc = section.locator('xpath=child::*')
        if child_loc.count():
            return True

        return False


    def visible_sections(self, section_locator: Locator, min_size=10) -> list[Locator]:
        """Get list of Locators for all visible child elements of the section."""

        visible_children = []
        children = section_locator.locator('xpath=child::*').all()

        for child in children:
            try:
                bounding_box = child.bounding_box(timeout=1000)
            except Error as e:
                bounding_box = None

            if (not child.is_visible()) or (not bounding_box):
                grand_children = self.visible_sections(child)
                if grand_children:
                    visible_children.extend(grand_children)
                continue
            
            # visible
            if not bounding_box:
                continue
            elif bounding_box['height'] < min_size or bounding_box['width'] < min_size:
                continue
            else:
                visible_children.append(child)
                
        return visible_children


    def element_split(self, section_locator: Locator, split_p=False) -> list[Locator]:
        """Recursively divide the section until components are small enough."""

        visible_children = self.visible_sections(section_locator, min_size=1)

        if not visible_children:
            return [section_locator]

        subsections = []
        for child in visible_children:
            if child.evaluate('element => element.tagName.toLowerCase()') == 'article':
                break
            elif self.should_split_elem(child, split_p):
                # recursively divide section
                child_sections = self.element_split(child, split_p)
                subsections += child_sections
            else:
                subsections.append(child)
        
        return subsections


    def merge_sections(self, section_list: list[Locator], min_length=4) -> list[Locator]:
        """If multiple sequential sections are the same type: combine them into list section."""

        if (section_list[0].bounding_box()):
            if (section_list[0].bounding_box()['height'] > 900) and (len(section_list) < 20):
                return section_list
            if (section_list[0].bounding_box()['width'] > 1200) and (len(section_list) < 20):
                return section_list

        combined_sections = []

        current_sequence = []
        current_tag = None
        current_class = None

        for section in section_list:
            tag = section.evaluate('element => element.tagName.toLowerCase()')
            class_name = str(section.get_attribute('class')).split(' ')[0]
            if tag == 'section':
                return section_list
            
            if (tag == current_tag) and (class_name == current_class):
                current_sequence.append(section)
            else:
                if len(current_sequence) >= min_length:
                    combined_sections.append(current_sequence)
                else:
                    combined_sections.extend(current_sequence)
                current_sequence = [section]
                current_tag = tag
                current_class = class_name
        
        # Handle the last sequence
        if len(current_sequence) >= min_length:
            combined_sections.append(current_sequence)
        else:
            combined_sections.extend(current_sequence)

        return combined_sections


    def split_section(self, section_locator: Locator) -> list[Locator]:
        """Recursively divide the section until components are small enough."""

        if (not self.cfg.split_page_sections):
            return [section_locator]

        visible_children = self.visible_sections(section_locator)

        if len(visible_children) >=4:  #> 5:
            visible_children = self.merge_sections(visible_children)

        subsections = []
        for child in visible_children:
            if self.should_split(child):
                # recursively divide section
                child_sections = self.split_section(child)
                subsections += child_sections
                continue

            if not isinstance(child, list):
                for child_table in child.locator('xpath=child::table').all():
                    if (child_table.bounding_box()) and (child_table.bounding_box()['height'] > 900):
                        subsections.append(child_table)
                        print("add child table")
            
            subsections.append(child)
        
        return subsections
    

    def grid_row_length(self, element_list) -> int:
        """"""

        if not element_list:
            return 0
        
        first_item_bbox = element_list[0].bounding_box()

        n = 1
        for item in element_list[1:]:
            current_bbox = item.bounding_box()
            if not current_bbox:
                continue
            if current_bbox['y'] > (first_item_bbox['y']+first_item_bbox['height']):
                break
            n += 1

        return n

    
    def get_list_type(self, element_list) -> str:
        """Return 'grid' if element list is arranged in a grid layout, 'list' otherwise."""

        if not isinstance(element_list, list):
            element_list = element_list.locator('xpath=child::li').all()

        if len(element_list) < 2:
            return 'list'

        first_item_bbox = element_list[0].bounding_box()
        if not first_item_bbox:
            return 'list'
        diff_x = False
        diff_y = False

        for item in element_list[1:]:
            current_bbox = item.bounding_box()
            if not current_bbox:
                continue
            if current_bbox['x'] != first_item_bbox['x']:
                diff_x = True
            if current_bbox['y'] != first_item_bbox['y']:
                diff_y = True
        
        if diff_x and diff_y:
            # print('grid')
            return 'grid'
        else:
            # print('list')
            return 'list'


    def create_section(self, section_loc: Locator, iframe_id=None, is_dialog=False, is_list=False) -> PageSection:
        """Create PageSection object from Playwright locator."""

        if section_loc.count() != 1:
            logger.error(f"section_loc.count() = {section_loc.count()}:\n{section_loc}")
            return None

        section_mem = PageSection()

        if iframe_id:
            section_mem.iframe_id = iframe_id
        scroll_height = self.page_scroll_height(iframe_id=iframe_id)

        bounding_box = section_loc.bounding_box()
        section_mem.bbox = BBox.from_playwright_bbox(bounding_box, scroll_height)
        tag = section_loc.evaluate('element => element.tagName.toLowerCase()')
        section_mem.type = tag
        section_mem.role = section_loc.get_attribute('role')
        class_name = section_loc.get_attribute('class')
        if class_name:
            if is_list:
                class_name = class_name.strip()
                class_name = class_name.split(' ')[0].strip()
            section_mem.class_name = class_name
        section_mem.id = section_loc.get_attribute('id')
        section_mem.data_level = section_loc.get_attribute('data-level')
        section_mem.data_index = section_loc.get_attribute('data-index')
        if tag == 'form':
            section_mem.is_form = True
        if is_dialog:
            section_mem.is_dialog = True
        if (tag == 'a') and (section_mem.class_name == 'external text'):
            return None

        parent = section_loc.locator('xpath=parent::*')
        if parent.count():
            section_mem.parent_tag = parent.evaluate('element => element.tagName.toLowerCase()')
            section_mem.parent_class = parent.get_attribute('class')
        
        if tag in ['ul', 'ol']:
            section_mem.list_type = self.get_list_type(section_loc)
            section_mem.list_item_tag = 'li'

        return section_mem


    def create_page_sections(self, subsections: list[Locator], iframe_id=None) -> list[PageSection]:
        """Convert HTML section elements into Agent memory objects."""

        page_sections = []
        scroll_height = self.env.page.evaluate("window.pageYOffset || document.documentElement.scrollTop")

        for section in subsections:
            section_mem = PageSection()
            if iframe_id:
                section_mem.iframe_id = iframe_id

            if isinstance(section, list):
                first_item_bbox = BBox.from_playwright_bbox(section[0].bounding_box(), scroll_height)
                last_item_bbox = BBox.from_playwright_bbox(section[-1].bounding_box(), scroll_height)
                if last_item_bbox:
                    first_item_bbox.x2_abs_px = last_item_bbox.x2_abs_px
                    first_item_bbox.y2_abs_px = last_item_bbox.y2_abs_px
                else:
                    first_item_bbox.y2_abs_px += 1000
                section_mem.bbox = first_item_bbox

                section_mem.list_type = self.get_list_type(section)
                section_mem.list_item_tag = section[0].evaluate('element => element.tagName.toLowerCase()')
                section_mem.data_level = section[0].get_attribute('data-level')
                section_mem.type = 'list'
                section_mem.class_name = section[0].get_attribute('class')
                if section_mem.class_name:
                    section_mem.class_name = section_mem.class_name.strip()
                    section_mem.class_name = section_mem.class_name.split(' ')[0].strip()
                parent = section[0].locator('xpath=parent::*')
                if parent.count():
                    section_mem.parent_tag = parent.evaluate('element => element.tagName.toLowerCase()')
                    section_mem.parent_class = parent.get_attribute('class')

                page_sections.append(section_mem)
                continue
            
            section_mem = self.create_section(section, iframe_id)
            if not section_mem:
                continue
            
            page_sections.append(section_mem)


        return page_sections


    def page_title(self, name="") -> str:
        """Return the title of the current page if found, else None."""
        
        title = ""

        self.env.page.evaluate(f"window.scrollTo(0, 0)")  # scroll to top of page

        title_loc = self.env.page.locator('h1').filter(visible=True)
        if (title_loc.count()):
            top_y = 350
            for title_loc in title_loc.all():
                text = title_loc.text_content().strip()
                if (text) and (title_loc.bounding_box()['y'] < top_y):
                    top_y = title_loc.bounding_box()['y']
                    title = text
        
        elif ((not title) or (not name) or (len(name)<3)) and (not self.cfg.llm_only):
            # Get title from VLM
            screenshot = self.env.get_screenshot(max_height=768)
            vlm_prompt = self.prompts.get_page_title
            vlm_out = self.model_manager.vlm_call(screenshot, vlm_prompt)
            title = self.extract_llm_answer(vlm_out, keyword='TITLE:', ret_full=True)

        return get_sim_url(title)


    def create_page_mem(self, website_mem: WebsiteMem, full_url=False) -> PageMem:
        """Init PageMem object."""
        page_mem = PageMem()
        url = self.env.page.url
        if (not full_url) and (not self.intent):
            url = self.env.page.url.split('#')[0]
        page_mem.url = url
        page_mem.website_url = website_mem.url

        return page_mem
    

    def divide_page(
        self, 
        website_mem: WebsiteMem,
        page_mem: PageMem,
        screenshot: bool = True,
        retries = 3
    ) -> PageMem:
        """Divide browser page and add sections to the PageMem."""

        page_mem.html_sections = []
        page_mem.list_section = None

        # Split page into sections and create PageMem
        body_loc = self.env.page.locator('xpath=//html/body')
        try:
            subsections = self.split_section(body_loc)
        except Error as e:
            if retries > 0:
                logger.warning(f"{str(e)}, retry")
                time.sleep(2.0)
                return self.divide_page(website_mem, page_mem, retries=retries-1)
        all_page_sections = self.create_page_sections(subsections)

        all_iframes = self.env.page.locator('iframe').filter(visible=True).all()
        for iframe in all_iframes:
            iframe_id = iframe.get_attribute('id')
            if (iframe_id) and ('/' not in iframe_id) and ('.' not in iframe_id):
                frame_loc = self.env.page.frame_locator(f'#{iframe_id}')
            else:
                iframe_id = iframe.get_attribute('title')
                if not iframe_id:
                    continue
                frame_loc = self.env.page.frame_locator(f'[title="{iframe_id}"]')
            try:
                frame_sections = self.split_section(frame_loc)
                all_page_sections += self.create_page_sections(frame_sections, iframe_id)
            except Error as e:
                logger.warning(f"Exception occured while splitting frame:\n\n{str(e)}")

        # If page not fully loaded, wait and try again
        if (len(all_page_sections) < 2) and (retries > 0) and (self.cfg.split_page_sections):
            if (not all_page_sections) or (all_page_sections[0].bbox.get_abs_px_height()<200):
                logger.warning(f"Page not loaded (sections={len(all_page_sections)}), retry")
                self.env.page.evaluate(f"window.scrollTo(0, 0)")
                time.sleep(2.0)
                return self.divide_page(website_mem, page_mem, retries=retries-1)

        nav = None
        for page_section in all_page_sections:
            if page_section.bbox.y1_abs_px == 0:
                nav = page_section
                break
        if not nav:
            nav_loc = self.env.page.locator('nav').first
            if nav_loc.is_visible() and nav_loc.bounding_box()['y'] == 0:
                all_page_sections = self.create_page_sections([nav_loc]) + all_page_sections


        for section_mem in all_page_sections:
            if section_mem.type in ['ul', 'ol', 'list'] and section_mem.bbox.get_abs_px_height() > 150:
                if not page_mem.list_section:
                    page_mem.list_section = section_mem
                elif (page_mem.list_section.type == 'list' and section_mem.type == 'list'):
                    page_mem.list_section.bbox.y2_abs_px = section_mem.bbox.y2_abs_px
                elif (section_mem.bbox.get_abs_px_height() > page_mem.list_section.bbox.get_abs_px_height()):
                    page_mem.list_section = section_mem
                else:
                    page_mem.add_section(section_mem)
                page_mem.is_list_item = True
            else:
                if section_mem.type=='table':
                    page_mem.is_list_item = True
                page_mem.add_section(section_mem)
        
        page_title = self.page_title(name=page_mem.name)
        if page_title:
            page_mem.name = page_title
        if self.env.page.url == get_website_url(self.env.page.url):
            page_mem.name = "Homepage"

        if screenshot:
            full_screenshot_arr = np.array(Image.open(BytesIO(self.env.page.screenshot(full_page=False, timeout=10000))))
            full_page_screenshot = Image.fromarray(full_screenshot_arr)
            for section_mem in (page_mem.html_sections + [page_mem.list_section]):
                if (not section_mem) or (not section_mem.bbox):
                    continue
                section_mem.print_section()
                full_page_screenshot = draw_bbox_on_image(full_page_screenshot, section_mem.bbox, abs=True)
            full_page_screenshot.save(f"{self.screenshot_dir}/sections.png")

        return page_mem


    #### ---- Element Exploration ---- ####

    def all_select_options(self, select_element: Element, section=None) -> list[str]:
        """Get list of all options for <select> element."""

        all_options = []
        select_locator = self.get_elem_locator(select_element, section)
        if select_locator.count() != 1:
            logger.warning(f"Can't locate {select_element.get_name()}, {select_locator.count()}")
            select_element.explored = False
            return []
        base_locator = self.get_base_locator(select_element)

        tag_name = select_locator.evaluate('element => element.tagName.toLowerCase()')
        if tag_name == 'select':
            options_text = select_locator.evaluate("select => Array.from(select.options).map(option => option.textContent.trim())")
            for option in options_text:
                all_options.append(option)
        
        elif select_element.role == "combobox":
            is_open = select_locator.get_attribute('aria-expanded')
            if is_open == "false":
                select_locator.click(force=True)
                time.sleep(0.5)
            options_id = select_locator.get_attribute('aria-owns')
            if not options_id:
                options_id = select_locator.get_attribute('aria-controls')
            options_loc = base_locator.locator(f'id={options_id}')
            all_options = options_loc.locator('li').all_text_contents()
            if (not self.intent):
                logger.info(f"Escape menu")
                self.key_press_action([], 'Escape')
        else:
            logger.error("Not a select element")
            return []

        if select_element.label in all_options:
            label_locator = select_locator.locator('xpath=ancestor::*[label]/child::label').first
            if label_locator.count():
                label = label_locator.evaluate("label => label.textContent.trim()")
                select_element.label = label

        return all_options
    

    def all_radio_options(self, radio_element: Element) -> list[Element]:
        """Get list options for the radio element."""
        
        radio_elements = []

        radio_locator = self.get_elem_locator(radio_element)
        base_locator = self.get_base_locator(radio_element)
        name = radio_locator.get_attribute('name')
        if name is None:
            return []
        
        radio_inputs = base_locator.locator(f'[name="{name}"]').all()
        for radio_loc in radio_inputs:
            radio_elem = self.elem_from_loc(radio_loc)
            radio_elem.explored = True
            radio_elements.append(radio_elem)

        # Get radio group label
        radio_label_locator = radio_locator.locator('xpath=ancestor::*[label]/child::label').first
        if radio_label_locator.count():
            radio_label = radio_label_locator.evaluate("label => label.textContent.trim()")
            for radio in radio_elements:
                radio.func_desc = radio_label

        return radio_elements


    def all_list_pages(self, website_mem: WebsiteMem, element: Element) -> list[PageMem]:
        """Get pages for all links in list."""

        all_pages = []

        start_url = self.env.page.url
        element.skip = True

        menu_loc = self.get_elem_locator(element)
        link_elements = menu_loc.locator('a').all()
        all_links = []
        for elem_loc in link_elements:
            all_links.append(elem_loc.get_attribute('href'))
        links = set(all_links)
        
        i = 0
        for url in links:
            if i > 500:
                break
            self.go_to_url_action([], url)
            if self.env.page.url not in website_mem.pages:
                page_mem = self.create_page_mem(website_mem)
                page_mem.name = self.page_title()
                page_mem.is_list_item = True
                all_pages.append(page_mem)
            i += 1
        
        # Go back to starting page
        _, _ = self.go_to_url_action([], start_url)

        return all_pages


    def analyze_input(self, website_mem: WebsiteMem, selected_element: Element) -> Element:
        """Get metadata of the input element."""

        locator = self.get_elem_locator(selected_element)
        if locator.count() != 1:
            logger.warning(f"can't locate element: {selected_element.get_name()}")
            return selected_element

        if selected_element.class_name == "fieldset":
            pass

        elif selected_element.tag == "select" or (selected_element.role == "combobox" and "select" in selected_element.class_name):
            select_options = self.all_select_options(selected_element)
            selected_element.options = select_options

        # <input> elements
        elif ("search" in str(selected_element.class_name).lower()) or ("search" in str(selected_element.id).lower()):
            selected_element.input_type = "search"
            selected_element.has_suggest = self.check_search_sugg(selected_element)
        elif selected_element.input_type == "file":
            #TODO
            pass

        elif selected_element.input_type == "text":
            selected_element.func_desc = locator.get_attribute('placeholder')


        # If element is part of form
        form_locator = locator.locator('xpath=ancestor::form')
        if form_locator.count():
            selected_element.contained_in = "form"

        # Check if element is in fieldset
        fieldsets = locator.locator('xpath=ancestor::fieldset').all()
        if fieldsets:
            legend_list = []
            for fieldset_locator in fieldsets:
                legend_locator = fieldset_locator.locator('xpath=descendant::legend').first
                if legend_locator.count():
                    legend = legend_locator.text_content().strip()
                    legend_list.append(legend)
            selected_element.fieldset = legend_list
            
            # fieldset but no form
            form_locator = locator.locator('xpath=ancestor::form')
            if form_locator.count() == 0:
                selected_element.contained_in = "form"
        
        selected_element.explored = True

        return selected_element

    
    def analyze_tab_section(self, website_mem: WebsiteMem, tab_element: Element, all_elements: list[Element]) -> Element:
        """Open the tab and record its content."""

        if self.env.page.url not in website_mem.pages:
            logger.warning(f"No page_mem for {self.env.page.url}")
            return tab_element
        page_mem = website_mem.pages[self.env.page.url.split('#')[0]]
        tab_locator = self.get_elem_locator(tab_element)
        if not tab_locator.count():
            logger.warning(f"{tab_element.get_name()} can't be located")
            return tab_element
        tab_element.explored = True
        self.click_element(tab_element)
        time.sleep(1.0)


        # Iterate tab section elements
        elements, _ = self.iterate_elements(
            website_mem,
            skip_first=True,
            stop_elements = all_elements,
            stop_cond = "dropdown",
            explore = False,
            print_elements=True
        )
        logger.success(f"Tab elements:")
        for elem in elements:
            print(f"{elem.get_name()}")
        
        
        base_locator = self.get_base_locator(tab_element)
        section_id = tab_locator.get_attribute('aria-controls')
        if not section_id:
            section_id = tab_locator.locator('xpath=..').get_attribute('aria-controls')
        if (not section_id):
            # describe content with VLM
            screenshot = self.env.get_screenshot()
            section_bbox = BBox()
            section_bbox.y1_abs_px = tab_element.bbox.y1_abs_px
            section_crop = crop_img(screenshot, section_bbox, abs=True, add_margin=True)
            if section_crop:
                section_crop.save(f"{self.screenshot_dir}/tab_section.png")

            tab_name = tab_element.get_name()
            vlm_prompt = f'What elements are contained under the "{tab_name}" section?'
            tab_element.func_desc = self.model_manager.vlm_call(section_crop, vlm_prompt)

            # return tab_element
            tab_element.dropdown_elements = elements
        else:
            # Create tab page_section
            section_loc = base_locator.locator(f'id={section_id}')
            if not section_loc.is_visible():
                logger.warning(f"{tab_element.get_name()} can't be located\n\n{section_loc}, {section_loc.count()}")
                return tab_element
            subsection_locs = self.split_section(section_loc)
            page_sections = self.create_page_sections(subsection_locs)
            
            # Save sections
            self.add_page_elements(page_mem, elements, sections=page_sections)
            for section in page_sections:
                section.toggle = True
                section.summary = self.summarize_section(section)
                tab_element.tab_sections.append(section)

        return tab_element


    def analyze_tabs(self, website_mem: WebsiteMem, tab_element: Element, all_elements: list[Element]) -> Element:
        """Analyze all tab elements and their content."""

        logger.info(f"analyze tab section")
        if self.env.page.url in website_mem.pages:
            page_mem = website_mem.pages[self.env.page.url.split('#')[0]]
        else:
            logger.warning(f"No page_mem for {self.env.page.url}")
            return tab_element
        
        tab_locator = self.get_elem_locator(tab_element)
        if not tab_locator.count():
            logger.warning(f"tab element {tab_element.get_name()} can't be located")
            return tab_element
        tab_element.explored = True
        
        # Get all tab elements
        all_tabs = [tab_element]
        tab_locator.focus()
        self.key_press_action([], 'ArrowRight')
        selected_element = self.get_foc_elem()
        while (not selected_element.in_list(all_elements+all_tabs)):
            if len(all_tabs) > 20:
                break
            print(f"add tab: {selected_element.get_name()}")
            all_tabs.append(selected_element)
            self.key_press_action([], 'ArrowRight')
            selected_element = self.get_foc_elem()

        # Save tab elements
        self.add_page_elements(page_mem, all_tabs)

        # Analyze tab content sections
        all_tabs.reverse()
        for tab_elem in all_tabs:
            tab_elem = self.analyze_tab_section(website_mem, tab_elem, all_elements+all_tabs)

        return tab_element


    def check_modal(self, element: Element=None, close_popup=False) -> bool:
        """Check if a modal dialog window is open."""
        
        dialog_loc = None

        try:
            base_locator = self.env.page.locator('body')
            modal_locs = base_locator.locator('[role="dialog"]').all()
            modal_locs += base_locator.locator('*.modal').all()
            for frame in self.env.page.frames:
                try:
                    frame.inner_html('body', timeout=1000)
                    modal_locs += frame.locator('[role="dialog"]').all()
                except Error as e:
                    continue

            for dialog in modal_locs:
                aria_modal = dialog.get_attribute('aria-modal')
                bounding_box = dialog.bounding_box()
                view_h = self.env.viewport_size['height']
                view_w = self.env.viewport_size['width']
                if (aria_modal != 'true') and ((view_h>=720) and (view_w<=1280)): 
                    if (bounding_box) and (bounding_box['y']>720 or bounding_box['x']>1250):
                        continue
                if self.page_obs:
                    dialog_class = dialog.get_attribute('class')
                    dialog_tag = dialog.evaluate('element => element.tagName.toLowerCase()')
                    is_page_section = False
                    for section in self.page_obs.html_sections:
                        if (section.type == dialog_tag) and (section.class_name == dialog_class):
                            is_page_section = True
                    if is_page_section:
                        continue
                
                if dialog.is_visible():
                    dialog_loc = dialog
                elif dialog.locator('xpath=child::div').first.is_visible():
                    dialog_loc = dialog.locator('xpath=child::div').first
                else:
                    continue
                
                if element:
                    element.dialog = True
                if close_popup:
                    dialog_buttons = dialog.locator('xpath=descendant::button | descendant::a').all()
                    words = ['ok', 'confirm', 'continue', 'accept', 'agree', 'allow', 'yes', 'close', 'cancel', 'dismiss', 'exit']
                    close_button = None
                    for button in dialog_buttons:
                        try:
                            class_name = str(button.get_attribute('class')).lower()
                            text = button.inner_text()
                            if not text:
                                text = button.get_attribute('aria-label')
                            text = str(text).strip().lower()
                            text_words = text.split(' ')
                        except Error as e:
                            continue
                        if any(w in text_words for w in words) or ('close' in class_name):
                            close_button = button
                            break
                    if close_button:
                        close_button.click(force=True)
                    self.key_press_action([], 'Escape')
        
        except Error as e:
            logger.warning(f"{str(e)}")
            if close_popup:
                self.key_press_action([], 'Escape')
        
        return dialog_loc
    

    def explore_dropdown(
        self, 
        website_mem: WebsiteMem, 
        element: Element,
        stop_elements: list[Element],
        selected_element: Element = None,
        next_element: Element = None
    ) -> list[PageMem]:
        """Get list of dropdown menu elements"""

        logger.debug(f"Iterate dropdown")

        start_url = self.env.page.url
        elem_loc = self.get_elem_locator(element, click_select=False)
        if not elem_loc:
            return []

        if next_element:
            stop_elements.append(next_element)
        if (element.text == "All") or (element.parent_menu and element.parent_menu.text=="All"):
            max_elements = 200
        else:
            max_elements = MAX_ELEMENTS
        stop_elements.append(website_mem.all_elements[0])

        dropdown_elements, new_pages = self.iterate_elements(
            website_mem, 
            stop_elements = stop_elements, # stop if cycle
            stop_cond = "dropdown", 
            explore = False,
            max_elements = max_elements,
            # KEY=key,
        )
        if len(dropdown_elements) >= 3:
            if (dropdown_elements[-2].equals(dropdown_elements[0])) or (dropdown_elements[-1].equals(dropdown_elements[-3])):
                dropdown_elements = dropdown_elements[:-2]
        element.dropdown_elements = dropdown_elements
        if len(element.dropdown_elements) > 100:
            website_mem.big_elements.append(element)

        # DEBUG
        logger.info(f"{element.get_name()} dropdown menu items:")
        for elem in element.dropdown_elements:
            elem.meta_data = 'menu_item'
            elem.parent_menu = element
            print(elem.get_name())

        if element.get_name() == 'History':
            return new_pages
        if (element.bbox.y2_abs_px < 175 or element.bbox.x2_abs_px < 100 or element.index<=10) and (element.meta_data != 'menu_item'):
            # Explore nav menu items
            logger.success("Explore dropdown menu")
            for item in element.dropdown_elements:
                if item.explored or (item.text == element.text) or (item.href=='#'):
                    continue
                menu_expanded = True
                elem_loc = self.get_elem_locator(element)
                if elem_loc.count() == 1:
                    if (elem_loc.get_attribute('aria-expanded') != 'true'):
                        menu_expanded = False

                item_loc = self.get_elem_locator(item, click_select=False)
                if (not item_loc.count()) or (not menu_expanded):
                    # reopen menu
                    success = self.click_element(element)
                    if not success:
                        logger.error(f"Can't click menu element")
                        break
                    if len(element.dropdown_elements) > 20:
                        x, y = element.bbox.get_center_xy_abs()
                        for i in range(item.index // 15):
                            time.sleep(0.5)
                            self.env.page.mouse.move(x, y+200)
                            self.env.page.mouse.wheel(0.0, 500.0)
                    time.sleep(1.0)  # wait for menu to open
                    if not item_loc.count():
                        success = self.click_element(element)
                        if not success: break
                        time.sleep(1.0)
                
                if item.collapse:
                    continue
                if item.index >= len(element.dropdown_elements) - 1:
                    stop_elem = element.dropdown_elements[0]
                else:
                    stop_elem = element.dropdown_elements[item.index + 1]
                stop_elems = stop_elements + [stop_elem]
                new_pages += self.explore_element(website_mem, item, stop_elems)
                if item.dropdown_elements:
                    element.meta_data = 'nested menu'

        success = self.click_element(element)
        self.key_press_action([], 'Escape')
        time.sleep(0.5)
        _, _ = self.go_to_url_action([], start_url)

        return new_pages
    

    def explore_element(
        self, 
        website_mem: WebsiteMem,
        element: Element, 
        all_elements: list[Element],
        locator = None,
        start_url: str = None,
        explore_table = False,
    ) -> list[PageMem]:
        """See what happens after clicking the element."""

        new_pages = []

        if self.intent:
            return []
        if not start_url:
            start_url = self.env.page.url.split('#')[0]
        page_mem = None
        if start_url in website_mem.pages:
            page_mem = website_mem.pages[start_url]
        elem_name = element.get_name().lower()
        KEY = 'Tab'
        if element.iter_key == 'ArrowDown':
            KEY = 'ArrowDown'
        logger.debug(f"Explore element: <{element.tag}> {elem_name}")

        if element.tag == "body":
            element.skip = True
            return []
        if element.role == "menu" and element.tag == "ul":
            logger.success(f"Get all menu items")
            return self.all_list_pages(website_mem, element)
        if element.href:
            if element.href.startswith('tel:'):
                return []
            if get_website_url(start_url).endswith(element.href):
                print("Homepage")
                element.func_desc = "Homepage link"
                return []
            if (get_website_url(element.href) != get_website_url(self.env.page.url)) and (element.href.startswith('h')):
                print("External link")
                print(f"href: {get_website_url(element.href)}")
                print(f"self.env.page.url: {get_website_url(self.env.page.url)}")
                element.func_desc = f"External link: {element.href}"
                return []
        # Don't click 'Sign out' or 'Delete'
        if ("sign" in elem_name or "log" in elem_name) and ("out" in elem_name):
            element.func_desc = "Click to sign out"
            return []
        if (elem_name in ["login", "log in", "sign in", "create account", "register"]):
            return []
        if ("delete" in elem_name) or ("save" in elem_name) or ("remove" in elem_name):
            element.func_desc = "Save changes"
            return []
        if ("collapse" in elem_name) or ("close" in elem_name):
            logger.debug(f"collapse button")
            return []
        if ("next" in elem_name) or ("prev" in elem_name):
            logger.debug(f"next button")
            return []
        if (elem_name == "add to wish list"):
            return []
        if (elem_name.startswith("print")):
            return []
        if (element.input_type in ["submit", "file"]) or ("subscribe" in elem_name):
            logger.debug(f"submit button")
            return []
        if (element.tag == "button") and (not element.aria_expanded):
            if ('table' in elem_name) and (not explore_table):
                logger.debug(f"table button")
                return []
            if ("submit" in elem_name) or ("submit" in str(element.class_name).lower()) or ("submit" in str(element.id)):
                logger.debug(f"submit button")
                return []
            if ("form" in str(element.class_name).lower()):
                logger.debug(f"form button")
                return []
            if ("file" in elem_name) or ("dark mode" in elem_name):
                logger.debug(f"file button")
                return []
            if ('upload' in str(element.text).lower()) or ('upload' in str(element.class_name).lower()):
                logger.debug(f"upload button")
                element.input_type = 'file'
                return []
        if element.tag in ['input', 'textarea', 'select']:
            element = self.analyze_input(website_mem, element)
            return []


        # locate element on page
        if locator:
            element_locator = locator
        else:
            element_locator = self.get_elem_locator(element)
        if (element_locator.count() == 0):
            logger.warning(f"Can't locate element: {elem_name}\nLocator count = {element_locator.count()}\n{element_locator}")
            return []
        elif (element_locator.count() > 1):
            element_locator = element_locator.first
        parent_loc = element_locator.locator('xpath=ancestor::div[1]').first
        if not parent_loc.count():
            logger.debug(f"no parent_loc")
            return []
        parent_bbox = parent_loc.bounding_box()

        if (element_locator.get_attribute('aria-pressed')):
            logger.debug(f"toggle button")
            element.func_desc = 'toggle'
            return []
        if (not explore_table) and (element_locator.locator('xpath=ancestor::table').count()):
            logger.debug(f"table action")
            element.contained_in = 'table'
            element.func_desc = 'table action'
            return []
        if (element.tag=='button') and ('dropdown' not in element.class_name) and (element_locator.locator('xpath=ancestor::form').count()):
            logger.debug(f"form button")
            element.contained_in = 'form'
            element.func_desc = 'form action'
            return []
        
        if (element_locator.get_attribute('role') == 'tab') or (element_locator.locator('xpath=..').get_attribute('role') == 'tab'):
            element = self.analyze_tabs(website_mem, element, all_elements)
            return []
        if (element.tag=='a') and (str(element.href).startswith('#')) and ('dropdown' not in element.class_name) and (not element.aria_expanded):
            if (element.bbox.y2_abs_px > 175 and element.bbox.x2_abs_px > 100):
                logger.debug(f"href '#'")
                return []
        element.explored = True


        # click element
        new_page = self.click_with_page_handling(elem_loc=element_locator, button='Enter')
        # Check if new browser tab opened
        if new_page:
            element.new_page = True
            element.destination_page = new_page.url
            if (get_website_url(new_page.url) != get_website_url(self.env.page.url)):
                element.func_desc = f"External link: {new_page.url}"
            new_page.close()
            return []
        if len(self.env.context.pages) > 1:
            self.env.page = self.env.context.pages[0]
            self.env.page.bring_to_front()
        page_url = self.env.page.url.split('#')[0]


        if page_url == start_url:  ## Same page ##
            if self.check_modal(element, close_popup=True):
                print("Dialog")
                return []
            if self.get_copied_text().strip() != '':
                element.func_desc = "Copy to clipboard"
                print("Copy button")
                return []
            
            if element_locator.count() == 1:
                controls = element_locator.get_attribute('aria-controls')
                if (controls != None) and (element_locator.get_attribute('aria-expanded')):
                    menu_loc = self.get_base_locator(element).locator(f'#{controls}')
                    if menu_loc.count():
                        for action_loc in menu_loc.locator('button, a').filter(visible=True).all():
                            elem = self.elem_from_loc(action_loc)
                            if elem:
                                element.dropdown_elements.append(elem)
                                print(elem.get_name())
                        return new_pages


            # Check for dropdown menu
            if element.index < len(all_elements) - 1:
                if element.iter_key == 'Tab' or element.iter_key == 'ArrowDown':
                    next_element = all_elements[element.index + 1]
                else:
                    next_element = all_elements[element.index - 1]
            else:
                next_element = element.next_element

            selected_element = self.get_foc_elem()
            selected_loc = self.get_elem_locator(selected_element)

            if (selected_loc.count() == 1) and (selected_element.equals(element)):
                if ('select' in selected_element.class_name.lower()) or ('select' in selected_element.text.lower()):
                    self.check_dropdown(element)
            if (selected_element != None) and (not selected_element.equals(element)):
                if not selected_element.in_list(all_elements):
                    if selected_element.tag == "textarea":
                        self.env.page.keyboard.type("test input")
                    menu_pages = self.explore_dropdown(website_mem, element, all_elements, selected_element, next_element)
                    new_pages += menu_pages
                    return new_pages
            self.key_press_action([], KEY)
            selected_element = self.get_foc_elem()
            if selected_element == None:
                return new_pages
            if selected_element.equals(next_element) and (element.index < len(all_elements)-1):
                self.key_press_action([], KEY)
                next_element = all_elements[element.index + 2] if (element.index < len(all_elements)-2) else all_elements[element.index + 1].next_element
                selected_element = self.get_foc_elem()
                if selected_element == None:
                    return new_pages

            if (selected_element.in_list(website_mem.all_elements[:3], class_only=True)):
                print("Current page link")
                element.destination_page = self.env.page.url
            elif not selected_element.equals(next_element):
                if not parent_loc.count(): new_parent_bbox = parent_bbox
                else: new_parent_bbox = parent_loc.bounding_box()
                if parent_bbox and new_parent_bbox:
                    if (parent_bbox['height'] > 200 and new_parent_bbox['height'] < 100) or (parent_bbox['width'] > 200 and new_parent_bbox['width'] < 100):
                        print("Collapse Menu")
                        element.collapse = True
                        self.click_element(element)
                        return []
                menu_pages = self.explore_dropdown(website_mem, element, all_elements, selected_element, next_element)
                new_pages += menu_pages

        else:  ## New page ##
            if get_website_url(self.env.page.url) != get_website_url(start_url):
                print("External link")
                element.func_desc = f"External link: {element.href}"
                _, _ = self.go_to_url_action([], start_url)
                return []
            
            # Create new PageMem if page hasn't been visited
            if (page_url not in website_mem.pages) and (not self.pages_match(page_mem)):
                logger.info(f"Create PageMem for {page_url}")
                new_page_mem = self.create_page_mem(website_mem)
                new_page_mem.name = element.get_name().strip()
                element.destination_page = self.env.page.url
                new_pages.append(new_page_mem)

            if element_locator.count() == 1:
                if element_locator.locator('xpath=ancestor::aside').count() and element.parent_tag == 'li':
                    element_locator.focus()
                    self.key_press_action([], KEY)
                    selected_element = self.get_foc_elem()
                    next_element = all_elements[element.index + 1]
                    if selected_element != next_element:
                        new_pages += self.explore_dropdown(website_mem, element, all_elements, selected_element, next_element)

        # Go back to starting page
        _, _ = self.go_to_url_action([], start_url)

        return new_pages
    

    #### ---- Page Exploration ---- ####

    def check_content_stop(
        self, 
        page_mem: PageMem, 
        element: Element,
        all_elements: list[Element] = None,
        debug: bool = True,
    ) -> bool:
        """Returns True when tab iteration reaches content section."""

        # element_locator = self.env.page.locator('*:focus')
        element_locator = self.get_elem_locator(element)

        if element_locator.count():
            table = element_locator.locator('xpath=ancestor::table').nth(0)
            if table.count() and table.bounding_box()['height'] > 200:
                element.contained_in = 'table'
                element.func_desc = 'Table action button'
                if debug: logger.debug(f"STOP {element.get_name()}: table element\n")
                return True
            if element_locator.locator(f'xpath=ancestor::*[@role="code"]').count():
                element.contained_in = 'code'
                element.func_desc = 'Code file editor'
                if debug: logger.debug(f"STOP {element.get_name()}: code editor\n")
                return True
            if element_locator.locator('xpath=ancestor::p').count():
                if len(element_locator.locator('xpath=ancestor::p').nth(0).inner_text()) > 100:
                    element.contained_in = 'p'
                    element.func_desc = 'Text hyperlink'
                    if debug: logger.debug(f"STOP {element.get_name()}: paragraph element\n")
                    return True
        else:
            logger.warning("no elem selected, continue")
            return False

        
        in_list = False
        if not page_mem:
            list_section = None
        else:
            list_section = page_mem.list_section

        if list_section:
            if element.bbox.center_in_bbox(list_section.bbox):
                if list_section.type == 'list':
                    in_list = element_locator.locator(f'xpath=ancestor::{list_section.list_item_tag}').count() > 0
                elif not list_section.class_name:
                    ancestor_loc = element_locator.locator(f'xpath=ancestor::{list_section.type}')
                    if (ancestor_loc.count()):
                        in_list = ancestor_loc.first.bounding_box()['height'] > 150
                else:
                    ancestor_xpath = f"ancestor::{list_section.type}[contains(concat(' ', normalize-space(@class), ' '), ' {list_section.class_name} ')]"
                    ancestor_locator = element_locator.locator(f'xpath={ancestor_xpath}')
                    in_list = ancestor_locator.count() > 0
                    

        if in_list:
            element.contained_in = 'list'
            if debug:
                logger.debug(f"STOP {element.get_name()}: list section")
                logger.debug(f"{element.bbox.get_abs_px_coords()}\n")
            return True
        
        
        return False


    def check_stop_cond(
        self, 
        website_mem: WebsiteMem, 
        element: Element, 
        stop_elements: list[Element], 
        all_elements: list[Element], 
        cond: str = "content"
    ) -> bool:
        """Returns True when iteration should be stopped, False otherwise."""

        if stop_elements is not None:
            repeat_elem = element.in_list(stop_elements)
            if (repeat_elem is not None):
                if repeat_elem.role == 'tooltip':
                    pass
                else:
                    print(f"stop_elem: {repeat_elem.get_name()}")
                    return True
        
        if element.in_list(all_elements) and element.index >= 3:
            if (all_elements[-1].in_list(all_elements[:-1], exact=True)) and (all_elements[-2].in_list(all_elements[:-2], exact=True)):
                print("3 repeat")
                return True


        # check if element is in list section
        if cond == "content":
            url = self.env.page.url

            if url not in website_mem.pages:
                logger.warning(f"page_mem not found for {url}, won't check list stop")
                page_mem = None
            else:
                page_mem = website_mem.pages[url]

            return self.check_content_stop(page_mem, element, all_elements)
        
        return False


    def iterate_elements(
        self,
        website_mem: WebsiteMem,
        start_element: Element = None,
        stop_elements: list[Element] = None,
        # reverse: bool = False,
        KEY: str = 'Tab',
        skip_first: bool = False,
        stop_cond: str = 'content',
        max_elements: int = MAX_ELEMENTS, # default = 75
        explore: bool = True,
        repeat: bool = False,
        print_elements = False,
    ) -> tuple[list[Element], list[PageMem]]:
        """Use 'Tab' key to iterate over the clickable elements and analyze them."""

        logger.debug(f"Iterating elements...")
        start_url = self.env.page.url

        all_elements = []
        explore_elements = []
        new_pages = []

        # Press 'Tab' to iterate forward, 'Shift+Tab' to iterate backward
        if KEY not in ['Tab', 'Shift+Tab', 'ArrowDown', 'ArrowUp']:
            KEY = 'Tab'

        # Select first element on page
        selected_element = self.get_foc_elem()
        if (not selected_element) or (selected_element.tag == "body"):
            self.key_press_action([], KEY)
        if start_element:
            element_locator = self.get_elem_locator(start_element)
            element_locator.focus()
        if skip_first:
            self.key_press_action([], KEY)
        selected_element = self.get_foc_elem()
        if not selected_element:
            self.key_press_action([], KEY)
            selected_element = self.get_foc_elem()
            if not selected_element:
                logger.warning("No element selected: breaking")
                return [], []
        if ('keyboard-navigatable' in str(selected_element.class_name)):
            KEY = 'ArrowDown'


        index = 0
        repeat_count = 0
        while index < max_elements:
            if (index % 15 == 0) and (index > 0) and (stop_cond=='dropdown'):
                coords = selected_element.bbox.get_center_xy_abs()
                self.env.page.mouse.move(coords[0], coords[1])
                self.env.page.mouse.wheel(0.0, 500.0)
            if selected_element is None:
                logger.warning("No element selected: breaking")
                break
            selected_element.index = index
            if self.check_stop_cond(website_mem, selected_element, stop_elements, all_elements, cond=stop_cond):  #and index > 0
                break

            # Check if element seen before on website
            saved_element = selected_element.in_list(website_mem.all_elements)
            if saved_element is None:
                website_mem.all_elements.append(selected_element)
            if (saved_element is not None) and (repeat == False):
                if (not self.intent) and (saved_element.dropdown_elements) and (saved_element.href!='#') and (not saved_element.destination_page) and (saved_element.func_desc != "big_element"):
                    explore_elements.append(selected_element)
                    website_mem.all_elements.append(selected_element)
                else:
                    print("Already saved")
                    selected_element = self.update_elem(new_elem=selected_element, saved_element=saved_element)

            elif selected_element.is_skip():
                selected_element.skip = True
                print(f"Skip link")
            elif selected_element.bbox.not_in_view():
                selected_element.skip = True
                print(f"Not in viewport")

            elif selected_element.tag in ['input', 'textarea', 'select'] or selected_element.role == 'combobox' or selected_element.class_name=='fieldset':
                selected_element = self.analyze_input(website_mem, selected_element)
            else:
                explore_elements.append(selected_element)
            
            # Add Element
            if print_elements:
                selected_element.print_element()
            print(f"{selected_element.get_name()} {selected_element.bbox.get_abs_px_coords()}")
            selected_element.iter_key = KEY
            all_elements.append(selected_element)
            
            # Select next element
            prev_element = selected_element
            self.key_press_action([], KEY)
            # time.sleep(0.5)
            selected_element = self.get_foc_elem()

            if selected_element is None:
                retries = 0
                while (selected_element is None) and (retries < 3):
                    self.key_press_action([], KEY)
                    selected_element = self.get_foc_elem()
                    retries += 1

                if not selected_element:
                    self.get_elem_locator(prev_element).focus()
                    break
                if selected_element.in_list(all_elements):
                    self.get_elem_locator(prev_element).first.focus()
                    break

            if selected_element.tag == "body":
                self.key_press_action([], KEY)
                selected_element = self.get_foc_elem()
                if selected_element.tag == "body":
                    break

            # Break loop if stuck
            if selected_element.equals(prev_element):
                logger.debug("repeated element")
                repeat_count += 1
            else:
                repeat_count = 0
            if repeat_count >= 3:
                logger.warning("Element iteration stuck: breaking")
                break
            index += 1
        self.env.page.locator('body').focus()
        self.env.page.locator('body').blur()
        logger.debug(f"Finished iterating elements.")


        # Try clicking elements for further analysis
        if explore == True:
            print(f"explore_elements: {len(explore_elements)}")
            if all_elements:
                all_elements[-1].next_element = selected_element
            
            explore_error_elems = []
            for element in explore_elements:
                if not self.exploration_start_time:
                    self.exploration_start_time = time.monotonic()
                if (time.monotonic() - self.exploration_start_time > self.cfg.exploration_time_limit):
                    logger.warning(f"Exceeded exploration time limit: {self.cfg.exploration_time_limit}s")
                    break
                try:
                    new_pages += self.explore_element(website_mem, element, all_elements.copy(), start_url=start_url)
                except Error as e:
                    element.func_desc = "Error occured during exploration"
                    explore_error_elems.append(element)
                    logger.error(f"Exception occured while exploring element {element.get_name()}\n\n{str(e)}")
                    # input("\ncontinue")
                    self.go_to_url_action([], start_url)
            website_mem.error_elements += explore_error_elems
        

        return all_elements, new_pages
    

    def explore_list(self, website_mem: WebsiteMem, page_mem: PageMem, end_elements=[]) -> list[PageMem]:
        """Explore elements of one list item."""

        start_url = self.env.page.url
        list_section = page_mem.list_section
        if list_section == None:
            return []
        section_loc = self.get_section_locator(list_section)
        if not section_loc.count():
            print("no section_loc")
            return []
        if list_section.type == 'list':
            item_loc = section_loc.all()[0]
        else:
            item_loc = section_loc.locator('xpath=child::li').nth(1)

        # Get item link elements
        list_elements = self.element_split(item_loc, split_p=True)
        action_elements = []
        locators = []
        for elem_loc in list_elements:
            try:
                elem_loc.focus()
                element = self.get_foc_elem()
            except:
                continue
            if element == None:
                continue
            if (element.tag not in ['input', 'a', 'button']) or (element.href == '#'):
                continue
            action_elements.append(element)
            locators.append(elem_loc)
        if action_elements:
            self.key_press_action([], 'Tab')
            next_elem = self.get_foc_elem()
            action_elements[-1].next_element = next_elem

        # Explore link elements
        new_pages: list[PageMem] = []
        for i in range(len(action_elements)):
            element = action_elements[i]
            element.index = i
            locator = locators[i]
            # Explore
            stop_elements = action_elements
            if end_elements:
                stop_elements.append(end_elements[0])
            new_pages += self.explore_element(website_mem, element, stop_elements, locator=locator, start_url=start_url)
            # Add element to list section mem
            if not element.in_list(list_section.elements, class_only=True):
                list_section.elements.append(element)
        for page in new_pages:
            page.is_list_item = True
            page.add_bookmark = False

        return new_pages


    def explore_table(self, website_mem: WebsiteMem, page_mem: PageMem, table_elem: Element, end_elements=[]) -> list[PageMem]:
        """Analyze the structure and content of the table and explore actions."""
        
        start_url = self.env.page.url
        table_section = None
        for page_section in page_mem.html_sections:
            if (table_elem.bbox.center_in_bbox(page_section.bbox)) or (page_section.type == "table"):
                table_section = page_section
                if table_section.type != 'fieldset':
                    table_section.list_type = 'table'
                break
        if not table_section:
            logger.warning(f"no table section")
            return []
        
        base_loc = self.get_base_locator(table_elem)
        row_loc = base_loc.locator("table > tbody > tr").nth(2)
        row_actions = row_loc.locator('xpath=descendant::a').all()

        action_elements = []
        locators = []
        for elem_loc in row_actions:
            tag = elem_loc.evaluate('element => element.tagName.toLowerCase()')
            if (tag not in ['input', 'a', 'button']) or (elem_loc.get_attribute('href') in ['#', 'javascript:void(0)']):
                continue
            elem_loc.focus()
            element = self.get_foc_elem()
            if element == None:
                continue
            action_elements.append(element)
            locators.append(elem_loc)
        
        if action_elements:
            self.key_press_action([], 'Tab')
            next_elem = self.get_foc_elem()
            action_elements[-1].next_element = next_elem
            self.key_press_action([], 'Shift+Tab')
        else:
            logger.warning(f"No action elements")
            print(f"row_loc: {row_loc}\n{row_loc.count()}")
        
        # Explore link elements
        new_pages: list[PageMem] = []
        for i in range(len(action_elements)):
            element = action_elements[i]
            element.index = i
            locator = locators[i]
            # Explore
            stop_elements = action_elements
            if end_elements:
                stop_elements.append(end_elements[0])
            new_pages += self.explore_element(website_mem, element, stop_elements, locator=locator, start_url=start_url, explore_table=True)
        for page in new_pages:
            page.is_list_item = True
            page.add_bookmark = False

        return new_pages


    def iterate_page(
        self,
        website_mem: WebsiteMem,
        page_mem: PageMem
    ) -> tuple[list[Element], list[PageMem]]:
        """"""

        all_elements = []
        all_new_pages = []

        # Tab iterate to analyze navbar
        begin_elements, begin_pages = self.iterate_elements(website_mem)
        if not begin_elements:
            return [], []
        all_elements += begin_elements
        all_new_pages += begin_pages
        

        if begin_elements[-1].next_element:
            logger.success(f"Reverse iterate to get end_elements")
            if not self.intent:
                self.go_to_url_action([], self.env.page.url)
            else:
                first_loc = self.get_elem_locator(all_elements[0]).first
                if first_loc.count():
                    first_loc.focus()
            end_elements, end_pages = self.iterate_elements(
                website_mem,
                stop_elements=all_elements[1:],
                KEY='Shift+Tab'
            )
            end_elements.reverse()
            end_pages.reverse()

            # Analyze content section
            list_pages = []
            content_element = begin_elements[-1].next_element
            if (content_element.contained_in == 'list') or (page_mem.list_section):
                logger.success(f"Analyze list elements")
                list_pages = self.explore_list(website_mem, page_mem, (end_elements+all_elements))
            elif content_element.contained_in == 'table':
                logger.success(f"Analyze table")
                list_pages = self.explore_table(website_mem, page_mem, content_element, (end_elements+all_elements))
            elif content_element.contained_in == 'code':
                logger.success(f"Code editor section")
                all_elements.append(content_element)
                pass
            elif content_element.contained_in == 'p':
                logger.success(f"Paragraph section")
                pass

            all_elements += end_elements
            all_new_pages += (list_pages + end_pages)

        logger.success(f"Done exploring page: {self.env.page.url}\n\n")

        return all_elements, all_new_pages


    def check_missing(self, website_mem: WebsiteMem, section: PageSection) -> list[Element]:
        """Check for icon elements with <i> tag that don't show up in axtree."""

        icon_elements = []

        section_loc = self.get_section_locator(section)
        if section_loc.count() != 1:
            return []

        icons = section_loc.locator('i').filter(visible=True).all()
        for i_loc in icons:
            if not i_loc.count():
                continue
            if i_loc.text_content():
                continue
            table = i_loc.locator('xpath=ancestor::table').nth(0)
            if table.count():
                continue
            elem = self.elem_from_loc(i_loc, iframe_id=section.iframe_id)
            if not elem:
                continue
            if 'rating' in str(elem.class_name):
                i_loc.click()

            parent = i_loc.locator('xpath=../..')
            text = parent.inner_text().strip()
            label_loc = parent.locator('xpath=preceding-sibling::label')
            if label_loc.count():
                label = label_loc.inner_text().strip()
                text = f"{label} {text}"
            if text == "":
                continue
            # elem.text = text
            elem.label = text
            elem.func_desc = text

            form_locator = i_loc.locator('xpath=ancestor::form')
            if form_locator.count():
                elem.contained_in = "form"

            # Check if element is already saved
            saved = False
            for e in section.elements:
                if (elem.bbox.center_in_bbox(e.bbox)) or (elem.get_name() in str(e.get_name())):
                    saved=True
                    break
            if saved == True:
                continue
            
            if elem.get_name().startswith('Sort'):
                self.click_element(elem, section)
                elem.dropdown_elements = self.get_menuitems(elem, section)
            
            logger.info(f"icon: {elem.class_name} text={clean_text(text)} {i_loc.bounding_box()}")
            icon_elements.append(elem)
            website_mem.all_elements.append(elem)
        
        section.elements += icon_elements

        return icon_elements
    

    def save_element_mem(self, page_mem: PageMem, element: Element, all_section_locs, debug=False):
        """Add the element to its corresponding PageSection."""

        logger.info(f"Save element: {element.get_name()}")
        if not isinstance(element.bbox, BBox):
            logger.info("outer element")
            page_mem.outer_elements.append(element)

        elem_loc = self.get_elem_locator(element)
        if not elem_loc.count():
            logger.warning(f"Can't locate {element.get_name()}\n{elem_loc}")
            return

        element_section = None
        for page_section, section_loc in all_section_locs:
            if (element.iframe_id != page_section.iframe_id):
                continue
            if (page_section.id) and (page_section.id == element.id):
                element_section = page_section
                break
            if section_loc.locator(f'{element.tag}').and_(elem_loc).count():
                element_section = page_section
                break
            elif element.bbox.center_in_bbox(page_section.bbox):
                element_section = page_section
                break
                
        if element_section:
            element_section.add_element(element)
        else:
            # Element not in any bbox
            logger.info("outer element")
            page_mem.outer_elements.append(element)
            
        return
    

    def add_page_elements(self, page_mem: PageMem, elements: list[Element], sections=None, debug=False):
        """"""

        if sections:
            page_sections = sections
        else:
            page_sections = page_mem.html_sections
        
        all_section_locs: list[tuple[PageSection, Locator]] = []
        for page_section in page_sections:
            section_loc = self.get_section_locator(page_section)
            if section_loc.count():
                all_section_locs.append((page_section, section_loc))
            else:
                logger.warning(f"Can't locate section: {section_loc}")

        for elem in elements:
            self.save_element_mem(page_mem, elem, all_section_locs)

        return
    

    def explore_page_v3(
        self, 
        website_mem: WebsiteMem,
        page_mem: PageMem,
        max_recurse=2,
        max_pages=MAX_NEW_PAGES,
        remake = False
    ):
        """Explore the website and save information in the Agent's memory."""

        if not self.intent:
            logger.debug(f"Change page url")
            self.go_to_url_action([], page_mem.url)
        self.env._wait_dom_loaded(networkidle=True)
        url = self.env.page.url
        if (url in website_mem.pages) and (not remake):
            if website_mem.pages[url].html_sections:
                logger.success(f"Page {url} already saved.\n\n")
                return
        if url.split('.')[-1] in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp']:
            return
        logger.success(f'Explore page: {page_mem.name} {page_mem.url}')
        
        # divide page into sections
        if not page_mem.html_sections:
            page_mem = self.divide_page(website_mem, page_mem)
        website_mem.set_page_mem(page_mem)
        
        # Check if page matches common structure
        page_type = self.matching_list_page(website_mem, page_mem)
        if (page_type) and (not self.intent):
            logger.success(f"Match {page_type.name}\n\n")
            page_type.similar_urls.append(page_mem.url)
            page_mem.list_page = page_type.url
            return
        
        if (page_mem.is_list_item) and (self.save_mem):
            logger.success("New List page\n\n")
            page_mem.similar_urls = [page_mem.url]
            website_mem.list_pages.append(page_mem)

        
        # Explore elements on the page
        all_elements, all_new_pages = self.iterate_page(website_mem, page_mem)

        # Add elements to their corresponding PageSection in memory
        self.add_page_elements(page_mem, all_elements)
        for section in page_mem.html_sections:
            try:
                self.check_missing(website_mem, section)
            except Error as e:
                continue

        if (not self.intent) or (self.save_mem):
            self.agent_save_mem(website_mem)


        # Explore new pages until max pages or max depth is reached
        if (self.env.page.url == website_mem.url) and (len(all_elements) <= 15):
            max_recurse = max_recurse
        elif (self.env.page.url == website_mem.url) and (len(all_elements) >= 75):
            max_recurse = max_recurse - 2
            logger.info(f"n_elements={len(all_elements)}, set max_recurse to {max_recurse}")
        else:
            max_recurse = max_recurse - 1

        if max_recurse < 0:
            return
        
        for new_page in all_new_pages:
            if (time.monotonic() - self.exploration_start_time > self.cfg.exploration_time_limit):
                logger.warning(f"Exceeded exploration time limit: {self.cfg.exploration_time_limit}s")
                break
            if len(website_mem.pages) > max_pages:
                break
            if page_mem.is_list_item and max_recurse < 1:
                new_page.is_list_item = True

            error_pages = []
            try:
                self.explore_page_v3(website_mem, new_page, max_recurse)
            except Error as e:
                error_pages.append(new_page.url)
                logger.error(f"Exception occured while exploring page {new_page.url}:\n\n{str(e)}")
                logger.error(traceback.format_exc())
                # input("\ncontinue")
            n_sections = len(new_page.html_sections)
            if new_page.list_section:
                n_sections += 1
            if (n_sections < 3) or (n_sections > 20):
                error_pages.append(new_page.url)

            website_mem.error_pages += error_pages
            if error_pages:
                print()
                logger.error(f"error pages: {website_mem.error_pages}")

        return

    


    #### ---- Summarize memories ---- ####

    def summarize_section(self, section: PageSection, vlm_prompt: str=None, text_summary=False) -> str:
        """For section choice, choose_navigate, and page_summary."""

        if not self.cfg.split_page_sections:
            return ""

        website_name = site_name(self.env.page.url)
        if not section.bbox:
            logger.warning(f"No section bbox")
            return ''
        if (section.type == 'img') and (section.bbox.get_abs_px_width() < 50):
            return ""
        section_loc = self.get_section_locator(section)
        has_dropdown = (section.n_dropdowns > 0)
        is_small = (section.bbox.get_abs_px_height()<175) or (section.bbox.get_abs_px_width()<100)
        is_sidebar = (section.id=='sidebar') or ('sidebar' in str(section.class_name))
        if self.cfg.llm_only:
            text_summary = True

        if (is_small or has_dropdown) and (not section.list_type) or (is_sidebar) or (text_summary):
            # Summarize with LLM
            elements = self.useful_elements(section.elements)
            if (not elements) and (not text_summary):
                logger.info(f"No elems to summarize")
                return ""
            
            if (section.type == 'list'):
                section_text = self.list_info(section)
            else:
                section_text = self.section_content(section_loc, caption_img=False, add_quotes=False)['text']
            section.section_text = section_text
            elem_context = self.elems_context(all_elements=elements)

            system_prompt = self.prompts.summarize_section
            class_info = ""
            if section.class_name:
                class_info = f' class=\"{section.class_name}\"'
            user_prompt = f"<{section.type}>{class_info}\n\nText:\n{section_text}"
            if len(elements) < 50:
                user_prompt += f"\n\nElements:\n{elem_context}"
            messages = self.format_llm_prompt(system_prompt, user_prompt, thinking=False)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            summary = self.extract_llm_answer(llm_out, keyword="**Section Summary**:")

        else:
            # Describe wth VLM
            section_text = ""
            if section_loc.count() == 1:
                try:
                    section_text = section_loc.inner_text()
                except:
                    logger.warning(f"Can't get section text")

            # Get section screenshot
            logger.debug(f"Summarize section: {section.type} {section.class_name}")
            section_crop, _ = self.section_screenshot(section, add_margin=True)
            if not section_crop:
                if section_loc.count() != 1:
                    return ""
                else:
                    logger.warning(f"No screenshot, try text summary")
                    return self.summarize_section(section, text_summary=True)

            if vlm_prompt:
                pass
            elif (len(section_text) > 1500):
                vlm_prompt = self.prompts.what_kind_info.format(website_name=website_name)
            else:
                vlm_prompt = self.prompts.describe_section_site.format(website_name=website_name)
            vlm_desc = self.model_manager.vlm_call(section_crop, vlm_prompt)
            if not vlm_desc:
                vlm_desc = ""
            section.desc = vlm_desc

            if (len(vlm_desc.split('\n')) > 2) or (len(vlm_desc) > 250):
                # Summarize VLM description with LLM
                system_prompt = self.prompts.summarize_desc
                user_prompt = f"Website: {vlm_desc}"
                messages = self.format_llm_prompt(system_prompt, user_prompt, thinking=False)
                llm_out = self.model_manager.llm_call(messages, show_prompt=False)
                summary = self.extract_llm_answer(llm_out, keyword="Summary:")
            else:
                summary = vlm_desc
            print()

        return summary


    def summarize_page(self, website_mem: WebsiteMem, page_mem: PageMem, all_sections=True, new_page=False):
        """For planning and navigation."""

        logger.success(f"Summarize page: {page_mem.name}\n")

        if all_sections:
            for section in page_mem.html_sections:
                print(f"{section.class_name}")
                section.summary = self.summarize_section(section)
        if page_mem.list_section:
            if self.manual:
                page_mem.list_section.summary = 'list section'
            else:
                page_mem.list_section.summary = self.summarize_section(page_mem.list_section)
        
        # Scroll to top of main section
        crop_nav = False
        if (page_mem.html_sections) and (page_mem.html_sections[0].bbox.get_abs_px_height() < 100):
            upper_section = page_mem.html_sections[min(1, len(page_mem.html_sections)-1)]
            self.scroll_to_bbox(upper_section.bbox, upper_margin=0)
            if self.page_scroll_height(upper_section.iframe_id) == 0:
                crop_nav = True
        else:
            self.env.page.evaluate(f"window.scrollTo(0, 0)")

        # Generate short page summary with VLM
        if not self.cfg.llm_only:
            screenshot = self.env.get_screenshot()
            if crop_nav:
                nav_h = 0
                if page_mem.html_sections:
                    nav_h = page_mem.html_sections[0].bbox.get_abs_px_height()
                screenshot = crop_img(screenshot, top=nav_h)
            website_name = site_name(page_mem.url)
            vlm_prompt = self.prompts.site_page_summary.format(website_name=website_name)
            description = self.model_manager.vlm_call(screenshot, vlm_prompt)
            page_mem.short_summary = description

        # More detailed page summary
        if not self.intent:
            prompt = self.prompts.site_page_report.format(
                website_name=site_name(page_mem.url), 
                page_name=page_mem.name)
            max_h = 1280 if page_mem.page_height<4000 else 720
            page_summary = self.model_manager.vlm_call(screenshot, prompt)
            page_mem.page_summary = page_summary

        # Scroll back to top
        self.env.page.evaluate(f"window.scrollTo(0, 0)")
        if (not self.intent) or (self.save_mem):
            website_mem.set_page_mem(page_mem)
            self.agent_save_mem(website_mem)

        return
    
    
    def update_elem(self, new_elem: Element, saved_element: Element) -> Element:
        """Return copy of saved_element with updated values."""

        if (not new_elem) or (not saved_element):
            return None

        elem_copy = copy.copy(saved_element)

        elem_copy.nth = new_elem.nth
        elem_copy.id = new_elem.id

        elem_copy.href = new_elem.href
        if new_elem.href == saved_element.href:
            elem_copy.destination_page = saved_element.destination_page
        elem_copy.aria_label = new_elem.aria_label
        elem_copy.aria_controls = new_elem.aria_controls
        elem_copy.aria_haspopup = new_elem.aria_haspopup
        elem_copy.aria_autocomplete = new_elem.aria_autocomplete
        elem_copy.aria_selected = new_elem.aria_selected
        elem_copy.title = new_elem.title
        elem_copy.text = new_elem.text
        elem_copy.bbox = new_elem.bbox
        elem_copy.name = new_elem.name

        elem_copy.input_value = new_elem.input_value
        elem_copy.input_type = new_elem.input_type
        if (saved_element.tag=="input") and (saved_element.role=="combobox"):
            if saved_element.input_value:
                elem_copy.input_value = saved_element.input_value
        if new_elem.options:
            elem_copy.options = new_elem.options
        elem_copy.placeholder = new_elem.placeholder

        elem_copy.required = new_elem.required
        elem_copy.contained_in = new_elem.contained_in

        elem_copy.clicked = saved_element.clicked
        elem_copy.input_edited = saved_element.input_edited
        elem_copy.is_disabled = new_elem.is_disabled

        elem_copy.func_desc = saved_element.func_desc

        return elem_copy


    def update_elements(
        self, 
        elements: list[Element], 
        section: PageSection=None, 
        first_only=False, 
        section_strict=False,
        get_all=True,
        inputs_only=False,
        debug=False
    ) -> dict[str, Any]:
        """Get updated list of elements on page by checking saved locators."""

        if not get_website_url(self.env.page.url):
            return {"current": [], "removed": [], "added": [], "modified": []}
        website_mem = self.website_mem

        if (section):
            if len(elements)>25:
                return {"current": elements, "removed": [], "added": [], "modified": []}

        current_elements = []
        removed_elements = []
        added_elements = []
        modified_elements = []
        other_elems = []

        for element in elements:
            if (inputs_only) and (element.tag not in ["input", "textarea", "select", "label"]) and (element.role != "tab") and (element.input_type!="submit"):
                if (not element.dropdown_elements) and ('dropdown' not in element.class_name):
                    other_elems.append(element)
                    continue
            if (element.in_list(current_elements, check_coord=True)):
                continue
            if element.func_desc == "big_element":
                element = element.in_list(website_mem.big_elements)
            if debug:
                logger.debug(f"element: {element.tag} {element.get_name()}")
            
            if (section) and (section.nth!=None):
                elem_loc = self.get_elem_locator(element, section=section, first_only=first_only, section_strict=section_strict)
            else:
                elem_loc = self.get_elem_locator(element)
                if elem_loc.count() > 1:
                    elem_loc = self.get_elem_locator(element, section=section, first_only=first_only, section_strict=section_strict)

            if elem_loc.count() == 1:
                # Add saved element
                new_element = self.elem_from_loc(elem_loc, section=section, iframe_id=element.iframe_id)
                if new_element == None:
                    removed_elements.append(element)
                    logger.warning(f"Element==None: {element.get_name()}")
                    continue
                if (element.contained_in == 'form') and (element.name != new_element.name):
                    removed_elements.append(element)
                    added_elements.append(new_element)
                    current_elements.append(new_element)
                    continue
                if (str(new_element.input_value) != str(element.input_value)):
                    print(f'{element.get_name()} value: "{element.input_value}" to "{new_element.input_value}"')
                    modified_elements.append(new_element)
                    new_element.input_edited = True
                new_element = self.update_elem(new_element, saved_element=element)
                current_elements.append(new_element)
            elif elem_loc.count() > 1:
                removed_elements.append(element)
                if element.tag == 'i':
                    added_elements += self.check_missing(website_mem, section)
                    break
                if (not get_all) or (first_only):
                    continue
                all_loc = elem_loc.all()
                for loc in all_loc:
                    new_element = self.elem_from_loc(loc, iframe_id=element.iframe_id)
                    if not new_element:
                        print("New elem is None")
                        continue
                    new_element.func_desc = element.func_desc
                    if new_element.in_list(current_elements, check_coord=True):
                        continue
                    added_elements.append(new_element)
                    current_elements.append(new_element)
                    logger.debug(f"New element: <{new_element.tag}> {new_element.get_name()} {new_element.bbox.get_abs_px_coords()}")
            else:
                removed_elements.append(element)
                logger.warning(f'missing: <{element.tag}> {element.get_name()}, count={elem_loc.count()}')


        if inputs_only:
            new_inputs = []
            for elem in added_elements:
                if elem.tag in ["input", "textarea", "select"]:
                    new_inputs.append(elem)
            added_elements = new_inputs

        elem_changes = {
            "current": current_elements + other_elems,
            "removed": removed_elements, 
            "added": added_elements, 
            "modified": modified_elements
        }

        return elem_changes


    def update_section(
        self,
        section: PageSection = None, 
        elems_subset: list[Element] = [],
        get_all = False,
        inputs_only = False,
        resummarize = False,
        website_mem: WebsiteMem = None,
        same_page = False,
        debug=False,
        summarize = True,
        check_dropdown = False,
        new_only = False
    ) -> tuple[PageSection, dict[str, Any]]:
        """Check for any changes to elements inside section and return diff.
        For update_pagemem, list_action, submit_form, and env_diff."""

        elem_diff = {"current": [], "removed": [], "added": [], "modified": []}
        if section:
            elem_diff['current'] = section.elements

        section_loc = self.get_section_locator(section)
        multiple_locs = False
        if section_loc.count() > 1:
            multiple_locs = True
            logger.warning(f"Multiple section_loc ({section_loc.count()}): {section_loc}")
        elif not section_loc.count():
            logger.warning(f"section missing: {section.class_name}\n{section_loc}")
            return None, elem_diff
        else:
            if section.bbox:
                saved_bbox = section.bbox.copy()
            else:
                saved_bbox = None
            scroll_h = self.page_scroll_height(iframe_id=section.iframe_id)
            section.bbox = BBox.from_playwright_bbox(section_loc.bounding_box(), scroll_h)
            if not section.bbox:
                logger.warning(f"No section bbox")
                section.bbox = BBox()
            if (saved_bbox) and (section.bbox.get_abs_px_height() != saved_bbox.get_abs_px_height()):
                resummarize = True
        logger.info(f'section: {section.type} class="{section.class_name}" {section.bbox.get_abs_px_str()}')
        if (section.updated) and (not check_dropdown):
            logger.debug(f"section already updated")
            return section, elem_diff
        
        if (not section.iframe_id) and (section.type not in ['header', 'nav']):
            if (section.parent_tag=='main') or (self.env.page.locator('main').and_(section_loc).count()):
                section.in_main = True

        first_only = False
        if ((section.data_level) and (section.data_level != '0')) and (section_loc.locator(f'{section.type}').count()):
            first_only = True

        elements = section.elements + elems_subset
        
        if not new_only:
            elem_diff = self.update_elements(
                elements, 
                section, 
                first_only=first_only, 
                section_strict=True, 
                get_all=get_all,
                inputs_only=inputs_only
            )
        current_elements = elem_diff['current']
        removed_elements = elem_diff['removed']
        added_elements = elem_diff['added']
        modified_elements = elem_diff['modified']


        # Check for new action elements
        if (not first_only) and (not multiple_locs):
            all_elems = self.all_section_elements(section_loc, section)
            if not all_elems:
                all_elems = self.all_section_elements(section_loc, section, get_any=True)
            new_action_elems = []
            updated_elems = []
            for n in range(len(all_elems)):
                action_elem = all_elems[n]
                if not action_elem:
                    continue
                action_elem.nth = n

                # if not self.cfg.split_page_sections:
                #     updated_elems.append(action_elem)
                #     continue

                if (action_elem.aria_hidden=="true") and (action_elem.attributes_dict.get('tabindex')=='-1'):  #(action_elem.href=='javascript:void(0)'):
                    continue
                if (action_elem.is_disabled) and ((action_elem.tag in ['input', 'textarea', 'select', 'label']) or (not action_elem.text)):
                    continue
                if (len(all_elems)>250) and (action_elem.tag in ['div', 'label']) and (not action_elem.role):
                    continue

                existing_elem = action_elem.in_list(current_elements)
                if existing_elem:
                    action_elem = self.update_elem(new_elem=action_elem, saved_element=existing_elem)
                else:
                    if website_mem:
                        saved_element = action_elem.in_list(website_mem.all_elements)
                        if saved_element:
                            action_elem = self.update_elem(new_elem=action_elem, saved_element=saved_element)
                    if (not inputs_only) or (action_elem.tag in ['input', 'textarea', 'select', 'label']):
                        logger.debug(f"New action element: {self.get_elem_str(action_elem, short=True)}")
                        new_action_elems.append(action_elem)
                updated_elems.append(action_elem)

            added_elements = added_elements + new_action_elems
            current_elements = updated_elems


        section.elements = current_elements

        actual_removed = []
        for removed_elem in removed_elements:
            if not removed_elem.in_list(added_elements):
                actual_removed.append(removed_elem)
        actual_added = []
        for added_elem in added_elements:
            if not added_elem.in_list(removed_elements):
                actual_added.append(added_elem)

        section.elem_changes = len(actual_removed) + len(actual_added)
        elem_diff = {
            "removed": actual_added, 
            "added": actual_added, 
            "modified": modified_elements
        }
        
        n_dropdowns = 0
        for elem in section.elements:
            if elem.dropdown_elements:
                n_dropdowns += 1
            if not elem.section:
                elem.section = section
        section.n_dropdowns = n_dropdowns
        if section.is_form:
            for elem in section.elements:
                elem.contained_in = 'form'
        if section.is_dialog:
            for elem in section.elements:
                if elem.contained_in != 'form':
                    elem.contained_in = 'dialog'

        if (not section.nth) and (summarize):
            if (resummarize) or (not self.intent) or (not section.summary) or (section.elem_changes>2):
                if self.manual:
                    section.summary = f"{[elem.get_name() for elem in section.elements]}"
                else:
                    section.summary = self.summarize_section(section)

        return section, elem_diff


    def update_pagemem(self, website_mem: WebsiteMem, page_mem: PageMem, saved_page: PageMem, new=False) -> PageMem:
        """Checks if saved locators are working in the active browser env and returns updated page_mem.
        For invalid element locators, will check for elements with same tag and class."""

        logger.success(f"Update page_mem {page_mem.url}")
        page_mem.html = self.env.page.content()
        start_scroll_h = self.page_scroll_height()

        page_height = self.env.page.evaluate("document.body.scrollHeight")
        for i in range(min(10, page_height//1280)):
            self.env.page.mouse.wheel(0, 1280)
            time.sleep(0.2)
        self.env.page.evaluate(f"window.scrollTo(0, {start_scroll_h})")

        same_page = False
        if page_mem == saved_page:
            page_mem = copy.copy(saved_page)
            same_page = True
        
        list_section = saved_page.list_section
        if list_section != None:
            list_text = self.list_info(list_section, n_items=5, caption_img=False)
            if (list_text != list_section.section_text) and (self.intent) and (self.time_step>0):
                list_section.sort_order = self.get_list_sort(list_section)
            list_loc = self.get_section_locator(list_section)
            if not list_loc.count():
                logger.warning(f"section missing: {list_section.class_name} id={list_section.id}")
                page_mem.n_missing_sections += 1
            else:
                page_mem.list_section = copy.copy(list_section)
                if list_section.type == 'list':
                    page_mem.list_section.bbox.y1_abs_px = list_loc.first.bounding_box()['y']
                    page_mem.list_section.bbox.y2_abs_px = list_loc.last.bounding_box()['y'] + list_loc.last.bounding_box()['height']
                else:
                    page_mem.list_section.bbox = BBox.from_playwright_bbox(list_loc.bounding_box(), start_scroll_h)
                logger.info(f'list_section: {list_section.type} class="{list_section.class_name}"')


        matched_sections = [False] * len(saved_page.html_sections)
        updated_sections = []
        for page_section in page_mem.html_sections:
            section_loc = self.get_section_locator(page_section)
            if (not section_loc.count()):
                logger.warning(f"section missing: {page_section.class_name}, {section_loc.count()}")
                page_mem.missing_sections.append(page_section)
                page_mem.n_missing_sections += 1
                continue

            # Check if section matches existing one
            if (not new) or (not self.intent):
                for i in range(len(saved_page.html_sections)):
                    if matched_sections[i] == True:
                        continue
                    saved_section = saved_page.html_sections[i]
                    if page_section.equals(saved_section):
                        matched_sections[i] = True
                        page_section = saved_section.copy_section()
                        break
            
            if (page_section.type=='table') or (page_section.list_type=='table'):
                scroll_h = self.page_scroll_height(iframe_id=page_section.iframe_id)
                page_section.bbox = BBox.from_playwright_bbox(section_loc.bounding_box(), scroll_h)
                if (not self.manual):
                    page_section.summary = self.summarize_section(page_section)
            elif new:
                page_section, elem_diff = self.update_section(page_section, website_mem=website_mem)
            else:
                page_section, elem_diff = self.update_section(page_section)
            
            updated_sections.append(page_section)
        page_mem.html_sections = updated_sections
        
        
        if page_mem.html_sections:
            page_mem.page_height = page_mem.html_sections[-1].bbox.y2_abs_px
        page_mem.task_info = saved_page.task_info
        page_mem.task_sections = saved_page.task_sections
        page_mem.bookmark_info = saved_page.bookmark_info
        if (not page_mem.short_summary):
            self.summarize_page(website_mem, page_mem, all_sections=False)
        
        self.env.page.evaluate(f"window.scrollTo(0, {start_scroll_h})")
        logger.success(f"Finished updating page_mem\n\n")

        return page_mem


    def summarize_list_page(self, website_mem: WebsiteMem, list_page: PageMem):
        """"""

        start_url = self.env.page.url
        if (self.intent) or (not self.save_mem):
            logger.debug(f"intent: {self.intent}, save_mem: {self.save_mem}")
            return

        list_section = list_page.list_section
        if list_section and list_section.elements:
            logger.debug(f"Change page url")
            self.go_to_url_action([], list_page.url)
            time.sleep(1.0)
            self.env.page.wait_for_load_state("domcontentloaded", timeout=20000)
            section_loc = self.get_section_locator(list_section)
            if list_section.type == 'list':
                all_item_locs = section_loc.all()
            else:
                all_item_locs = section_loc.locator('xpath=child::li').all()
            if len(all_item_locs) < 3:
                self.go_to_url_action([], start_url)
                return
            
            section_desc = f'LIST CONTENT: {self.summarize_section(list_section)}'
            list_elements = f'# LIST ITEMS #:\n'
            for i in range(3):
                item_loc = all_item_locs[i]
                item_section = self.create_section(item_loc, list_section.iframe_id)
                if not item_section:
                    continue
                if list_section.type != 'list':
                    item_section.list_section_tag = list_section.type
                    item_section.list_section_class = list_section.class_name
                else:
                    item_section.class_name = list_section.class_name
                item_section.nth = i
                item_section.elements = copy.copy(list_section.elements)
                item_section, elem_diff = self.update_section(item_section)
                if not item_section:
                    continue
                elems_str = self.elems_context(item_section.elements)
                list_elements += f'\nItem {i+1}:\n{elems_str}'

            system_prompt = self.prompts.summarize_list_actions
            user_prompt = f'{section_desc}\n\n{list_elements}'
            messages = self.format_llm_prompt(system_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages)
            actions_summary = self.extract_llm_answer(llm_out, keyword='**ITEM ACTIONS**:')
            list_section.list_actions = actions_summary

            page_mem = website_mem.pages[list_page.url]
            page_mem.list_section = list_section

            self.go_to_url_action([], start_url)
        else:
            self.summarize_page(website_mem, list_page)

        return

    
    def summarize_all_pages(self, website_mem: WebsiteMem, resummarize=False):
        """Generate summaries for all pages in website_mem."""

        logger.success(f"Summarize all pages")

        # Create general summaries for page types
        for list_page in website_mem.list_pages:
            self.summarize_list_page(website_mem, list_page)
        

        for url in website_mem.pages:
            page_mem = website_mem.pages[url]
            if (not resummarize) and (page_mem.short_summary != ""):
                continue
            
            logger.success(f"Summarize page: {page_mem.name} {page_mem.url}")
            self.go_to_url_action([], url)
            time.sleep(3.0)
            self.env._wait_dom_loaded()
            
            if (page_mem.list_page) and (page_mem.list_page != page_mem.url):
                list_page = website_mem.pages[page_mem.list_page]
                page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=list_page)
            else:
                page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=page_mem)

            self.memory.save_website_mem(website_mem, replace=True)

        return


    def pages_match(self, page_mem: PageMem, skip_small=True, debug=False) -> bool:
        """Return True if current page and page_mem have same structure."""

        if not page_mem:
            return False
        if page_mem.list_section:
            if len(page_mem.html_sections) <= 3:
                return False
        else:
            if len(page_mem.html_sections) <= 4:
                return False
        if debug:
            logger.debug(f"{page_mem.name}")

        try:
            all_sections = True
            for section in page_mem.html_sections:
                if section.toggle:
                    continue
                if skip_small and (section.bbox) and (section.bbox.y1_abs_px > 1280):
                    continue
                if section.type == 'main':
                    all_sections = False
                    continue
                section_loc = self.get_section_locator(section)
                if not section_loc.count():
                    all_sections = False
                    if debug: print(f"missing: {section_loc}\n")
                    continue
                if section_loc.count() == 1:
                    if not section_loc.bounding_box():
                        logger.warning(f"No bounding box for section_loc: {section_loc}")
                        all_sections = False
                    elif abs(section.bbox.get_abs_px_height() - section_loc.bounding_box()['height']) > 500:
                        if abs(section.bbox.x1_abs_px - section_loc.bounding_box()['x']) > 10:
                            all_sections = False
                            if debug: print(f"coord diff: {section_loc}\n{section.bbox.get_abs_px_coords()}\n{section_loc.bounding_box()}")

            if (page_mem.list_section):
                section_loc = self.get_section_locator(page_mem.list_section)
                if isinstance(section_loc, list):
                    section_loc = section_loc[0]
                if not section_loc.count():
                    if debug: print(f"missing: {section_loc}\n")
                    all_sections = False

            if all_sections:
                return True
            
        except Error as e:
            logger.warning(f"{str(e)}")

        return False


    def matching_list_page(self, website_mem: WebsiteMem, page_mem: PageMem, debug=False) -> PageMem:
        """Return list_page that matches structure of page_mem if it exists."""

        page_type = None

        has_table = False
        for section in page_mem.html_sections:
            if section.type == 'table':
                has_table = True

        match_pages = []
        for list_page in website_mem.list_pages:
            if has_table:
                if not any([s.type=='table' for s in list_page.html_sections]):
                    continue
            if (page_mem.list_section):
                if (list_page.list_section == None) or (list_page.list_section.type != page_mem.list_section.type):
                    continue
            if (self.pages_match(list_page, debug=debug)):
                match_pages.append(list_page)
            elif (page_mem.list_section) and (page_mem.list_section.equals(list_page.list_section)):
                page_mem.list_section = list_page.list_section.copy_section()
        
        max_sections = 0
        for match in match_pages:
            n_sections = len(match.html_sections)
            if (n_sections != len(page_mem.html_sections)) and (n_sections<10):
                continue
            if match.list_section:
                n_sections += 1
            if abs(n_sections - len(page_mem.html_sections)) >= 2:
                if debug:
                    logger.debug(f"{match.name}: section diff {abs(n_sections - len(page_mem.html_sections))}")
                continue
            if len(match.html_sections) < len(page_mem.html_sections)-1:
                continue
            logger.debug(f"Match: {match.name} {n_sections}")
            if (n_sections >= max_sections) and (match.url in self.env.page.url):
                return match
            if n_sections > max_sections:
                max_sections = n_sections
                page_type = match

        return page_type


    def agent_save_mem(self, website_mem: WebsiteMem):
        """"""

        if self.confirm_mem_save:
            inp = input("Enter 1 to save remaining pages without confirmation (0 for no save): ")
            if inp == "1":
                self.confirm_mem_save = False
            if inp != "0":
                self.memory.save_website_mem(website_mem, replace=True)
        elif not self.intent:
            self.memory.save_website_mem(website_mem, replace=True)

        return


    def get_mem(self, update=True, summarize=False, is_list_item=False, add_bookmark=True) -> tuple[WebsiteMem, PageMem]:
        """Load saved memory of website"""

        self.env.page.mouse.click(1, 1, button='right')
        time.sleep(1.0)
        self.env.page.wait_for_load_state('domcontentloaded')
        start_url = self.env.page.url
        if start_url.endswith('#'):
            start_url = start_url[:-1]
        self.save_history = False
        website_url = get_website_url(start_url)
        
        if not self.page_obs:
            self.check_modal(close_popup=True)

        # Check if website memory saved
        website_mem = None
        mem_exists = False
        for saved_url in self.memory.websites.keys():
            if (website_url == saved_url) or (saved_url.startswith(website_url)) or ('service-now.com' in start_url and saved_url.endswith('service-now.com')):
                mem_exists = True
                website_mem = self.memory.websites[saved_url]
                self.website_mem = website_mem
                logger.success(f"{website_url} already exists: loading saved memory")
                break

        if not mem_exists:
            logger.success(f"Create new website mem for {website_url}")
            website_mem = WebsiteMem()
            website_mem.url = website_url
            self.website_mem = website_mem
            self.memory.save_website_mem(website_mem, replace=True)
            logger.debug(f"Change page url")
            self.go_to_url_action([], website_url)
            page_mem = self.create_page_mem(website_mem)
            
            if self.intent:
                page_mem = self.divide_page(website_mem, page_mem, True)
                page_mem = self.update_pagemem(website_mem, page_mem, page_mem, new=True)
            else:
                # Explore website
                self.exploration_start_time = time.monotonic()
                self.explore_page_v3(website_mem, page_mem, max_recurse=self.cfg.exploration_depth)
                
                # Summarize each page 
                self.summarize_all_pages(website_mem)
                # Save memory to file
                self.memory.save_website_mem(website_mem, replace=True)
        
        else:
            # Get page_mem
            page_mem = None

            if not self.cfg.split_page_sections:
                page_mem = self.create_page_mem(website_mem)
                page_mem = self.divide_page(website_mem, page_mem, True)
                page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=page_mem, new=True)

            elif start_url in website_mem.pages:
                logger.info(f'Load page_mem: {start_url}')
                page_mem = website_mem.pages[start_url]
                if update:
                    page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=page_mem)
            
            if not page_mem:
                logger.success(f"Create new PageMem: {start_url}")
                page_mem = self.create_page_mem(website_mem)
                page_mem = self.divide_page(website_mem, page_mem, True)
                if not page_mem.html_sections:
                    logger.warning(f"No page sections")
                    return website_mem, page_mem
                if (self.intent) and (self.save_mem) and (self.confirm_mem_save):
                    add_bookmark = False
                page_mem.add_bookmark = add_bookmark
                
                page_type = self.matching_list_page(website_mem, page_mem)
                if page_type:
                    logger.success(f"Match page structure: {page_type.name}\n\n")
                    page_type.similar_urls.append(page_mem.url)
                    page_mem.list_page = page_type.url
                    page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=page_type, new=True)
                    if summarize:
                        self.summarize_page(website_mem, page_mem)
                elif not page_type:
                    if is_list_item:
                        page_mem.is_list_item = is_list_item
                    page_mem = self.update_pagemem(website_mem, page_mem=page_mem, saved_page=page_mem, new=True)

        self.save_history = True

        return website_mem, page_mem
    





    #### ---- Element Action functions ---- ####

    def click_coords(
        self, 
        point: tuple[float, float], 
        section_bbox: BBox, 
        elem_desc=None, 
        action_prompt=False,
        element=None, 
        mouse_button='left', 
        iframe_id: str=None,
    ) -> bool:
        """Scale coordinates to page size, then click at point. Return False if coords invalid."""

        action: AgentAction = create_action('click_coords', element, elem_desc)
        
        x, y = point  # (0 to 1.0 scale)
        if (x < 0) or (y < 0):
            logger.warning(f"Invalid click coords: {point}")
            return False
        width = section_bbox.get_abs_px_width()
        height = section_bbox.get_abs_px_height()
        
        x_px = section_bbox.x1_abs_px + (x * width)
        y_px = section_bbox.y1_abs_px + (y * height)
        viewport_size = self.env.page.viewport_size
        if (x_px > viewport_size['width']) or (y_px > viewport_size['height']):
            logger.warning(f"Invalid click coords: {x_px, y_px}")
            return False
        
        logger.success(f"{mouse_button} click coordinates: x={x_px}, y={y_px}")
        self.prev_screenshot = self.env.get_screenshot(max_height=960, iframe_id=iframe_id)
        if iframe_id:
            frame_loc = self.get_frame_locator(iframe_id).locator('body')
            frame_loc.click(position={"x": x_px, "y": y_px}, button=mouse_button, delay=10)
        else:
            self.env.page.mouse.click(x=x_px, y=y_px, button=mouse_button, delay=10)
        time.sleep(2.0)
        self.env.page.wait_for_load_state('domcontentloaded')

        # Save action to history
        if action_prompt:
            if elem_desc.startswith('Click '):
                elem_desc = elem_desc.replace('Click ', 'Clicked ')
            action_summary = elem_desc
        else:
            action_summary = f'Clicked {elem_desc}'
        action['action_summary'] = action_summary
        action['meta_data']['click_coords'] = (x_px, y_px)
        self.record_action(action)
        self.post_screenshot = self.env.get_screenshot(max_height=960, iframe_id=iframe_id)

        return True


    def vlm_click(
        self, 
        screenshot: Image, 
        crop_bbox: BBox,
        elem_desc: str, 
        action = False, 
        button='left',
        element=None,
        iframe_id: str=None,
    ) -> tuple[float, float]:
        """Prompt VLM for click coordinates of element."""

        vlm_prompt = self.vlm_click_prompt(elem_desc, action=action)
        vlm_out = self.model_manager.vlm_call(screenshot, vlm_prompt)
        points = self.parse_vlm_coords(vlm_out, screenshot.size)
        if not points:
            logger.warning("Invalid coord output")
            return None
        xy = points[0]

        success = self.click_coords(
            xy, 
            crop_bbox, 
            elem_desc, 
            action_prompt=action,
            element=element, 
            iframe_id=iframe_id
        )
        if not success:
            return None

        x, y = points[0]
        image = draw_point_on_image(screenshot, x, y)
        image.save(f"{self.screenshot_dir}/vlm_click.png")

        return points[0]
    

    def check_search_sugg(self, element: Element) -> bool:
        """"""

        before_img, _ = self.element_crop(element, crop_under=True)
        
        element_locator = self.get_elem_locator(element)
        
        try:
            element_locator.click()
        except Error as e:
            logger.warning(f"Click failed: {str(e)}")
            return False

        time.sleep(1.0)
        after_img, _ = self.element_crop(element, crop_under=True)

        if (not before_img) or (not after_img):
            return False

        before_img.save(f"{self.screenshot_dir}/img_1.png")
        after_img.save(f"{self.screenshot_dir}/img_2.png")

        # Check for visual change
        if not images_identical(before_img, after_img):
            return True

        return False


    def check_load_screen(self) -> bool:
        """"""

        if self.manual:
            return False
        if self.cfg.llm_only:
            time.sleep(2.0)
            return False

        loading = True
        n = 0
        while (loading) and (n < 3):
            n += 1
            screenshot = self.env.get_screenshot(max_height=960)
            if not screenshot: continue
            vlm_prompt = f'Is there an active loading icon displayed on the screen? Answer yes or no.'
            vlm_out = self.model_manager.vlm_call(screenshot, vlm_prompt)

            if not vlm_out: continue
            if vlm_out.lower().startswith('yes'):
                time.sleep(2.0)
            else:
                loading = False

        return loading


    def element_crop(self, element: Element, crop_under=False, margin=False, wide=False, save=True, highlight=False) -> tuple[Any, BBox]:
        """Returns crop of the page screenshot containing the element and BBox of crop region."""

        # Get current screenshot
        scroll_height = self.page_scroll_height(element.iframe_id)
        full_screenshot_arr = np.array(Image.open(BytesIO(self.env.page.screenshot(full_page=False, timeout=30000))))
        full_screenshot = Image.fromarray(full_screenshot_arr)
        img_width, img_height = full_screenshot.size

        crop_bbox = BBox()
        bbox = element.bbox.copy()
        if bbox.x1_abs_px < 0:
            bbox.x1_abs_px = bbox.x2_abs_px - 340
        bbox.shift_bbox_down(-scroll_height)
        print(bbox.get_abs_px_coords())

        if highlight:
            # cropped_screen = draw_bbox_on_image(cropped_screen, element.bbox, abs=True)
            full_screenshot = draw_bbox_on_image(full_screenshot, bbox, abs=True, add_margin=True)
            full_screenshot.save(f"{self.screenshot_dir}/full_draw.png")

        if wide:
            # 896x504
            crop_bbox.x1_abs_px = max(bbox.x1_abs_px - 400, 0)
            crop_bbox.x2_abs_px = min(bbox.x1_abs_px + 496, img_width)
            crop_bbox.y1_abs_px = max(bbox.y1_abs_px - 150, 0)
            crop_bbox.y2_abs_px = min(bbox.y1_abs_px + 354, img_height)
        else:
            # 500x500
            crop_bbox.x1_abs_px = max(bbox.x1_abs_px - 100, 0)
            crop_bbox.x2_abs_px = min(bbox.x1_abs_px + 400, img_width)
            crop_bbox.y1_abs_px = max(bbox.y1_abs_px - 10, 0)
            crop_bbox.y2_abs_px = min(bbox.y1_abs_px + 490, img_height)
            if crop_bbox.get_abs_px_width() < 400:
                crop_bbox.x1_abs_px -= 200

        if crop_under:
            crop_bbox.y1_abs_px = bbox.y2_abs_px  #+ 25

        # Crop around element
        cropped_screen = crop_img(full_screenshot, crop_bbox, abs=True, add_margin=margin)

        if (save == True) and (cropped_screen):
            cropped_screen.save(f"{self.screenshot_dir}/element_crop.png")

        return cropped_screen, crop_bbox


    def section_screenshot(self, section: PageSection, max_height=None, add_margin=False, expand=False) -> tuple[Any, BBox]:
        """Get screenshot cropped to section."""

        viewport_size = self.env.viewport_size

        if (max_height) and (max_height > viewport_size['height']):
            screenshot = self.env.get_screenshot(max_height=max_height, iframe_id=section.iframe_id)
            section_crop = crop_img(screenshot, section.bbox, abs=True, add_margin=add_margin)
            section_crop.save(f'{self.screenshot_dir}/full_list_section.png')
            screenshot.save(f'{self.screenshot_dir}/full_page.png')
            return section_crop, section.bbox
        section_loc = self.get_section_locator(section)
        if not section_loc.count():
            logger.error(f"Can't locate section, {section_loc.count()}")
            return None, None
        if section_loc.count() == 1:
            if (section.screenshot) and (section_loc.inner_html()==section.inner_html):
                return section.screenshot, section.bbox

        crop_bbox = section.bbox.copy()
        if (expand) and (crop_bbox.get_abs_px_height() < 200):
            crop_bbox.y2_abs_px = crop_bbox.y1_abs_px + 500

        if max_height:
            crop_bbox.y2_abs_px = crop_bbox.y1_abs_px + max_height
        
        scroll_height = self.page_scroll_height(section.iframe_id)
        screen_y2 = scroll_height + viewport_size['height']
        if (crop_bbox.y2_abs_px > screen_y2) or (crop_bbox.y1_abs_px < scroll_height+100):
            logger.debug(f"crop_bbox={crop_bbox.get_abs_px_coords()}, viewport_y=({scroll_height}, {screen_y2})")
            crop_bbox = self.scroll_to_bbox(crop_bbox, iframe_id=section.iframe_id)
            if (section.type != 'list' and section.type != 'table') and (section_loc.count() == 1):
                crop_bbox = BBox.from_playwright_bbox(section_loc.first.bounding_box())

        if not crop_bbox:
            logger.warning(f"crop_bbox: {crop_bbox}")
            return None, None
        if crop_bbox.y1_abs_px > viewport_size['height']:
            crop_bbox.shift_bbox_down(-scroll_height)
            logger.debug(f"scroll height {scroll_height}")

        crop_bbox.x1_abs_px = max(crop_bbox.x1_abs_px, 0)
        crop_bbox.x2_abs_px = min(crop_bbox.x2_abs_px, viewport_size['width'])
        crop_bbox.y1_abs_px = max(crop_bbox.y1_abs_px, 0)
        crop_bbox.y2_abs_px = min(crop_bbox.y2_abs_px, viewport_size['height'])
        logger.debug(f"crop_bbox: {crop_bbox.get_abs_px_coords()}")
        
        screenshot = self.env.get_screenshot(max_height=max_height, iframe_id=section.iframe_id)
        if not screenshot:
            return None, None
        section_crop = crop_img(screenshot, crop_bbox, abs=True, add_margin=add_margin)
        if not section_crop:
            logger.error(f"failed to crop image")
        section.screenshot = section_crop
        if section_loc.count() == 1:
            try:
                section.inner_html = section_loc.inner_html()
            except Error as e:
                pass

        if (section.list_type) and (section_crop):
            section_crop.save(f'{self.screenshot_dir}/list_section.png')

        return section_crop, crop_bbox


    def describe_screen(
        self, 
        section: PageSection=None, 
        prompt: str=None, 
        section_loc: Locator=None, 
        add_margin=True, 
        max_height=None
    ) -> str:
        """VLM describe details of section"""

        website_name = site_name(self.env.page.url)

        if section:
            section_loc = self.get_section_locator(section)
            section_text = ""
            if section_loc.count() == 1:
                try:
                    section_text = section_loc.inner_text()
                except:
                    logger.warning(f"Can't get section text")
            screenshot, _ = self.section_screenshot(section, max_height=max_height, add_margin=add_margin)
            if not screenshot:
                return ""
        elif section_loc:
            screenshot = self.env.get_screenshot(max_height=max_height)
            bbox = BBox.from_playwright_bbox(section_loc.bounding_box())
            screenshot = crop_img(screenshot, bbox, abs=True)
        else:
            screenshot = self.env.get_screenshot(max_height=max_height)
            nav_h = 0
            if self.page_obs.html_sections:
                nav_h = self.page_obs.html_sections[0].bbox.get_abs_px_height()
            if nav_h < 100:
                screenshot = crop_img(screenshot, top=nav_h)

        # VLM describe
        if prompt:
            vlm_prompt = prompt
        else:
            vlm_prompt = self.prompts.site_page_summary.format(website_name=website_name)
        vlm_out = self.model_manager.vlm_call(screenshot, vlm_prompt)

        return vlm_out


    def vlm_section_action(self, section: PageSection, action: str=None, element=None, max_actions=10):
        """VLM describe section observation, """

        if self.cfg.llm_only:
            return False

        logger.success(f"VLM screenshot action")
        start_url = self.env.page.url
        if max_actions < 1:
            return False
        if not section:
            section_loc = self.env.page.locator('body')
            section = self.create_section(section_loc)

        # Scroll to section bbox and get screenshot
        screenshot, crop_bbox = self.section_screenshot(section, max_height=1280, add_margin=False)
        if not screenshot:
            return False
        vlm_prompt = self.prompts.describe
        obs_desc = self.model_manager.vlm_call(screenshot, vlm_prompt=vlm_prompt)

        action_desc = f"click {self.get_elem_str(element, format='e')}."
        click_point = self.vlm_click(
            screenshot, 
            crop_bbox, 
            action_desc, 
            action=True, 
            element=element, 
            iframe_id=section.iframe_id
        )
        if not click_point:
            success = False
        else:
            success = True
        
        if max_actions <= 1:
            return success

        return success


    def vlm_click_dropdown(self, element: Element, nested=False, clicked=[], max_actions=3) -> str:
        """VLM lists dropdown menu options, LLM chooses option, then VLM clicks option.
        If successful return the selected option, else return empty string."""

        start_url = self.env.page.url

        # Get list of dropdown options from VLM
        elem_crop, crop_bbox = self.element_crop(element, margin=False)
        if not elem_crop:
            logger.error(f'No screenshot provided')
            return ''
        menu_name = self.get_elem_str(element, format='e', short=True)
        if not menu_name:
            menu_name = 'expanded'
        vlm_prompt = self.prompts.list_dropdown_md.format(menu_name=menu_name)
        if nested:
            vlm_prompt = f'List all options and suboptions shown in the search results menu.'
        vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt)
        dropdown_options = self.extract_bullet_list(vlm_out)
        if dropdown_options:
            element.options = dropdown_options
        options_list = [option for option in dropdown_options if option not in clicked]

        # Prompt LLM for action to perform
        sys_prompt = self.prompts.choose_element
        task = f"**TASK**: {self.intent}"
        subgoal = f"**CURRENT SUBGOAL**: {self.current_subgoal}"
        progress = f"{self.progress_str(add_current=True, subgoal_actions=True, title=False)}"
        user_prompt = f"{task}\n\n{subgoal}\n\n{progress}\n\n"
        user_prompt += '**PAGE ELEMENTS**:\n{context}'
        
        selected = self.llm_choose_list(options_list, options_list, sys_prompt, user_prompt, '**SELECT ELEMENT**:', n_ret=1, add_none=True, inline=False)
        selected_option = next(iter(selected), None)
        if not selected_option:
            logger.debug(f"Cancel")
            self.key_press_action([], 'Escape')
            return ''
        
        # VLM clicks option on screen
        if not clicked:
            action_desc = f"the '{selected_option}' option in the dropdown menu"
        else:
            action_desc = f"the '{selected_option}' element"
        vlm_prompt = self.vlm_click_prompt(action_desc, False)
        vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt)
        points = self.parse_vlm_coords(vlm_out, elem_crop.size)
        if not points:
            logger.warning("Invalid coord output")
            return ''
        success = self.click_coords(points[0], crop_bbox, action_desc, element=element)
        if not success:
            return ''
        elem_crop.save(f"{self.screenshot_dir}/dropdown_click_input.png")
        image = draw_point_on_image(elem_crop, points[0][0], points[0][1])
        image.save(f"{self.screenshot_dir}/dropdown_click_output.png")

        if (self.env.page.url==start_url):
            if (self.check_dropdown(element, get_options=False)):
                if max_actions <= 1:
                    self.key_press_action([], 'Escape')
                else:
                    clicked.append(selected_option)
                    return self.vlm_click_dropdown(element, clicked=clicked, max_actions=max_actions-1)

        return selected_option


    def handle_popup(self, element: Element) -> Page:
        """"""

        new_page = self.click_with_page_handling(element, button='Enter')
        if new_page:
            logger.success(f"Handle popup window")

            if not self.cfg.llm_only:
                screenshot_arr = np.array(Image.open(BytesIO(new_page.screenshot(full_page=False, timeout=30000))))
                screenshot = Image.fromarray(screenshot_arr)
                description = self.model_manager.vlm_call(screenshot, 'Describe this screenshot in one sentence.')
                self.page_obs.popup_info = description

            self.env.context.pages.append(new_page)

        return new_page


    def click_element(
        self,
        element: Element=None,
        section: PageSection=None,
        elem_loc: Locator=None,
        mouse_button='left',
        record_action=True
    ) -> bool:
        """Click the element on the page and return success status."""
        
        if element:
            locator = self.get_elem_locator(element, section=section)
        elif elem_loc:
            element = self.elem_from_loc(elem_loc)
            locator = elem_loc
        else:
            logger.warning(f"No element argument provided")
            return False
        logger.debug(f'Click element')
        elem_name = self.get_elem_str(element, format='d', max_options=0)
        
        action: AgentAction = create_action('click_element')
        action['meta_data']['element'] = element
        action['meta_data']['elem_crop'] = self.element_crop(element, wide=True)
        self.prev_screenshot = self.env.get_screenshot(max_height=960, iframe_id=element.iframe_id)

        section_locator = None
        if section:
            section_locator = self.get_section_locator(section)
            if section_locator.count() != 1:
                section_locator = None

        if (locator.count() != 1):
            logger.warning(f"Can't locate element {elem_name}\n{locator}, {locator.count()}")
            if not section_locator:
                return False
            result = self.vlm_section_action(section, element=element, max_actions=1)
            if result:
                return True
            else:
                return False

        # Playwright locator click
        if (element.input_type=='radio'):
            try:
                locator.set_checked(True, force=True)
            except:
                if locator.locator("xpath=following-sibling::label").count():
                    locator= locator.locator("xpath=following-sibling::label").first
                    bbox = locator.bounding_box()
                    x = bbox['width']-5
                    y = bbox['height']-5
                    locator.click(force=True, position={'x': x, 'y': y})
        elif mouse_button == 'Enter':
            if (element.new_page) or (not element.explored):
                self.handle_popup(element)
            else:
                locator.focus()
                time.sleep(0.5)
                self.key_press_action([], 'Enter')
        elif (mouse_button == 'left') or (mouse_button == 'right'):
            for _ in range(3):
                try:
                    locator.click(button=mouse_button, force=True, delay=10)
                    break
                except Error as e:
                    logger.error(f"Error: {e.message}")
                    time.sleep(0.5)
                    logger.info(f"Retry click")
        else:
            logger.warning(f"{mouse_button} is not a valid button")
            return False
        
        if (element.dropdown_elements or element.role=='tab') and (not element.destination_page):
            time.sleep(1.0)
        else:
            time.sleep(3.0)
        self.env.page.wait_for_load_state('domcontentloaded')

        # Save action
        action_summary = f"Clicked {elem_name}"
        if element.input_type == 'checkbox':
            if element.input_value == '':
                element.input_value = 'checked'
                action_summary = f"Checked {elem_name}"
            elif element.input_value == 'checked':
                element.input_value = ''
                action_summary = f"Unchecked {elem_name}"

        action['action_summary'] = action_summary
        action['meta_data']['click_button'] = mouse_button
        self.post_screenshot = self.env.get_screenshot(max_height=960, iframe_id=element.iframe_id)
        is_loading = self.check_load_screen()
        if record_action:
            self.record_action(action)

        return True


    def click_with_page_handling(self, element: Element=None, elem_loc=None, timeout=5000, button='left'):
        """Click element and check for new browser page."""
        
        try:
            with self.env.context.expect_page(timeout=timeout) as page_info:

                if elem_loc:
                    locator = elem_loc
                else:
                    locator = self.get_elem_locator(element)
                
                locator.focus()
                self.key_press_action([], "Enter")
                locator.hover()
                time.sleep(3.0)
                self.env._wait_dom_loaded(networkidle=True)
            
            if page_info:
                page = page_info.value
                page.wait_for_load_state("domcontentloaded", timeout=5000)
                for frame in page.frames:
                    try:
                        frame.wait_for_load_state("networkidle", timeout=5000)
                    except Error:
                        pass
                print(f"New page opened: {page.url}")
                is_loading = self.check_load_screen()
                return page
        except Error as e:
            # No new page is opened
            return None
    

    #### ---- Input Actions ---- ####

    def choose_elem_option(self, element: Element, options: list[str]=[], multiple=False) -> list[str]:
        """"""

        if len(options) == 1:
            logger.info(f"One option: {options}")
            return options[0]
        elif len(options) == 0:
            logger.info(f"No options: {options}")
            return None
        
        if multiple:
            sys_prompt = self.prompts.choose_select_multiple
            keyword = '**SELECT OPTIONS**:'
            choose_n = None
        else:
            sys_prompt = self.prompts.choose_select_option
            keyword='**OPTION**:'
            choose_n = 1

        if self.manual:
            user_prompt = 'INPUT OPTIONS:\n{context}'
        else:
            if self.form_plan:
                task = f"TASK: {self.form_plan}"
                progress = f'{self.progress_str(add_current=True, subgoal_actions=True)}'
            else:
                task = f"TASK: {self.intent}"
                progress = f'HISTORY:\n{self.episode_history()}'
            page = f'PAGE: {self.page_obs.page_summary}'
            input_info = f'INPUT FIELD: {self.get_elem_str(element)}'
            user_prompt = f'{task}\n\n{progress}\n\n{page}\n\n{input_info}\n\n'
            user_prompt += 'INPUT OPTIONS:\n{context}'
        
        show_prompt = True
        
        chosen = self.llm_choose_list(
            options, 
            options, 
            sys_prompt, 
            user_prompt, 
            keyword=keyword, 
            n_ret=choose_n, 
            inline=True,
            show_prompt=show_prompt
        )
        if not chosen:
            return None

        return chosen


    def select_option(self, select_element: Element, section: PageSection=None) -> bool:
        """Prompt LLM to choose a <select> element option and then set the option on the page."""

        action: AgentAction = create_action('choose_select_option', element=select_element)

        logger.success(f"Choose <select> element option")
        select_locator = self.get_elem_locator(select_element, section)
        if select_locator.count() != 1:
            logger.warning(f"Element cannot be located on the current page.")
            return False
        
        label = select_locator.get_attribute('aria-label')
        if label:
            select_element.aria_label = label
        select_name = select_element.get_name()
        multiple = False
        if (select_element.tag=='select') and (select_element.multiple):
            multiple = True
        
        # Update list of select options
        select_element.options = self.all_select_options(select_element)
        options = select_element.options

        if multiple:
            options = self.choose_elem_option(select_element, options, multiple=True)
        else:
            options = self.choose_elem_option(select_element, options)
            if options == None:
                return False
            option = options[0]

        # Set selection on page
        if select_element.tag == "select":
            try:
                selected_option = select_locator.select_option(options, force=True)
            except Error as e:
                logger.warning(f"Error: {e.message}")
                self.invalid_elems.append(select_element)
                return False
        elif select_element.role == "combobox":
            base_locator = self.get_base_locator(select_element)
            is_open = self.get_valid_attr(select_locator, 'aria-expanded')
            if is_open == "false":
                select_locator.click(force=True)
                time.sleep(1.0)
            options_id = select_element.attributes_dict.get('aria-owns')
            if not options_id:
                options_id = select_locator.get_attribute('aria-owns')
            if not options_id:
                options_id = select_locator.get_attribute('aria-controls')
            if not options_id:
                logger.warning(f"No aria-owns or aria-controls")
                return False
            options_loc = base_locator.locator(f'id={options_id}')
            # click chosen option
            opt_loc = options_loc.locator('[role="option"]').get_by_text(option, exact=True)
            if opt_loc.count() == 1:
                opt_loc.click()
            else:
                self.env.page.keyboard.type(option, delay=10)
                self.key_press_action([], 'Enter')
        else:
            logger.warning(f"{select_element.get_name()} is not a <select> element")

        # add action to history
        if not multiple:
            select_element.input_value = option
            action['action_summary'] = f'Selected value of "{option}" for field "{select_name}"'
            action['meta_data']['input_value'] = option
        else:
            select_element.input_value = options
            action['action_summary'] = f'Selected options {options} for the "{select_name}" listbox'
            action['meta_data']['input_value'] = options
        action['meta_data']['elem_crop'] = self.element_crop(select_element, wide=True)
        self.record_action(action)

        return True
    

    def enter_input_value(self, input_element: Element) -> str:
        """"""

        if (input_element.tag == 'textarea') and ('editor' in input_element.get_name().lower()):
            filename, content = self.file_name_content(input_element)
            return content
        if (input_element.tag == 'textarea') and (not self.page_obs.retrieved_history) and (self.time_step > 0):
            self.retrieve_history_details()

        # Format prompt
        if self.form_plan:
            system_prompt = self.prompts.enter_input_field
            task = f"TASK: {self.form_plan}"
            page = f'PAGE: {self.page_obs.page_summary}'
            progress = f'{self.progress_str(add_current=True, subgoal_actions=True)}'
            input_field = f'INPUT FIELD: {self.get_elem_str(input_element)}'
            user_prompt = f'{task}\n\n{page}\n\n{progress}\n\n{input_field}'
        else:
            system_prompt = self.prompts.type_input_value
            task = f"TASK: {self.intent}"
            history = f"HISTORY:\n{self.episode_history()}"
            current_page = f"CURRENT PAGE: {self.get_page_info(self.page_obs, full_details=True)}"
            if (input_element.contained_in == 'form'):
                current_page += f"\n\n{self.progress_str(subgoal=True)}"
            input_field = f'INPUT FIELD: {self.get_elem_str(input_element)}'
            user_prompt = f'{task}\n\n{history}\n\n{current_page}\n\n{input_field}'

        if (input_element.input_type in ["email", "password"]) and (get_website_url(self.page_obs.url) in self.memory.accounts):
            login = self.memory.accounts[get_website_url(self.page_obs.url)]
            email = login['username']
            password = login['password']
            user_prompt += f'\n\n**Note**: The user login is email="{email}", password="{password}"'
        if input_element.required:
            user_prompt += f'\n\n**Important**: This is a required field, provide a value even if the task does not specify one.'
        
        # Prompt LLM
        messages = self.format_llm_prompt(system_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=True)
        line_only = True
        if input_element.tag == 'textarea':
            line_only = False
        input_value = self.extract_llm_answer(
            llm_out, 
            keyword='**INPUT VALUE**:', 
            backups=['**INPUT VALUE:**', '**Input Value**:', '**Input Value:**'], 
            line_only=line_only)

        # convert sim_url back to local
        input_value = revert_sim_url(input_value)

        if (not input_element.input_truncated):
            if (input_element.name=='name') and (' ' in input_value):
                words = input_value.strip().split(' ')
                capitals = [w for w in words if w.capitalize()[0]==w[0]]
                if (words[0] in capitals) and (len(capitals) == 1):
                    input_element.input_truncated = True
                    input_value = words[0]
            elif (input_element.get_name().startswith('Search')) and ('/' in input_value):
                parts = input_value.split('/')
                first, last = parts[0], parts[-1]
                if (len(last) > 4) and (len(first) > 2):
                    input_element.input_truncated = True
                    input_value = last
        
        if "leave blank" in input_value.lower():
            input_value = ""

        return input_value


    def enter_input(
        self, 
        input_element: Element, 
        page_section: PageSection=None, 
        value: str=None, 
        focused=False,
        tab_after=True,
        dropdown=False
    ) -> bool:
        """Prompt LLM for argument value and then fill the input field.
        If successful return the entered value, else return empty string."""

        logger.success(f"Fill input field")
        action: AgentAction = create_action('enter_input', element=input_element)
        input_name = self.get_elem_str(input_element, section=page_section, short=True)

        input_loc = self.get_elem_locator(input_element, page_section)
        if input_loc.count() != 1:
            logger.error(f"Can't locate {input_name}: {input_loc} {input_loc.count()}")
            return False
        
        # if input is read_only, copy value to clipboard
        if 'readonly' in input_element.attributes_dict:
            if ('dropdown' in input_name) or ('dropdown' in input_element.class_name):
                return self.click_element(input_element, page_section)
            elif input_element.input_value:
                self.copy_to_clipboard(input_element)
                return True
            else:
                return False
        if input_loc.get_attribute('type') == 'file':
            logger.warning(f'Input of type "file" cannot be filled')
            return True

        if not focused:
            input_loc.click(position={"x":5, "y":5}, force=True)
        focused_elem = self.get_foc_elem()
        if ('date' in str(input_element.id)):
            logger.debug(f"Check date format")
            self.key_press_action([], 'ArrowRight')
            time.sleep(0.5)
            date_str = self.get_input_value(input_element=input_element, input_loc=input_loc)
            if '-' in str(date_str):
                input_element.date_format = 'YYYY-MM-DD'
            input_loc.clear(force=True)
            self.env.page.keyboard.press('ArrowLeft')
        input_element.aria_invalid = False

        
        if value:
            input_value = value
        elif self.manual:
            input_value = input("Enter input value: ")
        else:
            input_value = self.enter_input_value(input_element)
            if input_value in ['""', '(leave blank)', '(empty)']:
                input_value = ''
            if (not input_value) and (input_element.required):
                input_value = 'placeholder'
            if (not focused) and ((input_value == input_element.input_value) or (input_value=="(leave blank)")):
                action['action_summary'] = f'Entered value "{input_value}" for field "{input_name}"'
                action['meta_data']['input_value'] = 'unchanged'
                self.record_action(action)
                return True

        try:
            if input_loc.get_attribute('inputmode') == 'numeric':
                input_value = re.sub(r'\D', '', input_value)
        except Error as e:
            pass

        # Fill input field
        input_loc = self.get_elem_locator(input_element, page_section)
        if (focused_elem) and ('min' not in input_element.attributes_dict):
            if not focused:
                input_loc.clear(force=True)
            self.env.page.keyboard.type(input_value, delay=10)
            time.sleep(2.0)
            if focused_elem.aria_autocomplete:
                input_element.aria_autocomplete = focused_elem.aria_autocomplete
        else:
            input_loc.fill(value=input_value, force=True)
        

        dialog = False
        if (tab_after) and (not dropdown) and (not input_element.aria_autocomplete):
            try:
                with self.env.page.expect_event('dialog', timeout=3000) as dialog_info:
                    self.env.page.keyboard.press('Tab')
                dialog = True
            except Error as e:
                pass

        new_input_value = input_value
        input_element.input_value = new_input_value
        if page_section:
            page_section.elem_changes += 1
            page_section.input_modified = True
        if self.page_obs.dialog:
            self.page_obs.dialog_field_edited = True

        # add action to history
        action['action_summary'] = f'Entered value of "{input_value}" into {input_name}'
        if (input_element.tag == 'textarea') and (len(input_value.split('\n')) > 2):
            action['action_summary'] = f'Entered the following text into {input_name}: """{input_value}"""'
        action['meta_data']['input_value'] = new_input_value
        action['meta_data']['elem_crop'] = self.element_crop(input_element, wide=True)
        self.record_action(action)


        # Check for suggestions dropdown
        if ((input_element.aria_autocomplete in ['list', 'both']) or (input_name.startswith('select'))) and (self.cfg.compound_actions):
            logger.debug(f"Select suggestion")
            suggestions_id = None
            try:
                if input_loc.get_attribute('aria-controls'):
                    suggestions_id = input_loc.get_attribute('aria-controls')
                elif input_loc.get_attribute('aria-owns'):
                    suggestions_id = input_loc.get_attribute('aria-owns')
                
                if suggestions_id:
                    suggest_loc = self.get_base_locator(input_element).locator(f'id={suggestions_id}').filter(visible=True)
                    all_sugg_loc = suggest_loc.locator("li, [role='option']").all()
                    all_items = []
                    for item_loc in all_sugg_loc:
                        item = self.elem_from_loc(item_loc)
                        if not item: continue
                        all_items.append(item)
                    self.choose_element(all_items, page_section, dropdown=True)
                else:
                    self.choose_search_suggest(input_element)
            except Error as e:
                logger.warning(f"{str(e)}")

        # Check invalid input
        try:
            if (input_loc.count()==1) and (input_loc.get_attribute('aria-invalid')=='true'):
                input_element.aria_invalid = True
        except:
            pass

        if dialog:
            input_element.aria_invalid = True

        return True
    

    def fix_input_error(self, input_element: Element, page_section: PageSection=None):
        """"""

        website_name = site_name(self.page_obs.url)
        field_name = self.get_elem_str(input_element, format='e', section=page_section)
        input_value = input_element.input_value

        current_page = f"CURRENT PAGE: {get_sim_url(self.page_obs.url)}\n- {page_section.summary}"
        task = f"TASK: {self.intent}"
        prev_actions = f"PREVIOUS ACTIONS:\n{self.action_history()}"
        if self.page_obs.browser_dialog:
            prev_actions += f'\n\nERROR MESSAGE: "{self.page_obs.browser_dialog}"'
            self.page_obs.browser_dialog = None
        context = f"{current_page}\n\n{task}\n\n{prev_actions}"

        user_prompt = self.prompts.fix_input_error.format(
            website_name=website_name,
            context=context,
            field_name=field_name,
            input_value=input_value
        )
        messages = self.format_llm_prompt(user_prompt=user_prompt)

        llm_out = self.model_manager.llm_call(messages)
        values_str = self.extract_llm_answer(llm_out, keyword=f'**VALUES**:', line_only=False)
        values_list = self.extract_bullet_list(values_str)

        input_success = False
        i = 0
        while (not input_success) and (i < min(len(values_list), 5)):
            input_success = self.enter_input(input_element, page_section, value=values_list[i])
            i += 1

        return input_success


    def set_checkbox(self, element: Element, page_section: PageSection=None, check=False):
        """"""

        logger.success(f"Set checkbox value")
        action: AgentAction = create_action('enter_input', element)
        elem_name = self.get_elem_str(element, format='e')

        elem_loc = self.get_elem_locator(element, page_section)
        if elem_loc.count() != 1:
            logger.error(f"Can't locate {elem_name}: {elem_loc} {elem_loc.count()}")
            return ""
        try:
            elem_loc.focus()
            elem_loc.click(button='right', force=True)
        except Error as e:
            pass
        time.sleep(1.0)

        options = ['checked', 'unchecked']
        if not check:
            chosen = self.choose_elem_option(element, options)
            if not chosen:
                return ""
            chosen_value = chosen[0]
        else:
            chosen_value = 'checked'

        if chosen_value.lower() == 'checked':
            checked = True
            element.input_value = 'checked'
            action['action_summary'] = f"Checked {elem_name}"
        elif chosen_value.lower() == 'unchecked':
            checked = False
            element.input_value = ''
            action['action_summary'] = f"Un-checked {elem_name}"
        else:
            return ""
        
        try:
            elem_loc.set_checked(checked, force=True)
        except Error as e:
            logger.debug(f"click label")
            if elem_loc.locator("xpath=following-sibling::label").count():
                elem_loc = elem_loc.locator("xpath=following-sibling::label").first
                elem_loc.click(force=True)
            else:
                logger.warning(f"no label")
                elem_loc.click(force=True)
        
        time.sleep(1.0)
        self.env.page.wait_for_load_state('domcontentloaded')
        
        if not check:
            self.record_action(action)
        element.clicked = True

        return chosen_value


    def file_name_content(self, element: Element=None):
        """"""

        logger.debug(f"Get file value")

        # Get relevant info from prev steps
        relevant_details = self.retrieve_history_details()

        # Format prompt
        sys_prompt = self.prompts.create_file
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.get_page_info(self.page_obs)}"
        if relevant_details:
            current_page += f"\n  - Relevant info:\n{self.indent_str(relevant_details, 2, indent_first=True)}"
        input_field = f'FIELD: {self.get_elem_str(element)}'
        user_prompt = f'{task}\n\n{history}\n\n{current_page}\n\n{input_field}'

        if self.manual:
            print(user_prompt)
            filename = input("filename: ")
            content = input("content: ")
        else:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            filename = self.extract_llm_answer(llm_out, keyword='**FILE**:')
            content = self.extract_llm_answer(llm_out, keyword='**CONTENT**:', line_only=False)
            if content.startswith("```"):
                content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            content = revert_sim_url(content)

        return filename, content


    def create_file(self, element: Element=None) -> str:
        """Prompt LLM for the file name and content, then save in saved_files folder"""

        logger.success(f"Creating new file in {self.data_dir}/files/saved_files/")

        # Prompt LLM
        filename, content = self.file_name_content(element)
        if not filename:
            logger.error(f"No filename provided")
            return ''
        
        # Create file
        file_path = f"{self.data_dir}/files/saved_files/{filename}"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.success(f"Successfully saved content to: {file_path}")
            return filename
        except IOError as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            return ''
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return ''


    def upload_file(self, element: Element, page_section: PageSection=None) -> str:
        """Agent chooses a file to upload to the input element."""

        action: AgentAction = create_action('upload_file', element=element)
        logger.success(f"Upload file action")

        locator = self.get_elem_locator(element, page_section)
        if element.tag != 'input':
            locator = locator.locator('xpath=following-sibling::input')
        if locator.count() != 1:
            logger.warning(f"Can't locate file upload element: {element.get_name()}")
            return ''
        elem_name = element.get_name().lower()

        input_files = self.saved_info["files"]["input_files"]
        saved_files = self.saved_info["files"]["saved_files"]
        default_files = self.saved_info["files"]["default_files"]
        all_files = input_files + saved_files
        all_files.append('Create new file')

        if (not (input_files + saved_files)) and ('image' in elem_name):
            file = default_files[0]
        else:
            sys_prompt = self.prompts.choose_file
            task = f"TASK: {self.intent}"
            history = f"HISTORY:\n{self.episode_history()}"
            current_page = f"CURRENT PAGE: {self.get_page_info(self.page_obs, full_details=True)}"
            input_field = f'INPUT FIELD: {self.get_elem_str(element)}'
            user_prompt = f'{task}\n\n{history}\n\n{current_page}\n\n{input_field}\n\n'
            if self.input_image_desc:
                input_file_info = f'File info:'
                for file, desc in self.input_image_desc.items():
                    input_file_info += f"\n- {file}: {desc}"
                user_prompt += f"{input_file_info}\n\n"
            user_prompt += 'FILES:\n{context}'

            chosen = self.llm_choose_list(all_files, all_files, sys_prompt, user_prompt, '**ANSWER**:', n_ret=1, add_none=True)
            logger.success(f"Chosen file: {chosen}")
            if not chosen:
                return 'None'
            else:
                file = chosen[0]
        
        if file == 'Create new file':
            file = self.create_file(element)
            if not file:
                return ''
            saved_files.append(file)
            self.saved_info["files"]["saved_files"].append(file)

        if file in input_files:
            folder = "input_files"
        elif file in saved_files:
            folder = "saved_files"
        elif file in default_files:
            folder = "default_files"
        else:
            logger.warning(f"Invalid file name: {file}")
            return file

        file_path = f"{self.data_dir}/files/{folder}/{file}"
        locator.set_input_files(f"{file_path}")
        time.sleep(1.0)
        self.env.page.wait_for_load_state('domcontentloaded')

        uploaded_file = file
        element.input_value = file

        # Save action
        action['action_summary'] = f'Uploaded file: "{file}" into the "{element.get_name()}" field'
        action['meta_data']['input_value'] = uploaded_file
        action['meta_data']['elem_crop'] = self.element_crop(element, wide=True)
        self.record_action(action)

        return file


    def save_img(
        self, 
        img_element: Element=None, 
        img_loc: Locator=None, 
        page_section: PageSection=None,
        image: Image=None,
        image_path: str=None,
        image_desc: str=None,
        is_input_file = False
    ) -> bool:
        """Download the image to agent's files."""

        if (img_element==None) and (img_loc==None) and (image==None):
            logger.error(f"No image input provided.")
            return False
        if img_loc:
            locator = img_loc
        elif img_element:
            locator = self.get_elem_locator(img_element, page_section)
            if locator.count() != 1:
                return False
        
        save_dir = f"{self.data_dir}/files/saved_files"
        if is_input_file:
            save_dir = f"{self.data_dir}/files/input_files"

        if not os.path.exists(save_dir):
            logger.error(f"Directory {save_dir} doesn't exist")
            return False
        else:
            try:
                image_file = image_path.split('/')[-1]
                ext = image_file.split('.')[-1]
                if ext not in ['png', 'jpg', 'jpeg', 'webp']:
                    image_file = image_file.replace(ext, 'png')
                max_area = 1000000
                if (image.size[0]*image.size[1] > max_area):
                    image = image.resize((928, 928), resample=Image.LANCZOS)
                image.save(f"{save_dir}/{image_file}")
                if is_input_file:
                    self.saved_info['files']['input_files'].append(image_file)
                else:
                    self.saved_info['files']['saved_files'].append(image_file)
                if not image_desc:
                    vlm_prompt = self.prompts.desc_img_short
                    image_desc = self.model_manager.vlm_call(image, vlm_prompt, image_path=image_path)
                self.input_image_desc[image_file] = image_desc
                return True
            except Error as e:
                logger.warning(f"{repr(e)}")

        return False


    def get_copied_text(self) -> str:
        """Return copied text and clear clipboard after."""

        try:
            copied_text = self.env.page.evaluate("navigator.clipboard.readText()")
            self.env.page.evaluate("navigator.clipboard.writeText('')")
            return copied_text
        except Error as e:
            logger.warning(f"Error while trying to read clipboard:\n\n{str(e)}")
            return ""
    

    def copy_to_clipboard(self, input_elem: Element=None) -> str:
        """"""

        logger.success(f"Copy text to clipboard")
        action: AgentAction = create_action('copy')

        if input_elem:
            action['meta_data']['element'] = input_elem
            value = input_elem.input_value
            if not value:
                return ""
            self.clipboard = value
            action['action_summary'] = f'Copied value "{value}" from the "{input_elem.get_name()}" field'
        else:
            value = self.get_copied_text()
            if not value:
                return ""
            self.clipboard = value
            action['action_summary'] = f'Copied text to clipboard: "{value}"'

        self.record_action(action)

        return value


    def choose_search_suggest(self, search_element, prev_options=[]) -> str:
        """Optionally choose search suggestion(s) and return the selected query."""

        if self.cfg.llm_only:
            return ''

        start_url = self.env.page.url
        logger.success(f"Choose search suggestion option")

        # Get list of suggestions from VLM
        elem_crop, crop_bbox = self.element_crop(search_element, crop_under=True, margin=False)
        vlm_prompt = self.prompts.list_dropdown_md.format(menu_name='suggestions')
        vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt)
        options_list = self.extract_bullet_list(vlm_out)
        if (not options_list) or (options_list in prev_options):
            logger.debug(f"Cancel")
            return ''
        prev_options.append(options_list)
        
        # Prompt LLM to choose suggestion
        if self.manual:
            options_str = self.format_list_to_str(options_list, numbered=False)
            selected_option = input(f"Type option: ")
        else:
            sys_prompt = self.prompts.select_search_suggest
            user_prompt = f'PAGE: {self.page_obs.short_summary}\n\nTASK: {self.intent}\n\nSEARCH FIELD: {self.get_elem_str(search_element)}'
            user_prompt += '\n\nSUGGESTIONS:\n{context}'
            selected = self.llm_choose_list(options_list, options_list, sys_prompt, user_prompt, '**SELECT**:', n_ret=1, add_none=True, inline=False)
            selected_option = next(iter(selected), None)

        if not selected_option:
            logger.debug(f"Cancel")
            return ''
        elif selected_option in self.clicked_suggestions:
            logger.debug(f"Already tried suggestion")
            return ''
        else:
            self.clicked_suggestions.append(selected_option)
        

        # Click search suggestion
        action_desc = f"the '{selected_option}' element"
        vlm_prompt = self.vlm_click_prompt(action_desc, False)
        vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt)
        points = self.parse_vlm_coords(vlm_out, elem_crop.size)
        if not points:
            logger.warning("Invalid coord output")
            return ''
        success = self.click_coords(points[0], section_bbox=crop_bbox, elem_desc=f"the '{selected_option}' suggestion")
        if not success:
            return ''
        image = draw_point_on_image(elem_crop, points[0][0], points[0][1])
        image.save(f"{self.screenshot_dir}/dropdown_click.png")
        
        focused_elem = self.get_foc_elem()
        if (self.env.page.url == start_url) and (focused_elem) and (search_element.has_suggest) and (len(prev_options) < 3):
            next_chosen = self.choose_search_suggest(search_element, prev_options)
            selected_option += " "+next_chosen

        return selected_option


    def search(
        self,
        search_element: Element,
        page_section: PageSection = None,
    ) -> str:
        """Type search term and optionally choose search suggestion.
        If successful return the entered search term, else return empty string."""

        start_url = self.env.page.url
        logger.success(f"Execute search action")
        if search_element.tag not in ['input', 'textarea']:
            logger.warning(f"search_element {search_element.tag} is not an input")
            return ''
        dropdown_crop, _ = self.element_crop(search_element, crop_under=True)

        elem_loc = self.get_elem_locator(search_element, section=page_section)
        if elem_loc.count() != 1:
            logger.warning(f"{elem_loc}, {elem_loc.count()}")
            return ''
        success = self.click_element(search_element, page_section, mouse_button='left')
        if not success:
            return ''

        
        # Clear searchbar
        try:
            elem_loc.clear(force=True)
            elem_loc.focus()
        except Error as e:
            logger.warning(f"{str(e)}")

        # Check search suggestions
        if search_element.has_suggest:
            selected_option = self.choose_search_suggest(search_element)
            if start_url != self.env.page.url:
                return selected_option
        
        query = self.enter_input(search_element, page_section, focused=True, tab_after=False)


        # Check for suggestions dropdown
        selected_option = None
        after_crop, _ = self.element_crop(search_element, crop_under=True)
        if (not self.cfg.llm_only) and (not images_identical(dropdown_crop, after_crop)) and (not search_element.has_suggest) and (not search_element.aria_autocomplete):
            elem_crop, _ = self.element_crop(search_element)
            vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt=f"Are there any suggestions under the input field? Answer yes or no.")
            if vlm_out.startswith('Yes'):
                selected_option = self.choose_search_suggest(search_element)
                if selected_option:
                    query = selected_option

        if (self.env.page.url==start_url):
            focused_elem = self.get_foc_elem()
            if ((not selected_option) or ((focused_elem) and (focused_elem.tag=='input'))) and (self.cfg.compound_actions):
                success = self.click_element(search_element, page_section, mouse_button='Enter', record_action=True)

        return query


    def check_dropdown(self, element: Element, get_options=True) -> bool:
        """"""

        if self.cfg.llm_only:
            return False

        elem_crop, _ = self.element_crop(element)
        if not elem_crop:
            return False
        menu_name = self.get_elem_str(element, format='e', short=True)
        vlm_prompt = f'Is the dropdown menu expanded currently? Answer yes or no.'
        vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt=vlm_prompt)
        if vlm_out.startswith('Yes'):
            has_dropdown = True
        else:
            has_dropdown = False

        if has_dropdown and get_options:
            if not menu_name:
                menu_name = 'expanded'
            vlm_prompt = self.prompts.list_dropdown_md.format(menu_name=menu_name)
            vlm_out = self.model_manager.vlm_call(elem_crop, vlm_prompt)
            options_list = self.extract_bullet_list(vlm_out)
            if options_list:
                element.options = options_list

            element.click_only = True
            element.dropdown_elements = [element]
            
        return has_dropdown


    def get_menuitems(self, element: Element, page_section: PageSection) -> list[Element]:
        """"""

        base_loc = self.get_base_locator(element, page_section, section_strict=True)

        all_item_loc = base_loc.locator('[role="menuitem"]').filter(visible=True).all()
        if not all_item_loc:
            all_item_loc = base_loc.locator('li').locator('a').filter(visible=True).all()
        
        all_menuitems = []
        for menuitem_loc in all_item_loc:
            menuitem = self.elem_from_loc(menuitem_loc)
            if (menuitem):
                all_menuitems.append(menuitem)
        
        if not all_menuitems:
            page_section, elem_diff = self.update_section(page_section, check_dropdown=True)
            all_menuitems = elem_diff['added']

        return all_menuitems


    def click_check_dropdown(self, element: Element, page_section: PageSection) -> bool:
        """Click element and check for dropdown"""

        start_url = self.page_obs.url
        start_tabs = len(self.env.context.pages)
        start_html = self.env.page.content()
        elem_name = element.get_name().lower()
        dropdown_crop, _ = self.element_crop(element, crop_under=True)
        start_dialog = self.check_dialog(element)

        success = self.click_element(element, page_section)
        if not success:
            return False
        element.explored = True

        if (not self.cfg.compound_actions):
            return True
        if (len(self.env.context.pages) > start_tabs):
            return True
        if ((self.env.page.url!=start_url) and (self.env.page.url[:-1]!=start_url)):
            return True
        if (element.input_type=="submit") or (element.role=='presentation'):
            return True
        if (page_section) and (page_section.type=='table'):
            if (element.tag=='a') and (element.href) and (not str(element.href).startswith('#')):
                link_url = element.href
                if not element.href.startswith('http'):
                    link_url = get_full_url(link_url, self.env.page.url)
                self.go_to_url_action([], link_url)
            return True
        if (any(s in elem_name for s in ['clear', 'remove', 'delete', 'next', 'back', 'previous'])):
            return True
        dialog = self.check_dialog(element, check_only=False)
        if (dialog and not start_dialog) or ((dialog and start_dialog) and (dialog.class_name != start_dialog.class_name)):
            logger.debug(f"check_dialog = True")
            chosen_elem = self.choose_element(dialog.elements, dialog, dropdown=True)
            return True
        
        logger.success(f"click_check_dropdown")
        if not page_section:
            logger.debug(f"No page_section")
            return True
        section_loc = self.get_section_locator(page_section)
        if not section_loc.count():
            return True
        elem_loc = self.get_elem_locator(element, page_section)
        if not elem_loc.count():
            page_section, elem_diff = self.update_section(page_section, check_dropdown=True)
            added_elems = elem_diff['added']
            self.choose_element(added_elems, page_section, dropdown=True)
            return True

        after_crop, _ = self.element_crop(element, crop_under=True)
        if (not images_identical(dropdown_crop, after_crop)):
            if (element.aria_haspopup in ['menu', 'true']) or (element.aria_controls):
                added_elems = self.get_menuitems(element, page_section)
            else:
                page_section, elem_diff = self.update_section(page_section, check_dropdown=True)
                added_elems = elem_diff['added']
            n_inputs = 0
            for elem in added_elems:
                if elem.tag in ['input', 'textarea', 'select']:
                    n_inputs += 1
            if (self.is_form(added_elems, dropdown=True)):
                self.submit_form(page_section, added_elems, dropdown=True)
                return True
            elif ((page_section.type=='form') or (self.is_form(page_section.elements))) and (n_inputs>1):
                logger.debug(f"n_inputs: {n_inputs}")
                self.submit_form(page_section)
                return True
            if self.useful_elements(added_elems):
                chosen_elem = self.choose_element(added_elems, page_section, dropdown=True)
                if (chosen_elem) and (chosen_elem.input_type == 'text'):
                    self.click_element(chosen_elem, page_section, mouse_button='Enter')
            elif self.check_dropdown(element, get_options=False):
                logger.success(f"VLM click dropdown option")
                selected_option = self.vlm_click_dropdown(element)
            else:
                element.no_dropdown = True
        elif (self.env.page.content() == start_html):
            logger.debug(f"No page change")
            if (element.tag=='a') and (not str(element.href).startswith('#')):
                link_url = element.href
                if not element.href.startswith('http'):
                    link_url = get_full_url(link_url, self.env.page.url)
                self.go_to_url_action([], link_url)

        return True


    def dropdown_action(self, element: Element, page_section: PageSection, list_loc=None) -> bool:
        """Open dropdown menu and execute action."""

        self.dropdown_level += 1

        start_url = self.env.page.url
        if page_section:
            start_section_h = page_section.bbox.get_abs_px_height()
        logger.success(f"Dropdown Action")

        elem_loc = self.get_elem_locator(element, page_section)
        if (elem_loc.count() != 1) or (elem_loc.is_disabled()):
            logger.warning(f"Locator error for element: {element.get_name()}")
            return False
        
        # Open dropdown menu
        elem_loc.focus()
        if element.destination_page:
            elem_loc.hover()
        elif elem_loc.get_attribute('aria-expanded') != 'true':
            self.click_element(element, page_section, mouse_button='Enter')
            time.sleep(1.0)
            if (self.env.page.url != start_url):
                return True
        
        if (element.get_name()=='Star'):
            return True

        # scroll to bottom of menu if long
        if element.func_desc == "big_element":
            x, y = element.bbox.get_center_xy_abs()
            for i in range(len(element.dropdown_elements) // 15):
                self.env.page.mouse.move(x, y+200)
                time.sleep(0.25)
                self.env.page.mouse.wheel(0.0, 500.0)
                time.sleep(0.25)
                self.env.page.mouse.wheel(0.0, 500.0)


        dropdown_elements = element.dropdown_elements.copy()

        if (len(dropdown_elements)==1):
            if (dropdown_elements[0].equals(element)) and (not self.cfg.llm_only):
                logger.success(f"VLM click dropdown option")
                selected_option = self.vlm_click_dropdown(element)
                if selected_option:
                    return True
                else:
                    return False
            elif dropdown_elements[0].get_name() == element.get_name():
                return True
        if (dropdown_elements[0].equals(element)) or (dropdown_elements[0].tag in ['ol', 'ul']):
            dropdown_elements = dropdown_elements[1:]
        if (page_section) and (page_section.data_level):
            first_only = True
        else:
            first_only = False
        
        menu_id = element.aria_controls
        if menu_id:
            menu_loc = self.get_base_locator(element).locator(f'id={menu_id}')
            if menu_loc.count() == 1:
                page_section = self.create_section(menu_loc, iframe_id=element.iframe_id)
        
        if len(dropdown_elements) < 100:
            updated_elems = self.update_elements(dropdown_elements, page_section, first_only=first_only, get_all=True)
            dropdown_elements = updated_elems['current']
            element.dropdown_updated = True
            logger.debug(f"{len(dropdown_elements)}")
        
        if not menu_id:
            for dropdown_elem in dropdown_elements:
                dropdown_elem.section = element.section

        
        # Choose and click option
        if (menu_id) and ("filter" in element.get_name().lower()):
            self.submit_form(page_section, dropdown_elements, form_loc=menu_loc, dropdown=True)
        elif (self.is_form(dropdown_elements, dropdown=True)):  # or (any(e.contained_in=='form' for e in dropdown_elements)):
            self.submit_form(page_section, dropdown_elements, dropdown=True)
        else:
            chosen_elem = self.choose_element(dropdown_elements, page_section, dropdown=True)
                
        return True
    

    def tab_action(self, tab_element: Element, page_section: PageSection=None) -> bool:
        """Click tab then perform action on tab content section."""

        success = self.click_element(tab_element, page_section)
        if not success:
            return False

        if tab_element.tab_sections:
            start_scroll_h = self.page_scroll_height()
            tab_locator = self.get_elem_locator(tab_element, page_section)
            base_locator = self.get_base_locator(tab_element)
            section_id = tab_locator.get_attribute('aria-controls')
            section_loc = base_locator.locator(f'#{section_id}')
            if not section_loc.is_visible():
                logger.warning(f"{tab_element.get_name()} can't be located\n\n{section_loc}, {section_loc.count()}")
                return False
            subsection_locs = self.split_section(section_loc)
            present_sections = self.create_page_sections(subsection_locs)
            for section in present_sections:
                self.update_section(section)
            self.env.page.evaluate(f"window.scrollTo(0, {start_scroll_h})")
            self.page_obs.html_sections += present_sections

        return True


    def choose_element(self, element_list: list[Element], page_section: PageSection=None, choice_only=False, dropdown=False) -> Element:
        """Prompt LLM to choose from the elements available on the page, then execute action."""

        element_list = self.useful_elements(element_list)
        if not element_list:
            return None
        logger.success(f"Choose element")

        elem_str_list = self.elem_str_list(element_list, page_section)
        if len(element_list) > 100:
            elem_str_list = self.elem_str_list(element_list, page_section, max_options=0)
        
        if dropdown:
            element_list.append(None)
            elem_str_list.append('None')

        chosen_element = self.choose_action(element_list, elem_str_list, dropdown=dropdown)
        if (not chosen_element) and (len(element_list)==1):
            chosen_element = element_list[0]

        # Execute action on element
        if (chosen_element) and (not choice_only):
            self.element_action(chosen_element, page_section, dropdown=dropdown)

        return chosen_element


    def save_screenshot(self, initialize=False, max_height=1280):
        """"""

        if (initialize) and (os.listdir(self.trajectory_screenshot_dir)):
            logger.warning(f"Screenshots folder not empty, clearing.")
            for f in glob.glob(f'{self.trajectory_screenshot_dir}/*'):
                os.remove(f)

        # Find highest screenshot index
        max_index = -1
        filename_pattern = re.compile(r"^(\d+)\.png$")
        for item_name in os.listdir(self.trajectory_screenshot_dir):
            match = filename_pattern.match(item_name)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
        
        file_name = f"{max_index+1}.png"
        screenshot = self.env.get_screenshot(max_height=max_height)
        if not screenshot:
            logger.warning(f"screenshot is None, retry")
            screenshot = self.env.get_screenshot(max_height=max_height)
        if screenshot:
            screenshot.save(f"{self.trajectory_screenshot_dir}/{file_name}")
        else:
            logger.error(f"Failed to save screenshot")

        return file_name


    def log_trajectory(self, initialize=False, scrolled=False, score: int=None):
        """"""

        if (not self.trajectory_log_dir):
            return

        screenshot_file = self.save_screenshot(initialize=initialize)
        log_file_path = f"{self.trajectory_log_dir}/result.json"
        trajectory_data = None

        if initialize:
            # Create new trajectory file
            trajectory_data = {
                "task_id": self.task_id,
                "task": self.intent,
                "final_result_response": self.answer,
                "action_history": [],
                "thoughts": [],
                "score": None
            }
            # save json
            with open(log_file_path, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
        else:
            # load json
            if not os.path.exists(log_file_path):
                logger.error(f"Error: Log file {log_file_path} does not exist.")
                return
            with open(log_file_path, 'r') as f:
                trajectory_data = json.load(f)

            # update trajectory_data
            if trajectory_data:
                if score is not None:
                    trajectory_data["score"] = score
                elif self.answer:
                    trajectory_data["final_result_response"] = self.answer
                else:
                    if scrolled:
                        thought = ""
                        action_str = "Scroll"
                    else:
                        thought = self.prev_thought
                        action_str = self.prev_action
                    logger.debug(f"Update log: img={screenshot_file} thought='{thought}', action_str='{action_str}'")
                    trajectory_data["thoughts"].append(thought)
                    trajectory_data["action_history"].append(action_str)
            
                # save updated json
                with open(log_file_path, 'w') as f:
                    json.dump(trajectory_data, f, indent=4)

        return


    def record_action(self, action: AgentAction, subgoal: str=None):
        """Add action info to trajectory."""

        if not self.save_history:
            return

        action['nth'] = self.time_step
        action['reason'] = self.current_subgoal
        if subgoal:
            action['reason'] = subgoal
        self.action_stack.append(action)
        self.prev_thought = self.current_subgoal
        self.prev_action = action['action_summary']
        self.log_trajectory()

        return

    
    def element_action(
        self,
        element: Element,
        page_section: PageSection = None,
        dropdown=False
    ) -> bool:
        """Execute action on the browser environment using the element."""

        # Scroll element into view if needed
        scroll_height = self.page_scroll_height(iframe_id=element.iframe_id)
        if (element.bbox.y1_abs_px < scroll_height) or (element.bbox.y2_abs_px > scroll_height+self.env.viewport_size['height']-200):
            logger.info(f"initial scroll height: {scroll_height}")
            self.scroll_to_bbox(element.bbox, upper_margin=200)
            logger.info(f"new scroll height: {self.page_scroll_height(iframe_id=element.iframe_id)}")
            self.log_trajectory(scrolled=True)
        if element.func_desc == 'Table filter':
            self.env.page.evaluate(f"window.scrollTo(0, 0)")
        if element.section:
            page_section = element.section
        elem_name = element.get_name().lower()
        self.prev_screenshot = self.env.get_screenshot(max_height=960, iframe_id=element.iframe_id)


        if element.role == 'tab':
            result = self.tab_action(element, page_section)
        elif (self.dropdown_level < 3) and (element.dropdown_elements) and (self.cfg.compound_actions):
            result = self.dropdown_action(element, page_section)
        
        elif (element.input_type == "file"):
            result = self.upload_file(element, page_section)
        elif (element.tag=="select") or ((element.role=="combobox") and ("select" in element.class_name)):
            result = self.select_option(element, page_section)
        elif (element.tag in ["input", "textarea"]) or (element.role == 'spinbutton'):
            if element.input_type in ["submit", "reset", "button"]:
                result = self.click_element(element, page_section)
            elif (element.input_type=="search") or ('search' in element.get_name().lower()) or ('search' in str(element.id)):
                result = self.search(element, page_section)
            elif element.input_type == "checkbox":
                result = self.set_checkbox(element, page_section)
            elif element.input_type == "radio":
                result = self.click_element(element, page_section, mouse_button='left')
            else:
                result = self.enter_input(element, page_section, dropdown=dropdown)

        
        elif (not dropdown) and (not element.explored) and (element.input_type!='submit') and (element.contained_in!='dialog') and ('copy' not in elem_name):
            result = self.click_check_dropdown(element, page_section)
        elif (element.aria_haspopup) and (element.aria_haspopup != "false"):
            result = self.click_check_dropdown(element, page_section)
        elif (element.tag in ["button", "div"]):
            result = self.click_element(element, page_section)
            if ('copy' in elem_name):
                self.copy_to_clipboard()
        else:
            result = self.click_element(element, page_section)

        if result:
            element.clicked = True
        else:
            logger.warning(f"Action error for {element.get_name()}, add to invalid_elems")
            self.invalid_elems.append(element)
        self.post_screenshot = self.env.get_screenshot(max_height=960, iframe_id=element.iframe_id)

        return result
    



    #### ---- Agent Actions ---- ####

    def llm_choose_list(
        self, 
        options_list: list[Any], 
        options_str_list: list[str],
        sys_prompt: str, 
        user_prompt: str=None,
        keyword: str=None,
        backups: list[str] = [],
        chunk_size: int=None, 
        n_ret: int=None,
        trunc: bool=False,
        max_recurse: int=3,
        add_none: bool=False,
        allow_repeat: bool=False,
        inline: bool=True,
        show_prompt=True,
        show_options=False,
        ret_all_on_err=False,
        use_letters=False
    ) -> list[Any]:
        """"""

        # Used for:
        # choose_select_option
        # upload saved file
        # submit_form choose inputs, choose action
        # table_headers choose sort, filter
        # table_iter choose candidates, choose action
        # list_iter choose candidates
        # choose_element
        # choose_section (optional)
        # navigation goto saved page




        # options_list = options_list.copy()
        options_str_list = options_str_list.copy()

        if not chunk_size: chunk_size = len(options_list)
        if not n_ret: n_ret = len(options_list)

        if len(options_list) < 1:
            logger.warning(f"len(options_list) == 0")
            return []
        if len(options_list) == 1:
            return options_list
        if max_recurse <= 0:
            logger.warning(f"Hit recursion limit ({max_recurse}). Return top {n_ret} choices.")
            return options_list[:n_ret]
        if len(options_list) != len(options_str_list):
            logger.warning(f"options={len(options_list)}, labels={len(options_str_list)}")
            print(f"objects: {options_list}")
            print(f"labels: {options_str_list}")
            return []
        

        if len(options_list) > chunk_size:
            chosen_options = []
            chosen_str_list = []

            # Divide and recursively choose from sublists
            logger.debug(f"Decompose list, chunk_size={chunk_size}")
            for i in range(0, len(options_list), chunk_size):
                options_chunk = options_list[i:i + chunk_size]
                str_chunk = options_str_list[i:i + chunk_size]
                chosen_options += self.llm_choose_list(
                    options_chunk, 
                    str_chunk, 
                    sys_prompt, 
                    user_prompt, 
                    keyword=keyword,
                    backups=backups,
                    chunk_size = chunk_size, 
                    n_ret = n_ret,
                    max_recurse = max_recurse-1,
                    allow_repeat=allow_repeat,
                    add_none=add_none,
                    inline=inline,
                    show_prompt=show_prompt,
                    show_options=show_options,
                    ret_all_on_err=ret_all_on_err,
                    use_letters=use_letters
                )
            for i in range(len(options_list)):
                if options_list[i] in chosen_options:
                    chosen_str_list.append(options_str_list[i])

        else:
            chosen_options = []
            chosen_str_list = []

            if add_none:
                options_list.append(None)
                options_str_list.append('None')
            if use_letters:
                letters_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10}
            numbered_list = self.add_list_numbers(options_str_list, indent=True, letter=use_letters)
            options_context = '\n'.join(numbered_list)
            prompt = user_prompt.replace('{context}', options_context)
            if (not show_prompt) and (show_options):
                logger.info(f"\n{options_context}")

            choices = []
            if self.manual:
                choices_output = input(f"{prompt}\n\nManual (split with spaces): ")
                if not choices_output: choices = []
                else: choices = choices_output.split(' ')
            if not choices:
                # Prompt llm to choose list items
                messages = self.format_llm_prompt(sys_prompt, prompt)
                choices_output = self.model_manager.llm_call(messages, show_prompt=show_prompt)
                choice_error = False
                if keyword:
                    line_answer = self.extract_llm_answer(choices_output, keyword, backups, line_only=inline)
                    if line_answer:
                        choices_output = line_answer
                    else:
                        choice_error = True
                    if choices_output.startswith('None'):
                        return []
                if (inline) or (use_letters):
                    if (',' not in choices_output):
                        choices = choices_output.split(' ')
                    else:
                        choices = choices_output.split(',')
                    choices = [c.strip() for c in choices]
                else:
                    pattern = r'^\s*(\d{1,3}|100)\).*$'
                    matches = re.finditer(pattern, choices_output, re.MULTILINE)
                    choices = [match.group(1) for match in matches]
                if not allow_repeat:
                    choices = list(dict.fromkeys(choices))
            
            for c in choices:
                if c.endswith(')'):
                    c = c[:-1]
                try:
                    if use_letters:
                        c = letters_map[c]
                    c_index = int(c) - 1
                    if (c_index<0) or (c_index>=len(options_list)):
                        logger.warning(f"{c}, range={len(options_list)}")
                        continue
                    chosen_str_list.append(options_str_list[c_index])
                    chosen_options.append(options_list[c_index])
                except:
                    logger.warning(f"Invalid index: {c}")
                    choice_error = True
                    continue
                
                # Also add option with name=c in case of confusion
                for i in range(len(options_str_list)):
                    option = options_list[i]
                    if (f' "{c}"' in options_str_list[i]) and (option not in chosen_options):
                        chosen_options.append(options_list[i])
                        break


            if (not chosen_options) and (choice_error):
                if (ret_all_on_err):
                    return options_list
                if (len(options_str_list) > 50) and (max_recurse > 1):
                    logger.debug(f"LLM error, retry with smaller chunk size")
                    chunk_size = chunk_size // 2
                    chosen_options = self.llm_choose_list(
                        options_list, 
                        options_str_list, 
                        sys_prompt, 
                        user_prompt, 
                        keyword=keyword,
                        backups=backups,
                        chunk_size = chunk_size, 
                        n_ret = n_ret,
                        max_recurse = max_recurse-1,
                        allow_repeat=allow_repeat,
                        add_none=add_none,
                        inline=inline,
                        show_prompt=show_prompt,
                        show_options=show_options,
                        ret_all_on_err=ret_all_on_err,
                        use_letters=use_letters
                    )

        # Narrow down candidates if needed
        if len(chosen_options) > n_ret:
            logger.debug(f"Chosen={len(chosen_options)} > max_matches={n_ret}")
            if trunc:
                print("Truncate")
                return chosen_options[:n_ret]
            # Let LLM choose
            chosen_options = self.llm_choose_list(
                chosen_options, 
                chosen_str_list, 
                sys_prompt, 
                user_prompt, 
                keyword=keyword,
                backups=backups,
                chunk_size = chunk_size, 
                n_ret = n_ret,
                max_recurse = max_recurse-1,
                allow_repeat=allow_repeat,
                add_none=add_none,
                inline=inline,
                show_prompt=show_prompt,
                show_options=show_options,
                ret_all_on_err=ret_all_on_err,
                use_letters=use_letters
            )
            chosen_options = chosen_options[:n_ret]
        
        return chosen_options


    #### ---- Browser Context ---- ####

    # Element
    def get_input_value(self, input_element: Element=None, section: PageSection=None, input_loc=None) -> str:
        """Return current value of the input field, return None if error."""

        if (not input_loc) or (not isinstance(input_loc, Locator)):
            input_loc = self.get_elem_locator(input_element, section)
        if isinstance(input_loc, Locator):
            if not input_loc.count() == 1:
                return None
        if input_element.tag not in ["input", "textarea", "select"]:
            return None
        
        if (input_element.tag=="input") and ((input_element.role=="combobox") or (input_element.input_type=="file")):
            if input_element.input_value:
                return input_element.input_value

        try:
            if (input_element.tag == "select"):
                selected_text = input_loc.evaluate("sel => sel.options[sel.options.selectedIndex].textContent")
                if not selected_text:
                    selected_text = ""
                return selected_text
            elif (input_element.input_type == "radio") or (input_element.input_type == "checkbox"):
                checked = input_loc.is_checked(timeout=500)
                return "checked" if checked else ""
            elif (input_element.input_type == "file"):
                uploaded_file = input_loc.input_value().split('/')[-1]
                uploaded_file = uploaded_file.split('\\')[-1]
                return uploaded_file
            else:
                value = input_loc.input_value(timeout=500)
                return value
        except Error as e:
            logger.warning(f"{e.message}elem: {input_element.get_name()}\n")

            return None
    

    def get_elem_str(
        self, 
        element: Element, 
        format: Literal['a','b','c','d','e'] = 'd', 
        section: PageSection=None,
        short: bool=False,
        concise: bool=False,
        max_options: int=30
    ) -> str:
        """Return string describing basic details about the element."""

        value = element.input_value
        
        href = None
        text = clean_text(element.text, max_words=25)
        name = clean_text(element.get_name(), max_words=25)
        if (element.label):
            label = clean_text(element.label, max_words=25, max_lines=1)
            if label not in f" - {name}":
                name = f"{label} - {name}"
            if concise:
                name = label
        if (element.date_format):
            name += f" {element.date_format}"
        if (element.func_desc == f'Table filter'):
            if element.name:
                name = f"Filter table by {element.name}"
            else:
                name = f"Filter table - {name}"
        if (element.tag=='li') and (text):
            name = text
        input_type = ""
        options = ""
        if (element.options) and (max_options > 0) and (not name.startswith('Search')):
            if len(element.options) < max_options:
                options = str(element.options)
            else:
                options = str(element.options[:5])
                options = options[:-1] + f", ...]"

        # toggle elements
        if element.dropdown_elements:
            elem_role = "dropdown menu"
            if (not element.click_only) and (max_options > 0):
                option_names = []
                for elem in element.dropdown_elements:
                    if (elem.tag=='button') and ('close' in elem.get_name()):
                        continue
                    option_names.append(elem.get_name())
                if (len(element.dropdown_elements)<max_options):
                    options = str(option_names)
                else:
                    options = str(option_names[:5])
                    options = options[:-1] + f", ...]"
            if (self.intent):
                if (not element.dropdown_updated) and (element.section) and (element.section.in_main):
                    logger.debug(f"{name} dropdown options not updated yet")
                    options = ""
        elif element.role == "tab":
            elem_role = "tab"

        # input elements
        elif element.tag == "select":
            elem_role = "combobox"
        elif (element.role == "combobox") and (element.options):
            elem_role = "combobox"
            value = text
        elif element.tag == "textarea":
            elem_role = "textarea"
        elif (element.tag == "input"):
            elem_role = "input"
            input_type = element.input_type
            if (input_type=="radio") or (input_type=="checkbox"):
                elem_role = input_type
                input_type = ""
            if (input_type == "text"):
                elem_role = "textbox"
                input_type = ""
            if (input_type == "submit"):
                elem_role = "button"
                name = "Submit"
                input_type = ""
            if (element.role == "combobox") and (name != "Search"):
                elem_role = "combobox"
            if element.fieldset:
                if len(element.fieldset) > 1:
                    name = f"{element.fieldset[-1]} {name}"
        
        # navigation elements
        elif element.tag == "a":
            elem_role = "link"
            if (element.href) and (not element.href.startswith('javascript')):
                if element.href.split('.')[-1] in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp']:
                    name = "Image"
                if (not name) and (any(w in element.class_name for w in ['photo', 'image', 'img'])):
                    name = "Image"
                href = get_sim_url(element.href)
        elif element.tag == "button":
            elem_role = "button"
            if element.input_type == "submit":
                input_type = element.input_type
        elif element.role:
            elem_role = element.role
        elif element.tag in ['span', 'i']:
            elem_role = "button"
        else:
            elem_role = element.tag


        if format == 'a':
            elem_str = f'name: "{name}" role: "{elem_role}"'
            if value:
                elem_str += f' value: "{value}"'
        elif format == 'b':
            elem_str = f'{name}: role="{elem_role}"'
            if value:
                elem_str += f' value="{value}"'
        elif format == 'c':
            value_str = ""
            if value:
                value_str = f', value="{value}"'
            elem_str = f'"{name}" (role="{elem_role}"{value_str})'
        elif format == 'd':
            elem_str = f'{elem_role} "{name}"'
            if input_type:
                elem_str += f' type="{input_type}"'
            if short:
                return elem_str
            if element.tag == 'input':
                if (element.input_type not in ['radio', 'checkbox']):
                    if value:
                        elem_str += f' value="{value}"'
                    elif elem_role != 'combobox':
                        elem_str += f' (empty)'
                elif (element.input_type in ['radio', 'checkbox']):
                    if value:
                        elem_str += f' value="{clean_text(str(value))}"'
                    else:
                        elem_str += f' (empty)'
                if (element.input_type == 'range'):
                    if element.attributes_dict.get('max'):
                        elem_str += f" (max {element.attributes_dict.get('max')})"
            if element.tag in ['textarea', 'select']:
                if value:
                    elem_str += f' value="{clean_text(str(value))}"'
                else:
                    elem_str += f' (empty)'
            if href:
                elem_str += f' href="{href}"'
            if options:
                elem_str += f' options={options}'
            if element.func_desc == 'big_element':
                elem_str += f' - Main navigation menu'
        elif format == 'e':
            if short:
                return f'"{name}"'
            elem_str = f'the "{name}" {elem_role}'
        else:
            logger.error(f"Invalid elem_str format: {format}")
            return name
        
        if not concise:
            if (element.contained_in == 'dialog') and (len(element.section.elements)<10):
                elem_str = f"dialog {elem_str}"
            if element.row_info:
                elem_str += f" {element.row_info}"
        
        if (element.input_type=='radio') and (not element.label):
            element.more_info = f'id="{element.id}"'
        if element.more_info:
            elem_str += f", {element.more_info}"
        if (element.required) or ('required' in element.class_name):
            if (not element.input_value):
                elem_str += f" *required"
        if (element.clicked) and (element.input_type!='submit'):
            elem_str += f' (clicked)'
        if element.is_disabled:
            elem_str += f' (disabled)'
        
        if element.aria_selected:
            elem_str += f" [selected={element.aria_selected}]"
        if element.aria_expanded == 'true':
            elem_str += f" [expanded]"

        return elem_str


    # Section
    def useful_elements(self, element_list: list[Element], section: PageSection=None) -> list[Element]:
        """"""

        useful_elems = []

        for elem in element_list:
            
            if elem.in_list(self.invalid_elems):
                continue

            elem_name = elem.get_name().lower()
            if (elem.tag=='a'):
                if (elem.href) and (not elem.href.startswith('#')):
                    if get_full_url(str(elem.href), self.env.page.url) == self.env.page.url:
                        continue
                if (elem.href) and (elem.href.startswith('tel:')):
                    continue
                if (elem.href=='#'):
                    if ('active' in elem.class_name) and (elem.clicked):
                        continue
                    if (elem_name == 'layers'):
                        continue
                if (str(elem.href).startswith('https')):
                    if not self.check_url_allowed(elem.href):
                        continue
                if ('feed_token' in str(elem.href)):
                    continue
            if (elem.tag not in ['input', 'textarea']):
                if (elem.open):
                    continue
                if (elem.aria_expanded=='true'):
                    continue
            if (elem.role=='tab') and (elem.aria_selected=='true'):
                continue
            if ('tab' in str(elem.data_qa)) and ('active' in str(elem.class_name)):
                continue
            if elem.in_list(self.invalid_elems):
                continue
            if not elem.bbox:
                continue
            if (elem.tag=='button') and (elem.bbox.get_abs_px_width()<=50):
                if ('toolbar' in str(elem.parent_class)):
                    continue
            if (elem.bbox.get_abs_px_width()<=50) and ('draggable' in elem.class_name):
                continue
            if (section) and (section.done_form) and (elem.input_type=='submit'):
                continue
            if ("full screen" in elem_name):
                continue
            if (elem_name.startswith('print')):
                continue
            if elem_name=='undefined':
                continue
            if (elem.tag=='div'):
                if (len(elem.dropdown_elements)==1) and (elem.contained_in=='form'):
                    continue
                if (elem.class_name == "fieldset") and (elem.bbox.get_abs_px_width() > 250):
                    continue
            if (elem.tag=='span') and (elem.attributes_dict.get('aria-readonly')=="true"):
                continue
            if (elem.input_type in ['radio', 'checkbox']) and (elem.clicked):
                continue
            if (elem.tag == 'select') and (not elem.text) and (not elem.options):
                continue
            if elem.bbox.get_abs_px_width() == 0:
                print(elem.bbox.get_abs_px_coords())
                continue

            useful_elems.append(elem)

        return useful_elems


    def elem_str_list(self, element_list: list[Element], page_section=None, max_options=30, add_action=False) -> list[str]:
        """"""

        str_list = []

        if (self.cfg.benchmark=='workarena') and (page_section) and (page_section.n_dropdowns >= 5):
            max_options = 5

        for i in range(len(element_list)):
            element = element_list[i]
            elem_str = self.get_elem_str(element, section=page_section, max_options=max_options)
            if add_action:
                if (element.tag in ["input", "textarea", "select"]) or (element.role == "combobox"):
                    elem_str = f"edit {elem_str}"
                else:
                    elem_str = f"click {elem_str}"
            str_list.append(elem_str)

        return str_list


    def add_elem_info(self, all_elements: list[Element]):
        """"""

        logger.debug(f"Make element names unique")
        elem_names = {all_elements[0].get_name()}

        for elem in all_elements:
            if elem.id:
                elem.more_info = f'id="{elem.id}"'
            elif elem.data_track_property:
                elem.more_info = f'property="{elem.data_track_property}"'
            elif elem.data_testid:
                elem.more_info = f'id="{elem.data_testid}"'

        return


    def elems_context(self, all_elements: list[Element], page_section=None, numbered=True, max_options=30) -> str:
        """Get description of elements in the PageSection to provide context in LLM prompt."""

        all_elements = self.useful_elements(all_elements)
        concise = False

        if (page_section) and (self.intent):
            if (self.cfg.benchmark=='workarena') and (page_section.n_dropdowns >= 5):
                max_options = 5
            all_elems = []
            for elem in all_elements:
                if not elem.section:
                    elem.section = page_section
                if (elem.func_desc == "big_element") and (not elem.dropdown_elements):
                    website_mem = self.website_mem
                    elem = elem.in_list(website_mem.big_elements)
                all_elems.append(elem)
            all_elements = all_elems

        element_list = []
        add_info = False

        for element in all_elements:
            elem_str = self.get_elem_str(element, section=page_section, concise=concise, max_options=max_options)
            short_str = self.get_elem_str(element, section=page_section, short=True)
            
            if (elem_str in element_list) or (short_str in element_list):
                add_info = True
                break
            element_list.append(elem_str)

        if add_info:
            self.add_elem_info(all_elements)
            element_list = []
            for element in all_elements:
                elem_str = self.get_elem_str(element, section=page_section, concise=False)
                element_list.append(elem_str)

        context = ""
        for i in range(len(element_list)):
            elem_str = element_list[i]
            if numbered:
                context += f'{i+1}. {elem_str}\n'
            else:
                context += f'- {elem_str}\n'

        return context.strip()
    

    def analyze_section_images(self, section_loc: Locator, text: str, img_query: str=None, md: bool=False):
        """"""

        all_images = dict()
        all_img_desc = []

        saved_image_captions = {}
        if self.page_obs.list_section:
            saved_image_captions = self.page_obs.list_section.image_captions

        img_locs = section_loc.locator('img').all()
        for img_loc in img_locs:
            try:
                img_url = img_loc.get_attribute('src', timeout=1000)
            except Error as e:
                logger.warning(f"{str(e)}")
                continue
            if (not img_url) or ('/placeholder/' in img_url) or (img_url in all_images):
                continue
            img_size = img_loc.bounding_box()
            if (not img_loc.is_visible()) or (img_size['width'] < MIN_IMG_X) or (img_size['height'] < MIN_IMG_Y):  # ignore < 50px
                continue
            if '_thumbnail.png' in img_url:
                img_url = img_url.replace('_thumbnail.png', '.png')
            img_file = img_url.split('/')[-1]
            if img_file in text:
                if md:
                    img_url = extract_file_url_md(text, img_file)
                else:
                    img_url = extract_url_snapshot(text, img_file)
            try:
                img_loc.scroll_into_view_if_needed()
                img_url = get_full_url(img_url, self.env.page.url)
                res = requests.get(img_url)
                if res.status_code == 200:
                    logger.info(f"Page img: {img_url}")
                    image = Image.open(BytesIO(res.content))
                    # VLM describe
                    vlm_prompt = self.prompts.desc_img_short
                    if img_query:
                        vlm_prompt = img_query
                    elif (img_size['height']<100):
                        vlm_prompt = self.prompts.desc_thumbnail
                    elif (img_size['height']>250 and img_size['width']>250):
                        vlm_prompt = self.prompts.desc_img_long
                        self.save_img(image=image, image_path=img_url)
                    elif (self.cfg.benchmark == 'visualwebarena'):
                        vlm_prompt = self.prompts.desc_img_med
                    
                    if (img_url in saved_image_captions):
                        vlm_out = saved_image_captions[img_url]
                    else:
                        vlm_out = self.model_manager.vlm_call(image, vlm_prompt, image_path=img_url)
                    
                    if (vlm_prompt == self.prompts.desc_img_long):
                        if len(vlm_out) > 10000:
                            vlm_out = vlm_out[:10000] + ' ...'
                    else:
                        if len(vlm_out) > 2000:
                            vlm_out = vlm_out[:500] + ' ...'
                    
                    all_img_desc.append(f'{img_url}: {vlm_out}')
                    all_images[img_url] = {
                        "image": image,
                        "desc": vlm_out
                    }
                else:
                    logger.warning(f"Image load failed with status code {res.status_code}, {res.text}")
            except Exception as e:
                logger.warning(f"Error while loading image {img_url}: {str(e)}")
                continue

        return all_images, all_img_desc


    def section_content_md(
        self, 
        section_loc: Locator, 
        section: PageSection=None,
        img_query: str=None, 
        remove_nested: str=None, 
        caption_img=True,
        md_only = False,
        escape_misc = False,
        add_quotes = True,
        scout=False
    ) -> dict[str, Any]:
        """Convert html into markdown and describe any images to provide context for LLM."""

        if section_loc.count() != 1:
            return {"text": '', "images": []}
        
        # Highcharts
        has_chart = False
        if (not self.cfg.llm_only) and (section) and (section_loc.locator('svg[aria-label="Interactive chart"]').count()):
            has_chart = True
            try:
                section_loc.scroll_into_view_if_needed()
            except Error as e:
                pass
            if scout:
                prompt = 'Provide the title of this chart followed by a brief description of how its data is presented.'
                chart_info = self.describe_screen(prompt=prompt, section_loc=section_loc)
                chart_info += f"\n\n- Select this item to view full chart details."
            else:
                chart_info = self.chart_interact(section_loc)
                if section.chart_info:
                    section.chart_info += '\n\n'
                section.chart_info += chart_info
            return {"text": chart_info, "images": [], "has_chart": True}


        info_str = ""

        # Convert html into markdown
        section_html = section_loc.inner_html()
        section_html = remove_html_tags(section_html, tags_to_remove=['script', 'option'])

        if remove_nested:
            section_html = remove_nested_html(section_html, remove_nested)
        section_md = md(
            section_html, 
            heading_style='atx_closed', 
            escape_misc=escape_misc, 
            keep_inline_images_in=['td', 'span']
        )
        section_md = process_md(section_md, self.env.page.url)

        # Images
        all_images = []
        all_img_desc = []
        if (caption_img) and (not self.cfg.llm_only):
            all_images, all_img_desc = self.analyze_section_images(section_loc, section_md, img_query, md=True)


        all_table_info = []

        has_list = False
        if section_loc.locator('ol, ul').filter(visible=True).count():
            has_list = True

        for time_loc in section_loc.locator('time').all():
            text = time_loc.text_content()
            title = time_loc.get_attribute('title')
            if title:
                section_md = section_md.replace(text, title)


        # Format info
        image_str = ''
        if all_img_desc:
            image_str = f'Image captions:'
            for desc in all_img_desc:
                image_str += f"\n- {desc}"
        
        for table_info in all_table_info:
            section_md += f'\n{table_info}'
        if len(section_md) > 12000:
            section_md = section_md[:12000]

        if add_quotes:
            text_str = f'"""\n{section_md}\n"""'
        else:
            text_str = section_md
        if image_str:
            text_str = '\n\nText:\n' + text_str

        info_str = f"{image_str}{text_str}"
        if md_only:
            info_str = section_md
        info_str = get_sim_url(info_str)

        output = {
            "text": info_str,
            "images": all_images,
            "tables": all_table_info,
            "has_list": has_list,
            "has_chart": has_chart
        }

        return output
    

    def section_content(
        self, 
        section_loc: Locator, 
        section: PageSection=None,
        img_query: str=None, 
        remove_nested: str="", 
        caption_img=True,
        md_only = False,
        escape_misc = False,
        add_quotes = False,
        scout=False
    ) -> dict[str, Any]:
        """Convert html into markdown and describe any images to provide context for LLM."""

        if section_loc.count() != 1:
            return {"text": '', "images": []}
        
        # Highcharts
        has_chart = False
        if (not self.cfg.llm_only) and (section) and (section_loc.locator('svg[aria-label="Interactive chart"]').count()):
            has_chart = True
            try:
                section_loc.scroll_into_view_if_needed()
            except Error as e:
                pass
            if scout:
                prompt = 'Provide the title of this chart followed by a brief description of how its data is presented.'
                chart_info = self.describe_screen(prompt=prompt, section_loc=section_loc)
                chart_info += f"\n\n- Select this item to view full chart details."
            else:
                chart_info = self.chart_interact(section_loc)
                section.chart_info = chart_info
            return {"text": chart_info, "images": [], "has_chart": True}


        info_str = ""
        
        snapshot = section_loc.aria_snapshot(timeout=30000)
        snapshot = snapshot.replace(' [disabled]', '')
        remove_tags = ['table']
        if remove_nested:
            remove_tags.append(remove_nested)
        snapshot = process_aria(snapshot, remove_tags=remove_tags)

        # Images
        all_images = []
        all_img_desc = []
        if (caption_img) and (not self.cfg.llm_only):
            all_images, all_img_desc = self.analyze_section_images(section_loc, snapshot, img_query)

        # Table
        all_table_info = []
        table_locs = section_loc.locator('table').filter(visible=True).all()
        for table_loc in table_locs:
            table_html = table_loc.inner_html()
            table_md = md(table_html, heading_style='atx_closed', escape_misc=False, keep_inline_images_in=['td'])
            table_md = process_md(table_md, self.env.page.url)
            all_table_info.append(table_md)

        has_list = False
        if section_loc.locator('ol, ul').filter(visible=True).count():
            has_list = True

        for time_loc in section_loc.locator('time').all():
            text = time_loc.text_content()
            title = time_loc.get_attribute('title')
            if title:
                snapshot = snapshot.replace(text, title)


        # Format info
        image_str = ''
        if all_img_desc:
            image_str = f'Image captions:'
            for desc in all_img_desc:
                image_str += f"\n- {desc}"
        
        for table_info in all_table_info:
            snapshot += f'\n{table_info}'
        if len(snapshot) > 20000:
            snapshot = snapshot[:20000]

        if add_quotes:
            text_str = f'"""\n{snapshot}\n"""'
        else:
            text_str = snapshot
        if image_str:
            text_str = '\n\nText:\n' + text_str

        info_str = f"{image_str}{text_str}"
        if md_only:
            info_str = snapshot
        

        info_str = get_sim_url(info_str)
        output = {
            "text": info_str,
            "images": all_images,
            "tables": all_table_info,
            "has_list": has_list,
            "has_chart": has_chart
        }

        return output

    
    def chart_interact(self, section_loc: Locator):
        """"""

        frame_loc = self.env.page.locator('body')
        bounding_box = frame_loc.bounding_box()
        xr = bounding_box['x'] + bounding_box['width'] - 1
        xb = bounding_box['y'] + bounding_box['height'] - 1
        frame_loc.click(position={"x": xr, "y": xb}, force=True)

        prompt = 'Provide the title of this chart and summarize the data it displays.'
        chart_desc = self.describe_screen(prompt=prompt, section_loc=section_loc)
        chart_info = f"### Interactive Chart:\n{chart_desc}"

        all_series_locs = section_loc.locator('g.highcharts-tracker[aria-hidden=false]').all()
        all_series = []
        label_format = True
        for series_loc in all_series_locs:
            aria_label = str(series_loc.get_attribute('aria-label'))
            all_series.append((series_loc, aria_label.split(', ', 1)[-1]))
            if ' of ' not in aria_label:
                label_format = False
        if label_format:
            all_series = sorted(all_series, key=lambda x: x[1])

        point_labels = []
        all_point_locs = []
        for series_loc, label in all_series:
            point_locs = series_loc.locator('[role=img]').all()
            for point_loc in point_locs:
                label = str(point_loc.get_attribute('aria-label'))
                label = re.sub(r'^\d+\. ', '', label)
                point_labels.append(f"'{label}'")
                all_point_locs.append(point_loc)
        options_str = self.format_list_to_str(point_labels, numbered=True)

        # Choose point to interact with
        selected_point_loc = None
        if self.manual:
            index = int(input(f"{options_str}\n\nEnter index: "))
            selected_point_loc = all_point_locs[index-1]
        else:
            user_prompt = self.prompts.choose_chart_point
            user_prompt += f"\n\nTask: '{self.intent}'\n\n{chart_info}"
            user_prompt += f"\n\nData points:\n{options_str}"
            messages = self.format_llm_prompt(user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            choices_str = self.extract_llm_answer(llm_out, keyword='SELECT POINT:')
            if (choices_str) and (not choices_str.startswith('None')):
                index = int(choices_str)
                selected_point_loc = all_point_locs[index-1]

        if selected_point_loc:
            if series_loc.locator('rect[role=img]').count():
                selected_point_loc.hover(force=True)
            elif series_loc.locator('path[role=img]').count():
                selected_point_loc.focus()
                for _ in range(index):
                    self.key_press_action([], 'ArrowRight')
            time.sleep(0.5)
            prompt = 'Extract the exact text contained in the chart tooltip. Provide the extracted text only.'
            tooltip_info = self.describe_screen(prompt=prompt, section_loc=section_loc)
            chart_info += f"\n\nAll data points:\n{options_str}\n\n{tooltip_info}"
        else:
            chart_info += f"\n\nAll data points:\n{options_str}"
            chart_info += f"\n\nThe chart does not contain any data points relevant to the task."

        return chart_info


    def section_info(self, page_section: PageSection, remove_nested: str=None, caption_img=True, vlm_info=False, scout=False):
        """Extract and format context of the page section to provide to extract_details prompt."""

        website_name = site_name(self.env.page.url)

        if page_section.chart_info:
            return page_section.chart_info
        if (page_section.type=='form') and (page_section.input_modified):
            return self.elems_context(page_section.elements, page_section, numbered=False)
        if (page_section.bbox.get_abs_px_height()<100) or (page_section.bbox.get_abs_px_width()<100):
            if (page_section.n_dropdowns):
                return self.elems_context(page_section.elements, page_section, numbered=False)
        
        section_loc = self.get_section_locator(page_section)

        # List, table and form sections
        if (page_section.list_type):
            if (len(page_section.list_items) >= 50) or (len(page_section.relevant_items_str)>5000):
                return page_section.relevant_items_str
            if (page_section.n_items>10) or (page_section.extracted_attr_info) or (1 < len(page_section.list_items) < 10):
                list_details = f"{page_section.summary}"
                list_details += f"\n\n{page_section.relevant_items_str}"
                return list_details
            else:
                return page_section.section_text
        if (page_section.type == 'form') or (self.is_form(page_section.elements)):
            if (page_section.type == 'form') and (not self.cfg.llm_only):
                if (page_section.bbox.get_abs_px_height()>1280):
                    return self.elems_context(page_section.elements, page_section, numbered=False)
                vlm_prompt = self.prompts.highlight_interface.format(website_name=website_name)
                page_section.desc = self.describe_screen(page_section, prompt=vlm_prompt)
                info = f"{page_section.desc}\n\n### Elements:\n"
            else:
                info = self.section_content(section_loc, page_section, img_query=False, add_quotes=False)['text']
                info += f"\n\n### Elements:\n"
            return info + self.elems_context(page_section.elements, page_section, numbered=False)
        

        if not section_loc.count():
            logger.warning(f"{section_loc}, {section_loc.count()}")
            return ""
        elif section_loc.count() == 1:
            section_loc.focus()

        # Markdown
        section_data = self.section_content(section_loc, section=page_section, remove_nested=remove_nested, caption_img=caption_img, add_quotes=False)
        if (page_section.role == 'code'):
            section_data = self.section_content_md(section_loc, remove_nested=remove_nested, caption_img=caption_img, add_quotes=False)
        section_text = section_data['text']
        page_section.section_text = section_text


        elems_context = self.elems_context(page_section.elements, page_section, numbered=False)
        if (page_section.n_dropdowns) or ('(clicked)' in elems_context):
            section_text += f"\n\nElements:\n{elems_context}"
        
        if ('iframe' in section_text):
            section_text = elems_context

        return section_text




    # Form Section
    def is_form(self, element_list: list[Element], dropdown=False) -> bool:
        """Returns True if element_list contains form inputs."""

        non_form = 0
        input_elems = 0
        buttons = 0

        for elem in element_list:
            if (elem.tag == "button") and ((elem.input_type == "submit") or (elem.get_name() in ["Post", "Comment"])):
                buttons += 1
            elif (elem.contained_in == "form") and (elem.tag in ["input", "textarea", "select", "label"]):
                if (elem.input_type in ["submit", "reset", "button"]):
                    buttons += 1
                elif (elem.input_type != "search") and ('search' not in elem.get_name().lower()):
                    input_elems += 1
            else:
                non_form += 1

        if (dropdown) and (input_elems>0) and (buttons>0):
            return True
        elif (input_elems > 0) and (input_elems+buttons > 2):
            return True
        else:
            return False
    

    def sort_elements(self, elements: list[Element], form_elems=[], new_elems=[], dropdown=False, form_loc=None, no_clicked=True):
        """Return dict with elements grouped by type."""

        elements = self.useful_elements(elements)

        input_elems = []
        comboboxes = []
        radios = []
        textboxes = []

        action_elems = []
        tab_elems = []
        button_elems = []
        submit_elems = []
        other_elems = []

        for elem in elements:
            elem_name = elem.get_name()
            if (elem.contained_in != 'form') and (elem.input_type!='submit'):
                if not elem.in_list(form_elems):
                    logger.warning(f"skip {elem_name}")
                    continue
            if (dropdown and (not form_loc)) and (not elem.in_list(form_elems)) and (not elem.in_list(new_elems)):
                logger.warning(f"skip {elem_name}")
                continue
            if (elem.tag in ["input", "textarea", "select", "label"]) or (elem.role == "combobox"):
                if (elem.input_type=='submit'):
                    submit_elems.append(elem)
                if (no_clicked) and (elem.clicked) and (not elem.input_value):
                    continue
                input_elems.append(elem)
                if (elem.tag=='select') or (elem.role=='combobox'):
                    comboboxes.append(elem)
                elif elem.input_type in ['radio', 'checkbox']:
                    radios.append(elem)
                else:
                    textboxes.append(elem)
            else:
                action_elems.append(elem)
                if (elem.role == "tab"):
                    tab_elems.append(elem)
                elif (elem.tag == "button"):
                    button_elems.append(elem)
                    if (elem.input_type == 'submit') or ('Submit' in elem_name) or any(elem_name.startswith(s) for s in ["Post", "Comment", "Create"]):
                        submit_elems.append(elem)
                else:
                    other_elems.append(elem)

        grouped_elems = {
            "input_elems": input_elems,  #comboboxes + radios + textboxes,
            "action_elems": action_elems,
            "tab_elems": tab_elems,
            "button_elems": button_elems,
            "submit_elems": submit_elems,
            "other_elems": other_elems,
        }

        return grouped_elems
    

    def field_action(self, field: Element, page_section: PageSection, optional=False, recurse=True, max_retries=1):
        """"""

        start_url = self.env.page.url

        button_clicked = False
        if (field.tag not in ['input', 'textarea', 'select', 'label']) and (field.role!='combobox'):
            self.click_element(field, page_section)
            button_clicked = True
            if (not field.get_name().startswith('Add')) or (self.env.page.url != start_url):
                return

        if (field.input_value == "") and (self.get_input_value(field, page_section)):
            field.input_edited = True
        if (field.input_edited) and (not field.clicked):
            logger.debug(f"auto-filled")
            return
        
        if not button_clicked:
            input_success = self.element_action(field, page_section)
            if (field.aria_invalid==True) and (field.tag in ['input', 'textarea']):
                logger.info(f"Fix invalid field value")
                input_success = self.fix_input_error(field, page_section)
        
        if (self.env.page.url != start_url) and (self.env.page.url[:-1] != start_url):
            logger.debug(f"page url changed")
            return

        page_section, elem_diff = self.update_section(page_section, inputs_only=True)
        added_elems = elem_diff['added']
        field_coords = field.bbox.get_abs_px_coords()
        for new_elem in added_elems:
            if new_elem.tag in ["input", "textarea", "select"]:
                print(f"New field: {self.get_elem_str(new_elem)}")
                if (new_elem.input_type != 'radio') and (new_elem.bbox.get_abs_px_coords() != field_coords):
                    if (len(added_elems) == 1) and (recurse):
                        self.field_action(new_elem, page_section, recurse=False)
                    else:
                        self.element_action(new_elem, page_section)
        
        field.input_value = self.get_input_value(field, page_section)
        if self.slow_mode: self.wait_user_input()

        return

    
    def form_check(self, elements: list[Element], grouped_elems: dict, prev_action_result: str=None):
        """"""

        logger.debug(f"Check if form complete")

        sys_prompt = self.prompts.form_check
        if self.form_plan:
            task = f"TASK: {self.form_plan}"
        elif grouped_elems['tab_elems']:
            sys_prompt = self.prompts.form_checktab
            task = f"USER: '{self.intent}'"
        elif grouped_elems['submit_elems']:
            sys_prompt = self.prompts.submit_check
            task = f"USER: '{self.intent}'"
            if any(elem.get_name()=="Continue" for elem in grouped_elems['submit_elems']):
                task += f"\n\nHISTORY:\n{self.episode_history()}"
        else:
            task = f"TASK: {self.intent}"
        if self.cfg.current_date:
            date_str = date.today().strftime("%B %d, %Y")
            task += f" (message sent {date_str})"
        progress = self.progress_str(add_current=True, subgoal_actions=True)
        if prev_action_result:
            progress += f"\n\n**Update**: {prev_action_result}"
        form = "FORM MENU:\n{context}"
        user_prompt = f"{task}\n\n{progress}\n\n{form}"

        if len(elements) < 20:
            max_options = 20
            add_action = True
        else:
            max_options = 0
            add_action = False
        elems_info = self.elems_context(elements)
        options_str_list = self.elem_str_list(elements, max_options=max_options, add_action=add_action)
        elements.append(None)
        options_str_list.append('Exit')

        chosen = self.llm_choose_list(
            elements, 
            options_str_list, 
            sys_prompt, 
            user_prompt, 
            keyword='**CHOICE**:',
            backups=['CHOICE:', 'CHOICE:**'],
            inline=True
        )
        option = next(iter(chosen), None)
        logger.info(f"chosen option: {option}")

        if self.slow_mode: self.wait_user_input()
        form_complete = (not option)

        return option


    def submit_form(self, page_section: PageSection, form_elements: list[Element]=[], form_loc=None, dropdown=False, max_actions=15):
        """"""

        logger.success(f"Submit form")
        start_url = self.env.page.url

        if form_loc:
            page_section = self.create_section(form_loc, page_section.iframe_id)
            page_section.is_form = True
            page_section, elem_diff = self.update_section(page_section, inputs_only=False)
            self.form_plan = self.intent
        else:
            page_section, elem_diff = self.update_section(page_section, form_elements, inputs_only=True)
        if not page_section:
            section_loc = self.env.page.locator('main')
            if section_loc.locator('form').count():
                section_loc = section_loc.locator('form').first
            page_section = self.create_section(section_loc)
            page_section, elem_diff = self.update_section(page_section, form_elements, inputs_only=True)
            form_elements = page_section.elements

        if form_elements and not form_loc:
            elements = form_elements
        elif (page_section.type=='form') or (form_loc):
            elements = page_section.elements
        else:
            elements = []
            for elem in page_section.elements:
                if elem.contained_in == "form":
                    elements.append(elem)

        grouped_elems = self.sort_elements(elements, no_clicked=False)
        inputs_only = False
        if (form_loc) or (dropdown) or (len(page_section.elements)>25 and grouped_elems['submit_elems']):
            logger.debug(f"update inputs only")
            inputs_only = True
        input_elements = grouped_elems['input_elems']
        n_inp_elems = len(input_elements)
        if (n_inp_elems < 2) and (len(elements) > 1):
            input_elements = elements
        inputs_str_list = self.elem_str_list(input_elements, page_section)

        logger.debug(f"Choose form fields")
        sys_prompt = self.prompts.choose_form_fields
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        current_page += f"\n- Summary: {self.page_obs.page_summary}"
        if self.page_obs.task_info:
            current_page += f"\n- Relevant info: {self.indent_str(self.page_obs.task_info)}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n"
        user_prompt += "FORM:\n{context}"

        if (len(page_section.task_elems)==1) and (len(elements)>10) and (not dropdown) and (page_section.task_elems[0].tag in ['input', 'textarea', 'select']):
            chosen_elements = page_section.task_elems
        else:
            chosen_elements = self.llm_choose_list(input_elements, inputs_str_list, sys_prompt, user_prompt, '**EDIT FIELDS**:', inline=True, trunc=True)
        if not chosen_elements:
            if form_loc:
                self.page_obs.html_sections.append(page_section)
                return
            elif not dropdown:
                chosen_elements = page_section.task_elems

        no_op = 0
        for i in range(len(chosen_elements)):
            elem = chosen_elements[i]
            if (elem.input_type=='radio') and (elem.input_value=='checked'):
                self.element_action(elem, page_section)
                no_op += 1
                continue
            elem_loc = self.get_elem_locator(elem, page_section)
            if elem_loc.count():
                if (elem.tag not in ['input', 'textarea', 'select', 'label']) and (elem.role!='combobox'):
                    if (not elem.aria_expanded) or (elem_loc.get_attribute('aria-expanded') != 'true'):
                        self.field_action(elem, page_section)
                    else:
                        logger.debug(f"{elem.get_name()} already expanded")
                    break
                self.field_action(elem, page_section)
            else:
                logger.warning(f"{elem.get_name()} no longer available")
            if (self.env.page.url != start_url) and (self.env.page.url[:-1] != start_url):
                break

        # Check for missing required input fields
        empty_fields = []
        for elem in input_elements:
            if (not elem.clicked) and (elem.input_type!='radio') and (not elem.input_value):
                empty_fields.append(elem)
        if empty_fields: logger.debug(f"Check remaining fields")
        for elem in empty_fields:
            if (elem.required) and (not self.get_input_value(elem, page_section)):
                self.element_action(elem, page_section)


        if ((no_op) == len(chosen_elements)) or (page_section.is_dialog):
            return


        # Check if form filling complete
        prev_action_result = None
        done = False
        i = 0
        while ((self.env.page.url == start_url) or (self.env.page.url[:-1]==start_url)) and (not done) and (i < max_actions):
            page_section, elem_diff = self.update_section(page_section, inputs_only=inputs_only)
            if not page_section:
                break
            grouped_elems = self.sort_elements(page_section.elements, form_elements, elem_diff['added'], dropdown, form_loc)
            if (len(page_section.task_elems)==1) and (not dropdown) and (not grouped_elems['submit_elems']):
                logger.info(f"Done editing form field\n")
                return
            input_elems = grouped_elems['input_elems']
            action_elems = grouped_elems['action_elems']
            if (dropdown) and ((n_inp_elems-len(input_elems)) > 5):
                logger.debug(f"input_elems: {n_inp_elems} to {len(input_elems)}, break")
                break
            # Maybe choose form buttons/new fields only
            chosen_elem = self.form_check(input_elems+action_elems, grouped_elems, prev_action_result)
            if action_elems:
                self.page_obs.filled_form = True
            if not chosen_elem:
                page_section.done_form = True
                break
            self.field_action(chosen_elem, page_section)
            i += 1
            if chosen_elem in grouped_elems['submit_elems']:
                break
            if (chosen_elem.tag=='button') or (chosen_elem.role=='button'):
                prev_action_result = self.screen_diff(action_str=self.prev_action)

        if ('http://localhost:9999' in self.env.page.url) and (self.page_title() == "500 Internal Server Error"):
            logger.debug(f"Redirect to profile page to address Reddit env submission error")
            self.env.page.goto('http://localhost:9999/user/MarvelsGrantMan136')

        logger.info(f"Done submitting form\n")
        self.form_plan = ""

        return



    
    # Table Section
    def get_table_sort(self, table_loc: Locator, column_labels: list[str]=[], sorted_col: str=None) -> str:
        """"""

        logger.debug(f"Check table sort order")

        table_str = self.table_info(table_loc, column_labels, max_rows=10, add_row_count=False)
        user_prompt = self.prompts.table_sort_order.format(table_info=table_str)
        if sorted_col:
            user_prompt = f'I recently performed an action to sort a table by the "{sorted_col}" column. {user_prompt}'

        if self.manual:
            check_sort = input("Type Y to check sort:")
            if not check_sort: return
        messages = self.format_llm_prompt(system_prompt=None, user_prompt=user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=False)
        answer = self.extract_llm_answer(llm_out, keyword='ANSWER:')

        if answer.startswith('None'):
            sort_order = f"The table rows are not sorted currently."
        elif (len(answer) > 150):
            sort_order = ""
        else:
            sort_order = f"{answer}"

        return sort_order


    def table_headers(self, table: PageSection, table_loc: Locator) -> tuple[list[str], str]:
        """Get list of table header labels and table caption. (Optionally sort/filter columns if available)"""

        if not table_loc.count():
            logger.error(f"Locator error for table, count={table_loc.count()}\n{table_loc}")
            return None, None
        thead = table_loc.locator('xpath=descendant::thead')
        if not thead.count():
            return None, None

        # Get all column header labels
        column_labels = []
        sortable_cols = []
        headers = thead.locator('th').all()

        for h in headers:
            if (h.locator('xpath=descendant::input').count()) or (h.locator('xpath=descendant::select').count()):
                label = 'Select'
            else:
                label = h.inner_text().strip()
            column_labels.append(label)
            # check for sortable columns
            class_name = h.get_attribute('class')
            sort = h.get_attribute('sortable')
            if ('sort' in str(class_name).lower()) or (sort != None):
                sortable_cols.append(label)

        n_rows = table_loc.locator('tbody > tr').filter(visible=True).count()

        sorted_order = f""

        if not table.table_filters:
            for input_loc in thead.locator("input, select").filter(visible=True).all():
                input_elem = self.elem_from_loc(input_loc, section=table, iframe_id=table.iframe_id)
                if (input_elem.input_type=='checkbox') and (not input_elem.label):
                    input_elem.label = f'Select all table rows for bulk action'
                input_elem.func_desc = f'Table filter'
                input_elem.section = table
                table.table_filters.append(input_elem)
        if table.table_filters:
            self.page_obs.filter_table = True
            table.elements += table.table_filters

        return column_labels, sorted_order


    def table_subgoal(self, table_section: PageSection) -> str:
        """"""

        if self.manual: return ""
        logger.success(f"Table subgoal")

        table_summary = table_section.desc
        
        system_prompt = self.prompts.table_subgoal
        user_prompt = f"""\
Your task request and the details about the table on the current page are given below.

TASK: {self.intent}

PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})

TABLE: {table_summary}

Now, please reason about the task requirements and then describe the entries to find.
"""
        messages = self.format_llm_prompt(system_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages)
        goal = self.extract_llm_answer(llm_out, keyword="**GOAL**:", line_only=False)

        return goal


    def table_row_info(
        self, 
        row_loc: Locator, 
        col_headers: list[str]=[],
        prev_row_entries = [],
        no_action = False,
    ) -> tuple[list, str]:
        """Return list of table row column values and a string with values formatted by column for LLM."""

        if row_loc.count() != 1:
            logger.warning(f"Locator error for table row, count={row_loc.count()}\n{row_loc}")
            return None
        
        remove_links = False
        if row_loc.locator('a').count() > 10:
            remove_links = True

        row_str = ''
        values: list[dict[str, Any]] = []  # {"value": str, "rowspan": int}

        all_td_locs = row_loc.locator('td').all()
        for i in range(len(all_td_locs)):
            value_loc = all_td_locs[i]
            value_html = value_loc.inner_html()
            value_str = process_md(md(value_html, escape_misc=False), self.env.page.url, remove_links)
            value_str = clean_text(value_str, max_lines=None)
            rowspan = value_loc.get_attribute('rowspan')
            if rowspan:
                value_rowspan = int(rowspan)
            else:
                value_rowspan = 1
            values.append({"value": value_str, "rowspan": value_rowspan})
        
        for i in range(len(prev_row_entries)):
            prev_value = prev_row_entries[i]
            if (prev_value['rowspan']) and (prev_value['rowspan'] >= 2):
                values.insert(i, {"value": prev_value["value"], "rowspan": prev_value["rowspan"]-1})

        
        # check for row header
        row_label = None
        if row_loc.locator('th').filter(visible=True).count():
            th_html = row_loc.inner_html()
            row_label = process_md(md(th_html, escape_misc=False))
        if row_label:
            row_str = f'**{row_label}**: '
        
        # Format row values with column labels
        labeled_values = []

        if not col_headers: col_headers = []
        if not values: values = []
        if len(col_headers) != len(values):
            col_headers = []
        
        for i in range(len(values)):
            value = values[i]["value"]
            if (value == ''):
                continue
            attr_str = ''
            if col_headers:
                attr_str = f'{col_headers[i]} = '
                if no_action and ('Action' in col_headers[i]):
                    continue
            attr_str += f'"{value}"'
            labeled_values.append(attr_str)
        
        values_str = ', '.join(labeled_values)
        row_str += values_str
        row_str = get_sim_url(row_str)

        return values, row_str
    

    def table_info(
        self, 
        table_loc: Locator, 
        col_headers: list[str]=[], 
        start_i: int=None,
        max_rows: int=None,
        subset: list[int]=[],
        add_indices: bool=False,
        add_row_count: bool=True,
        get_end: bool=False
    ) -> str:
        """Get values of first max_rows table rows and format with column headers into a markdown string."""

        if table_loc.count() > 1:
            max_loc = None
            max_h = 0
            for loc in table_loc.all():
                logger.debug(f"{loc.bounding_box()['height']}")
                if loc.bounding_box()['height'] > max_h:
                    max_loc = loc
            table_loc = max_loc
        if table_loc.count() == 0:
            logger.warning(f"Locator error for table, count={table_loc.count()}\n{table_loc}")
            return None
        if col_headers == None:
            col_headers = []

        table_str = f""
        if col_headers:
            if add_indices:
                col_headers = [""] + col_headers
            table_str = "| " + " | ".join(col_headers) + " |\n"
            table_str += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"

        all_row_locs = table_loc.locator('tbody > tr').filter(visible=True).all()
        if not all_row_locs:
            logger.warning(f"No table rows")
        end_i = len(all_row_locs)
        if not start_i:
            start_i = 0
        if max_rows:
            end_i = min(end_i, (start_i+max_rows))
        else:
            max_rows = end_i
        if get_end:
            all_row_locs.reverse()

        prev_row_entries = []
        for i in range(start_i, end_i):
            if (subset) and (i not in subset):
                continue
            row_loc = all_row_locs[i]
            row_values, row_str = self.table_row_info(row_loc, col_headers, prev_row_entries)
            value_str_list = [val['value'] for val in row_values]
            if add_indices:
                value_str_list = [f'{i+1}'] + value_str_list
            table_str += "| " + " | ".join(value_str_list) + " |\n"
            prev_row_entries = row_values  # pass to next row to handle multi-row entries

        if (add_row_count) and (max_rows < len(all_row_locs)) and (len(all_row_locs) > start_i+max_rows):
            table_str += f"...\n{len(all_row_locs) - (start_i+max_rows)} more rows"

        return table_str.strip()



    def check_table_info(self, table_loc: Locator, column_labels: list[str]=[]) -> bool:
        """"""

        if self.manual:
            return False
        logger.debug(f"Check table info")

        need_more_info = False

        system_prompt = self.prompts.check_table_info
        table_str = self.table_info(table_loc, column_labels, max_rows=10)
        user_prompt = f"Your objective and the table information are provided below.\n\n"
        user_prompt += f"OBJECTIVE: {self.current_subgoal}\n\nTABLE:\n{table_str}\n\n"
        user_prompt += f"Determine if all details required for your objective are provided in the table columns or if viewing more details for the entries is required."
        
        messages = self.format_llm_prompt(system_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages)
        answer = self.extract_llm_answer(llm_out, keyword='**ANSWER**:')
        if answer.startswith('Missing details'):
            need_more_info = True

        return need_more_info




    def select_table_rows(
        self,
        list_section: PageSection,
        section_loc: Locator,
        start_i: int=0,
        max_rows: int=20, 
        col_headers: list[str]=[],
        subset: list[int]=[],
        doublecheck: bool=True
    ) -> list[int]:
        """"""

        logger.success(f"Select table rows")

        sys_prompt = self.prompts.select_table_rows
        use_short_prompt = False
        if (not list_section.reselect_items) and ((max_rows>=50) or (self.time_step>10)):
            sys_prompt = self.prompts.select_rows_short
            use_short_prompt = True

        objective = f'TASK: {self.intent}'
        if not use_short_prompt:
            objective += f'\n\nHISTORY:\n{self.episode_history()}'
        page = f'PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})'

        list_info = f'LIST INFO: {list_section.summary} {list_section.sort_order}'
        if (list_section.list_item_tag == 'details'):
            list_info = f'LIST INFO: Table of contents. Select a section header to view full details.'

        if (list_section.list_type == 'table') or (list_section.type=='table'):
            items_str = self.table_info(section_loc, col_headers, start_i, max_rows, subset=subset, add_indices=True)
        else:
            items_str = self.list_info(list_section, start_i, max_rows, subset=subset, add_indices=True, caption_img=True)
        list_items = f'LIST ITEMS:\n{items_str}'
        user_prompt = f'{objective}\n\n{page}\n\n{list_info}\n\n{list_items}'
        user_prompt += f'\n\nPlease identify any list items that are relevant to the task and provide their indices.'

        max_rows = min(list_section.n_items, max_rows)
        if subset:
            all_indices = subset
        else:
            all_indices = range(start_i, start_i+max_rows)

        if (not self.cfg.filter_page_info):
            list_section.section_text = items_str
            return all_indices
        if (0 < list_section.n_items < 5):
            return all_indices
        if self.manual:
            choices_str = input(f"{user_prompt}\n\nChoose items: ")
        else:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            choices_str = self.extract_llm_answer(llm_out, keyword='**SELECT ITEMS**:')

        if choices_str.startswith('None'):
            return []
        elif choices_str.startswith('All'):
            return all_indices
        else:
            choices_str_list = choices_str.split(',')
        chosen_indices = []
        for choice in choices_str_list:
            try:
                index = int(choice.strip()) - 1
                chosen_indices.append(index)
            except:
                logger.warning(f"Invalid index: {choice}")
                continue
        
        # Double-check other rows
        if (doublecheck) and (len(chosen_indices)>=(max_rows/2)) and (max_rows <= 25) and (not subset):
            unchosen = [i for i in all_indices if i not in chosen_indices]
            if (len(unchosen)>2):
                logger.debug(f"Double-check remaining rows")
                chosen_indices += self.select_table_rows(
                    list_section, 
                    section_loc, 
                    start_i=start_i,
                    max_rows=max_rows,
                    col_headers=col_headers, 
                    subset=unchosen
                )
        logger.debug(f"Choices: {chosen_indices}")

        return chosen_indices


    def done_table_iter(
        self, 
        table_summary: str, 
        sort_order: str, 
        table_loc: Locator,
        matches: list[dict],
        last_n: int, 
        col_headers: list[str]=[],
        list_type='table',
        all_match = False,
        n_rows: int=None
    ) -> bool:
        """"""

        logger.success(f"Check if table iteration is finished.")
        if self.manual:
            if input("type to break:"):
                return True
            else:
                return False
        if (sort_order) and (len(matches)==1):
            if matches[0]['n'] == last_n:
                return False

        objective = f"OBJECTIVE: {self.intent}"
        table_info = f"TABLE INFO: {table_summary} {sort_order}".strip()

        rows_checked = f"ROWS CHECKED:"
        
        start_i = 0
        for i in range(len(matches)):
            match_str = matches[i]["item_str"]
            match_index = matches[i]['n']
            for n in range(start_i, match_index):
                rows_checked += f"\n{n+1}. (Doesn't match)"
            rows_checked += f"\n{match_index+1}. (Match) {match_str}"
            start_i = match_index + 1
        for i in range(start_i, last_n+1):
            if list_type=='table':
                row_loc = table_loc.locator('tbody > tr').nth(i)
                row_dict, row_str = self.table_row_info(row_loc, col_headers, no_action=True)
            else:
                item_loc = table_loc.locator('xpath=child::li').all()
            if all_match:
                rows_checked += f"\n{i+1}. (Match)"
            elif last_n-i <= 2:
                rows_checked += f"\n{i+1}. (Doesn't match)"
            else:
                rows_checked += f"\n{i+1}. (Doesn't match)"
        
        if (n_rows) and (last_n < n_rows):
            rows_checked += f"\n\n... {(n_rows - last_n)} more rows below"

        system_prompt = self.prompts.done_table_iter
        user_prompt = f"Your objective, the table information, and the list of table rows checked so far are given below.\n\n"
        user_prompt += f"{objective}\n\n{table_info}\n\n{rows_checked}\n\n"
        user_prompt += f"Determine if the entries specified by the objective have been found, or if we need to search the rest of the table."

        messages = self.format_llm_prompt(system_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages)
        answer = self.extract_llm_answer(llm_out, keyword='**COMPLETE**:', backups=['**COMPLETE:**'])
        if (answer.startswith('Yes')):
            return True
        elif (not answer.startswith('No')):
            logger.warning(f"Invalid LLM answer: '{answer}'")
        
        return False


    def table_iter(self, table_section: PageSection, table_loc: Locator, max_iter=200):
        """Iterate table rows and choose rows relevant to task, then perform actions on selected rows."""

        logger.success(f"Iterate table rows")
        start_url = self.page_obs.url
        table_section.list_items_url = start_url
        table_page = self.page_obs
        col_labels = table_section.column_labels

        need_more_info = False
        if (table_loc.locator('tbody > tr').locator('a').count()) and (not self.page_obs.similar):
            self.current_subgoal = self.table_subgoal(table_section)
            table_section.section_plan = self.current_subgoal
            need_more_info = self.check_table_info(table_loc, col_labels)
            self.page_obs.table_subgoal = self.current_subgoal
            self.page_obs.table_need_info = need_more_info
        elif (self.prev_states) and (self.page_obs.similar):
            prev_state = self.prev_states[-1]
            prev_page = prev_state['observation']['page_mem']
            need_more_info = prev_page.table_need_info

        all_row_locs = table_loc.locator('tbody > tr').filter(visible=True).all()
        n_rows = len(all_row_locs)
        table_section.n_items = n_rows
        logger.debug(f"{n_rows} table rows on page")
        if max_iter == None:
            max_iter = n_rows
        if n_rows > 1000:
            return []
        n_rows = min(n_rows, max_iter)
        sort_order = table_section.sort_order
        if (not sort_order) and (n_rows >= 10):
            sort_order = self.get_table_sort(table_loc, col_labels)
            table_section.sort_order = sort_order

        candidates = []
        prev_row_entries = []
        complete = False
        start_i = 0
        chunk_size = self.cfg.table_chunk_size
        chunk_n = 0
        all_chosen_indices = []
        check_all = False

        while (start_i < n_rows) and (not complete):
            if (n_rows - start_i) < 10:
                chosen_indices = range(start_i, n_rows)
            else:
                chosen_indices = self.select_table_rows(table_section, table_loc, start_i, min(chunk_size, n_rows), col_labels)
            if chosen_indices:
                chunk_n += 1
                all_chosen_indices += chosen_indices

            for i in range(start_i, min(start_i+chunk_size, n_rows)):
                if i not in chosen_indices:
                    continue
                row_loc = all_row_locs[i]
                row_values, row_str = self.table_row_info(row_loc, col_labels, prev_row_entries)
                details = None
                logger.info(f"Table row {i}:")
                
                checkbox_loc = row_loc.locator('input[type="checkbox"]').filter(visible=True)
                if (checkbox_loc.count()==1):
                    checkbox = self.elem_from_loc(checkbox_loc, iframe_id=table_section.iframe_id)
                    self.set_checkbox(checkbox, table_section, check=True)
                
                row_dict = {
                    "n": i,
                    "row_loc": row_loc,
                    "row_values": row_values,
                    "item_str": row_str,
                    "details": details
                }
                candidates.append(row_dict)
                prev_row_entries = row_values  # pass to next row to handle multi-row entries
            
            start_i += chunk_size
            if (start_i > 60) and (len(candidates) == 1) and (row_dict['n']<start_i+40): break
            if (start_i < n_rows-1) and (not complete) and (candidates) and (not check_all):
                complete = self.done_table_iter(table_section.summary, sort_order, table_loc, candidates, i, col_labels, n_rows=n_rows)
                if (candidates) and (not complete):
                    check_all = True
            if complete: break


        if not candidates:
            logger.warning(f"No matching rows found")
            table_section.relevant_items_str = f"No relevant items found in table"
        elif n_rows == 1:
            matches_str = f"List contains a single item:"
            row_str = candidates[0]["item_str"]
            matches_str += f"\n1. {row_str}"
            table_section.relevant_items_str = matches_str
        elif (len(candidates) >= 50):
            matches_str = self.table_info(table_loc, col_labels, 0, n_rows, subset=all_chosen_indices, add_indices=True)
            table_section.relevant_items_str = matches_str
        else:
            if (need_more_info) and (len(candidates)>1):
                matches_str = f"The table contains {len(candidates)} items that are potentially relevant to the task but we need to view the details of each item to confirm:"
            elif (len(candidates) == n_rows):
                matches_str = f"All {n_rows} table items:"
            else:
                matches_str  = f"I checked the first {n_rows} items visible on the current page and found {len(candidates)} relevant ones:"

            for candidate in candidates:
                row_n = candidate["n"]
                row_str = candidate["item_str"]
                matches_str += f"\n   {row_n+1}. {row_str}"
            table_section.relevant_items_str = matches_str
            self.page_obs.list_info = matches_str

        logger.success(f"Finished table iteration")

        return candidates
    

    def all_row_actions(self, rows: list[dict], table_section: PageSection) -> list[Element]:
        """"""

        prev_elems = []
        for elem in table_section.elements:
            if (not elem.table_row) and (not elem.row_info):
                prev_elems.append(elem)
        table_section.elements = prev_elems

        elements = []
        for i in range(len(rows)):
            row = rows[i]
            row_loc = row['row_loc']
            row_str = row['item_str']
            row_n = row['n']
            
            row_loc.hover(force=True)
            if 'clickable' in str(row_loc.get_attribute('class')):
                row_elem = self.elem_from_loc(row_loc, trow=row_n)
                elements.append(row_elem)
                row_elem.row_info = f"{row_str}"
                row_elem.print_element()

            action_locs = row_loc.locator("a, button, input").filter(visible=True).all()
            for loc in action_locs:
                elem = self.elem_from_loc(loc, trow=row_n)
                if elem:
                    if len(rows) > 1:
                        elem.row_info = f"for list item {row_n+1}"
                    elements.append(elem)

        return elements


    def table_action(self, table_section: PageSection):
        """"""

        logger.success(f"Table action")
        start_url = self.page_obs.url

        table_loc = self.get_section_locator(table_section)
        if table_section.type != 'table':
            table_loc = table_loc.locator('table').filter(visible=True)
        if table_loc.count() > 1:
            max_loc = None
            max_h = 0
            for loc in table_loc.all():
                logger.debug(f"multi table_loc {loc.bounding_box()['height']}")
                if loc.bounding_box()['height'] > max_h:
                    max_loc = loc
                    max_h = loc.bounding_box()['height']
            table_loc = max_loc
        if table_loc.count() != 1:
            logger.error(f"table_loc: {table_loc.count()}\n{table_loc}")
            return
        if ((table_loc.locator('table').count()) and (table_loc.bounding_box()['height']<2000)) or (not self.cfg.filter_page_info):
            self.update_section(table_section)
            table_section.section_text = self.section_content_md(table_loc)['text']
            return

        # get column header labels and optionally change sort
        if (not table_section.column_labels):
            column_labels, sorted_order = self.table_headers(table_section, table_loc)
            table_section.column_labels = column_labels
            table_section.sort_order = sorted_order
        else:
            column_labels = table_section.column_labels

        needs_update = False
        if (self.table_info(table_loc, column_labels, max_rows=25)!=table_section.section_text) or (table_section.list_items_url!=self.env.page.url):
            needs_update = True
            table_section.inner_html = table_loc.first.inner_html()
            table_section.section_text = self.table_info(table_loc, column_labels, max_rows=25)

        # identify relevant table rows
        matching_rows = table_section.list_items
        if (len(matching_rows) == 1) or (needs_update):
            table_section.reselect_items = True if (not needs_update) else False
            matching_rows = self.table_iter(table_section, table_loc)
            table_section.list_items = matching_rows
            row_actions = self.all_row_actions(matching_rows, table_section)
            table_section.elements += row_actions

        return


    # List Section
    def list_info(
        self, 
        list_section: PageSection, 
        start_i: int=None, 
        n_items: int=10, 
        subset: list[int]=[],
        add_indices: bool=False, 
        add_remaining: bool=True, 
        get_end: bool=False, 
        caption_img=False
    ) -> str:
        """Get info for first n list items."""

        section_loc = self.get_section_locator(list_section)
        if not section_loc.count():
            logger.warning(f"Section not present: {list_section.class_name}\n{section_loc}")
            return ""
        
        if list_section.type == 'list':
            all_item_locs = section_loc.all()
        elif list_section.li_class:
            all_item_locs = section_loc.locator(f'li.{list_section.li_class}').filter(visible=True).all()
        else:
            all_item_locs = section_loc.locator('xpath=child::li').all()
        list_section.n_items = len(all_item_locs)

        row_length = 1
        if list_section.list_type == 'grid':
            row_length = self.grid_row_length(all_item_locs)
            list_section.row_length = row_length

        if not all_item_locs:
            logger.warning(f"No list items")
        end_i = len(all_item_locs)
        if not start_i:
            start_i = 0
        if n_items:
            end_i = min(end_i, (start_i+n_items))
        else:
            n_items = end_i
        if get_end:
            all_item_locs.reverse()

        img_query = self.prompts.desc_img_med

        list_str = f""
        for i in range(start_i, end_i):
            if (subset) and (i not in subset):
                continue
            item_loc = all_item_locs[i]
            remove_nested = None
            if (list_section.type == 'list') and (item_loc.get_attribute('data-level')):
                remove_nested = list_section.list_item_tag
            info_dict = self.section_content_md(
                item_loc, 
                section=list_section,
                img_query=img_query,
                remove_nested=remove_nested, 
                caption_img=caption_img,
                escape_misc=False,
                add_quotes=False,
                scout=True
            )
            info_str = info_dict['text']

            # temp
            images_dict = info_dict['images']
            if images_dict:
                image_captions = {}
                for k, v in info_dict['images'].items():
                    image_captions[k] = v['desc']
                list_section.image_captions.update(image_captions)
            
            if (self.cfg.benchmark != 'visualwebarena') or (not info_dict['images']):
                info_dict_aria = self.section_content(
                    item_loc, 
                    section=list_section,
                    img_query=img_query,
                    remove_nested=remove_nested, 
                    caption_img=caption_img,
                    escape_misc=False,
                    add_quotes=False,
                    scout=True
                )
                aria_str = info_dict_aria['text']
                if (len(aria_str)<len(info_str)) or ((len(info_str.split('\n\n'))>25) and (len(info_str)<1000)):
                    info_dict = info_dict_aria
                    info_str = aria_str
            
            if (list_section.list_item_tag == 'details') and (add_indices):
                info_str = item_loc.locator('summary').first.inner_text().strip()

            list_section.list_items_dict[i] = info_dict
            list_str += f'{i+1}.'
            if (list_section.list_type == 'grid') and (row_length>1):
                row = (i // row_length) + 1
                col = (i % row_length) + 1
                list_str += f' (row {row}, column {col})'
            list_str += f' """\n{info_str}\n"""\n\n'

        if (add_remaining) and ((start_i+n_items) < len(all_item_locs)):
            list_str += f"... {len(all_item_locs) - (start_i+n_items)} more"

        return list_str.strip()


    def get_list_sort(self, list_section: PageSection, vlm=True) -> str:
        """Return a sentence describing the list sort order."""

        if list_section.list_item_tag == 'img':
            return ""
        if list_section.sort_order and (list_section.section_text==self.list_info(list_section)):
            return list_section.sort_order

        # LLM
        sys_prompt = self.prompts.get_list_sort
        user_prompt = f"PAGE: {get_sim_url(self.env.page.url)}"
        list_items = self.list_info(list_section)
        list_section.section_text = list_items
        if (not list_items) or (list_section.n_items<10):
            return ""
        user_prompt += f"\n\n{list_items}\n\n"
        user_prompt += f'Please analyze this information and check if the list items are sorted \
from first to last (ie. if the list item values decrease from 1 to 5, then they are sorted in descending order). \
If there is no clear sort applied to the items then give "None" as your answer instead.'
        
        logger.debug(f"Check list sort order")
        if self.manual:
            check_sort = input("Type Y to check sort:")
            if not check_sort: return
        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=False)
        sort_order = self.extract_llm_answer(llm_out, keyword='**SORT ORDER**:')

        if sort_order.startswith('None') and vlm:
            screenshot = self.env.get_screenshot(max_height=720)
            vlm_prompt = self.prompts.vlm_list_sort
            vlm_out = self.model_manager.vlm_call(screenshot, vlm_prompt)
            if vlm_out:
                sort_order = vlm_out
        if 'None' in sort_order:
            sort_order = "The list items are currently not sorted."

        return sort_order
    

    def list_item_structure(self, list_section: PageSection):
        """"""

        if (list_section.list_item_type) or (self.manual):
            return
        if (list_section.n_items < 10):
            list_section.list_item_type = "list item"
            return
        logger.debug(f"Get list structure")
        
        user_prompt = self.prompts.item_structure
        list_items = self.list_info(list_section, n_items=20, add_remaining=False)
        user_prompt += f"\n\nLIST ITEMS:\n{list_items}"
        
        messages = self.format_llm_prompt(user_prompt, thinking=False)
        llm_out = self.model_manager.llm_call(messages, show_prompt=False)
        data_class_str = self.extract_llm_answer(llm_out, 'DATA CLASS:', line_only=False)
        class_name, attr_list = parse_class_string(data_class_str)
        if (not data_class_str) or (data_class_str.startswith('None')) or ('optional' in data_class_str.lower()) or (len(attr_list)>10):
            list_section.list_item_type = "list item"
            return
        if class_name.lower().endswith('entry') and (len(class_name)>5):
            class_name = class_name[:-5]
        list_section.list_item_type = class_name
        list_section.list_item_attributes = attr_list

        self.relevant_list_attr(list_section, data_class_str)

        return
    

    def relevant_list_attr(self, list_section: PageSection, class_str: str):
        """"""

        user_prompt = self.prompts.item_attributes
        user_prompt += f"\n\nQUERY: '{self.intent}'"
        user_prompt += f"\n\n{class_str}"

        messages = self.format_llm_prompt(user_prompt=user_prompt, thinking=False)
        llm_out = self.model_manager.llm_call(messages, show_prompt=False)
        atts = self.extract_llm_answer(llm_out, 'ATTRIBUTES:', line_only=False)
        if not atts:
            return
        list_section.relevant_attributes = [a.strip() for a in atts.split(',')]

        return


    def extract_item_info(self, list_section: PageSection, candidates: list[dict]) -> str:
        """"""

        logger.debug(f"Extract item attributes")
        user_prompt = self.prompts.extract_list_info
        attributes = ', '.join(list_section.relevant_attributes)
        user_prompt = user_prompt.format(attr_list=attributes)

        list_items = f""
        for i in range(len(candidates)):
            nth = candidates[i]["n"]
            row_str = candidates[i]["item_str"]
            list_items += f'{nth+1}.'
            row_length = list_section.row_length
            if (list_section.list_type == 'grid') and (row_length>1):
                row = (nth // row_length) + 1
                col = (nth % row_length) + 1
                list_items += f' (row {row}, column {col})'
            list_items += f' """\n{row_str}\n"""\n\n'
        list_items = list_items.strip()
        if (not list_section.relevant_attributes) or (len(list_section.relevant_attributes)>1) or (len(candidates)==1):
            return list_items
        user_prompt += f"\n\nLIST ITEMS:\n{list_items}"

        messages = self.format_llm_prompt(user_prompt=user_prompt)
        llm_out = self.model_manager.llm_call(messages)
        item_values = self.extract_llm_answer(llm_out, 'ITEM VALUES:', line_only=False)
        if (not item_values) or ('None' in item_values):
            return list_items
        if len(list_section.relevant_attributes) == 1:
            item_values = f"{attributes} values for the relevant {list_section.list_item_type.lower()}s:\n{item_values}"
        
        list_section.extracted_attr_info = item_values
        self.saved_info["saved_links"].append(item_values)

        return item_values


    def list_iter_page(self, list_section: PageSection, max_iter=200, one_page=False):
        """Iterate over list items on page and return items that match subgoal."""

        start_url = self.page_obs.url
        list_section.list_items_url = start_url
        section_loc = self.get_section_locator(list_section)
        if not section_loc.count():
            logger.error(f"Section locator error, {section_loc.count()}")
            return []

        if list_section.type == 'list':
            all_item_locs = section_loc.all()
        elif list_section.li_class:
            logger.debug(f"List item class: {list_section.li_class}")
            all_item_locs = section_loc.locator(f'li.{list_section.li_class}').filter(visible=True).all()
        else:
            all_item_locs = section_loc.locator('xpath=child::li').filter(visible=True).all()
        n_items = len(all_item_locs)
        list_section.n_items = n_items
        n_items = min(n_items, max_iter)
        logger.info(f"{n_items} list items")


        candidates = []
        complete = False
        start_i = 0
        chunk_size = self.cfg.list_chunk_size
        chunk_n = 0
        all_chosen_indices = []
        check_all = False

        while (start_i < n_items) and (not complete):
            chosen_indices = self.select_table_rows(list_section, section_loc, start_i, chunk_size)
            all_chosen_indices += chosen_indices
            if chosen_indices:
                chunk_n += 1

            for i in range(start_i, min(start_i+chunk_size, n_items)):
                item_loc = all_item_locs[i]
                if i not in chosen_indices:
                    continue
                logger.info(f"List item {i}:")
                
                remove_nested = None
                if (list_section.type == 'list') and (item_loc.get_attribute('data-level')):
                    remove_nested = list_section.list_item_tag
                info_dict = list_section.list_items_dict[i]
                if info_dict.get('has_chart'):
                    info_dict = self.section_content_md(item_loc, section=list_section, remove_nested=remove_nested, escape_misc=False)
                if info_dict.get('images'):
                    list_section.images += info_dict['images']
                    for img_url, img_dict in info_dict['images'].items():
                        self.save_img(image=img_dict['image'], image_path=img_url, image_desc=img_dict.get('desc'))

                summary = info_dict['text']
                item_dict = {
                    "page_url": start_url,
                    "n": i,
                    "item_loc": item_loc,
                    "info_dict": info_dict,
                    "item_str": summary,
                }
                candidates.append(item_dict)

            start_i += chunk_size
            if (start_i > 60) and (len(candidates) == 1): break
            if (start_i < n_items-1) and ((n_items>=50) or self.manual) and (not complete) and (candidates) and (not check_all):
                complete = self.done_table_iter(list_section.summary, list_section.sort_order, section_loc, candidates, i, list_type='list', n_rows=n_items)
                if (candidates) and (not complete):
                    check_all = True
                if complete: break
        

        if not candidates:
            logger.warning(f"No matching rows found")
            matches_str = f"No relevant items found in list"
            list_section.relevant_items_str = matches_str
            self.page_obs.list_info = matches_str
        elif n_items == 1:
            matches_str = f"- List contains a single item:"
            item_str = candidates[0]["item_str"]
            matches_str += f"\n  1. <li>\n{item_str}\n</li>"
            list_section.relevant_items_str = matches_str
            self.page_obs.list_info = matches_str
        else:
            list_items_info = self.extract_item_info(list_section, candidates)
            items_type = f"{list_section.list_item_type.lower()}s"
            if len(candidates) == n_items:
                matches_str = f"All {n_items} list items:"
            elif len(candidates) == 1:
                matches_str = f"I checked all {n_items} {items_type} on the page and this one was the closest match:"
            else:
                matches_str = f"I checked all {n_items} {items_type} on the page and found {len(candidates)} that are potentially relevant:"
                if (len(candidates)>1) and (list_section.extracted_attr_info):  # or (list_section.images):
                    matches_str = f"- Found {len(candidates)} {items_type} on the page that are relevant to the task."
            matches_str += f"\n{self.indent_str(list_items_info, indent_first=True)}"
            if len(candidates) < n_items:
                matches_str += f"\n\nThe other {n_items-len(candidates)} {items_type} are not relevant to the task."
            list_section.relevant_items_str = matches_str
            self.page_obs.list_info = matches_str
        list_section.relevant_item_indices = all_chosen_indices

        return candidates
    

    def list_action(self, list_section: PageSection, max_iter: int=200, max_pages: int=1):
        """"""

        section_loc = self.get_section_locator(list_section)
        if not section_loc.count():
            logger.error(f"section_loc error, {section_loc.count()}")
            return
        logger.success(f"Iterate list items")
        start_url = self.page_obs.url

        list_section_tag = None
        list_section_class = None
        if list_section.type != 'list':
            list_section_tag = list_section.type
            list_section_class = list_section.class_name
            if section_loc.locator('li > ul').count():
                logger.debug(f"Get nested list items")
                item_loc = section_loc.locator('li > ul > li').nth(0)
                item_class = item_loc.get_attribute('class')
                if item_class:
                    list_section.li_class = item_class.split(' ')[0]
        
        if list_section.list_item_tag == 'img':
            return

        needs_update = False
        if (list_section.inner_html!=section_loc.first.inner_html()) or (list_section.list_items_url!=self.env.page.url):
            list_section.inner_html = section_loc.first.inner_html()
            logger.debug(f"NOTE: input_modified={list_section.input_modified}")
            if (self.page_obs.name != "Homepage") and (not list_section.input_modified):
                needs_update = True
                list_section.sort_order = self.get_list_sort(list_section)
                self.list_item_structure(list_section)


        # Get relevant list items
        matching_items = list_section.list_items
        if (len(matching_items) <= 1) or (needs_update):
            list_section.reselect_items = True if (not needs_update) else False
            matching_items = self.list_iter_page(list_section, max_iter)
            list_section.list_items = matching_items
            if (not list_section.item_actions_template) and (len(list_section.elements)<10):
                list_section.item_actions_template = list_section.elements
        
            # Get elements for selected list items
            all_actions = []
            for i in range(len(matching_items)):
                item = matching_items[i]
                nth = item['n']
                item_info = item['info_dict']
                item_loc = item['item_loc']
                is_list = list_section.type=='list'
                item_section = self.create_section(item_loc, list_section.iframe_id, is_list=is_list)
                if not item_section:
                    logger.warning(f"Missing item_loc: {i}")
                    continue
                item_section.list_section_tag = list_section_tag
                item_section.list_section_class = list_section_class
                item_section.nth = nth
                item_section.summary = item['item_str']
                item_section.elements = copy.copy(list_section.item_actions_template)
                item_section, diff = self.update_section(item_section)
                if not item_section:
                    continue
                for elem in item_section.elements:
                    elem.section = item_section
                    if (self.cfg.filter_page_info):
                        elem.row_info = f"for {list_section.list_item_type.lower()} {nth+1}"
                    all_actions.append(elem)
            
            if all_actions:
                self.log_trajectory(scrolled=True)
            list_section.elements = all_actions
            if self.slow_mode: self.wait_user_input()

        return




    # Page
    def get_page_info(
        self, 
        page_mem: PageMem,
        summary=True,
        task_summary=True,
        list_info=False,
        full_details=False,
    ) -> str:
        """"""

        page_info = f"{page_mem.name} ({get_sim_url(page_mem.url)})"
        
        if summary:
            page_info += f"\n  - Summary: {page_mem.page_summary}"
        
        if self.cfg.filter_page_info:
            if (list_info) and (page_mem.list_info):
                page_info += f"\n  - {self.indent_str(page_mem.list_info)}"
            elif (full_details) and (page_mem.task_info):
                page_info += f"\n  - Relevant info:\n{self.indent_str(page_mem.task_info, 2, indent_first=True)}"
        

        if self.screen_change:
            page_info += f'\n  - {self.screen_change}'
        
        
        if self.clipboard:
            page_info += f'\n  - **Clipboard**: "{self.clipboard}"'


        if page_mem.alerts:
            page_info += f"\n  - **Alert message**:"
            for alert_text in page_mem.alerts:
                page_info += f' "{alert_text}"'
        
        if page_mem.dialog:
            page_info += f""  #TODO
        if page_mem.browser_dialog:
            page_info += f""  #TODO
        # (maybe) replace with new tab notification
        if page_mem.popup_info:
            page_info += f"\n  - Additionally, a popup window has opened displaying the following information: '{page_mem.popup_info}'"
        
        page_mem.alerts = []
        page_mem.popup_info = None
        page_mem.browser_dialog = None

        return page_info


    def page_sections_context(
        self, 
        section_list: list[PageSection]=[], 
        page_mem: PageMem=None, 
        summary_only=False,
        relevant_only=False,
        task_summary=True,
        task_details=False,
        add_title=True,
        indent=False
    ) -> str:
        """Format list of page section summaries for LLM to choose from."""

        if page_mem:
            section_list = page_mem.html_sections.copy()
            if page_mem.list_section:
                section_list.append(page_mem.list_section)

        section_str_list = []
        for i in range(len(section_list)):
            page_section = section_list[i]
            section_summary = ""
            if (page_section.task_summary) and (task_summary):
                section_summary = page_section.task_summary
            elif (page_section.summary) and (not relevant_only):
                section_summary = page_section.summary
            class_str = ""
            if page_section.class_name:
                class_str = clean_text(page_section.class_name, max_words=1, max_lines=1)
            if (not class_str) and (page_section.id):
                class_str = page_section.id
            
            if (task_details) and (not self.cfg.filter_page_info):
                if not page_section.section_text:
                    section_loc = self.get_section_locator(page_section)
                    page_section.section_text = self.section_content(section_loc, caption_img=False, add_quotes=False)['text']
                section_str = f'{page_section.type} class="{class_str}":'
                section_str += f'\n{self.indent_str(page_section.section_text, indent_first=True)}'
                section_str_list.append(section_str)
                continue
            if (summary_only) and (not section_summary):
                continue
            
            if (page_section.list_type in ['list', 'grid', 'table']) and (page_section.bbox.get_abs_px_height() > 150):
                if (self.page_obs.similar) and (not page_section.list_items) and (page_section.type=='ol') and (page_section.sort_order):
                    logger.debug(f"NOTE: similar={self.page_obs.similar}, list_items={len(page_section.list_items)}, sort_order={page_section.sort_order}")
                    section_str = f'{page_section.type} class="{class_str}": {page_section.sort_order}'
                elif page_section.extracted_attr_info:
                    section_str = f'{page_section.type} class="{class_str}": {page_section.summary}'
                    section_str +=  f'\n{self.indent_str(page_section.relevant_items_str, indent_first=True)}'
                elif page_section.task_summary:
                    section_str = f'{page_section.type} class="{class_str}": {page_section.summary} {section_summary}'
                    if (task_details) and (page_section.task_details):
                        section_str += f"\n{self.indent_str(page_section.task_details, indent_first=True)}"
                else:
                    section_str = f'{page_section.type} class="{class_str}": {page_section.summary}'
            else:
                section_str = f'{page_section.type} class="{class_str}": {section_summary}'
                if (task_details) and (page_section.task_details):
                    section_str += f"\n{self.indent_str(page_section.task_details, indent_first=True)}"
            section_str_list.append(section_str)

        if add_title:
            sections_context = "PAGE SECTIONS:\n"
        else:
            sections_context = ""
        for i in range(len(section_str_list)):
            section_str = section_str_list[i]
            list_item = f'{i+1}) {section_str}\n'
            if indent:
                list_item = f'	{list_item}'
            sections_context += list_item

        return sections_context.strip()
    

    def choose_top_sections(self, section_list: list[PageSection], done_obs=False):
        """"""

        if (len(section_list) == 1):
            return section_list
        logger.success(f"Identify most promising page sections")

        system_prompt = self.prompts.promising_sections
        task = f'TASK: {self.intent}'
        history = f'HISTORY:\n{self.episode_history()}'
        page = f'CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})'
        if done_obs:
            page += f'\n  - Summary: {self.page_obs.task_summary}'
        sections_context = self.page_sections_context(section_list, task_summary=False, indent=True)
        user_prompt = f'{task}\n\n{history}\n\n{page}\n\n{sections_context}'

        if self.manual:
            logger.info(f"\n{sections_context}")
            choices_str = input("choose sections: ")
        else:
            messages = self.format_llm_prompt(system_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            choices_str = self.extract_llm_answer(llm_out, keyword='SECTIONS**:', backups=['SECTIONS:**', 'SECTIONS:'])
        choices_str_list = choices_str.split(',')
        
        chosen_indices = []
        for choice in choices_str_list:
            try:
                index = int(choice.strip())
                chosen_indices.append(index-1)
            except:
                continue
        
        chosen_sections = []
        for i in range(len(section_list)):
            section = section_list[i]
            if i in chosen_indices:
                chosen_sections.append(section)
            elif (i>0) and (section.n_dropdowns>5):
                chosen_sections.append(section)
            elif ('main-actions' in str(section.class_name)):
                chosen_sections.append(section)
            elif (section.list_type in ['list', 'grid']) and (section.bbox.y1_abs_px < 960) and (not self.manual):
                chosen_sections.append(section)
            elif (section.bbox.get_abs_px_width()>475) and (section.bbox.get_abs_px_height()>250) and (section.bbox.y1_abs_px<720):
                chosen_sections.append(section)

        return chosen_sections


    def record_state_v1(self):
        """Add current system state to episode history."""

        system_state: SystemState = dict()
        system_state['nth'] = self.time_step
        system_state['url'] = self.env.page.url

        # Save current environment state
        system_state['observation'] = {
            "html": self.env.page.content(),
            "screenshot": self.env.get_screenshot(),
            "browser_tabs": self.browser_tabs,
            "page_mem": copy.copy(self.page_obs),
            "focused_elem": self.get_foc_elem(),
            "scroll_height": self.page_scroll_height(),
            "clipboard": self.clipboard,
            "dialog": self.page_obs.dialog
        }

        # Save agent state
        system_state['saved_info'] = self.saved_info
        system_state['task_status'] = {
            "intent": self.intent,  
            "task_complete": self.task_complete,
            "completed_subgoals": self.completed_subgoals,
            "last_milestone": self.last_milestone,
            "current_subgoal": self.current_subgoal,
        }
        self.trajectory.append(system_state)
        self.prev_states.append(system_state)
        self.last_state_i = len(self.trajectory)-1

        return


    def reload_state(self, system_state: SystemState, env_only=False):
        """"""

        state_obs = system_state['observation']
        self.page_obs = state_obs['page_mem']

        if not env_only:
            self.time_step = system_state['nth']
            self.saved_info = system_state['saved_info']
            task_status = system_state['task_status']
            self.intent = task_status['intent']
            self.completed_subgoals = task_status['completed_subgoals']
            self.last_milestone = task_status['last_milestone']
            self.current_subgoal = task_status['current_subgoal']

        return

    
    def revert_action(self, prev_action: AgentAction, prev_state: SystemState):
        """"""
        
        element: Element = prev_action['meta_data']['element']
        if element:
            self.invalid_elems.append(element)
        
        if len(self.env.context.pages) > len(prev_state['observation']['browser_tabs']):
            self.env.context.pages[-1].close()
            self.env.page = self.env.context.pages[-1]
            self.env.page.bring_to_front()
        else:
            self.env.page.goto(prev_state['url'])
            time.sleep(2.0)

        # Rewind trajectory to before last action
        self.reload_state(prev_state)
        self.trajectory = self.trajectory[:-1]

        return


    def check_url_allowed(self, url: str):
        """"""

        if not self.allowed_websites:
            return True

        website_url = get_website_url(url)
        website_name = site_name(url, sim_url=False).lower()

        if (website_url in self.allowed_websites):
            return True
        if any(website_name in l for l in self.allowed_websites):
            return True
        if any(site_name(l, sim_url=False).lower() in url for l in self.allowed_websites):
            return True
        
        return False


    def update_state(self):
        """"""

        if (self.time_step < 1) or (not self.save_history) or (len(self.trajectory)<2):
            return
        
        website_url = get_website_url(self.env.page.url)
        website_mem = None
        if website_url in self.memory.websites:
            website_mem = self.memory.websites[website_url]
        elif 'service-now' in website_url:
            website_mem = self.website_mem
        prev_state: SystemState = self.trajectory[self.last_state_i]
        prev_html = prev_state['observation']['html']
        prev_page = prev_state['observation']['page_mem']
        prev_action = self.trajectory[-1]
        if prev_action['action_type'] == 'compound':
            prev_action = prev_action['actions'][-1]
        self.screen_change = ""
        self.dropdown_level = 0
        element: Element = prev_action['meta_data']['element']
        if prev_action['state_updated']:
            # skip if already updated (tab switch, go_back to state)
            return


        # Check env state change
        if (len(self.env.context.pages) != len(self.browser_tabs)):
            self.env.page = self.env.context.pages[-1]
            self.env.page.bring_to_front()
            if not self.check_url_allowed(self.env.page.url):
                logger.warning(f"Invalid website: {website_url}, revert last action")
                self.revert_action(prev_action, prev_state)
                return
            self.browser_tabs = self.env.context.pages
            self.browser_tabs_obs.append(self.page_obs)
            self.browser_tab_index = len(self.browser_tabs) - 1
            prev_action['env_change'] = f'New tab opened: "{get_sim_url(self.env.page.url)}".'
            logger.success(f'New tab opened: "{get_sim_url(self.env.page.url)}".')
        
        if (self.env.page.url != prev_state['url']):
            if not self.check_url_allowed(self.env.page.url):
                logger.warning(f"Invalid website: {website_url}, revert last action")
                self.revert_action(prev_action, prev_state)
                return
            if (self.website_bookmarks) and (website_mem) and (website_url not in self.website_bookmarks):
                logger.debug(f"New website")
                self.bookmarks = self.relevant_bookmarks(website_mem)
                self.website_bookmarks[website_url] = self.bookmarks
            if (self.env.page.url in self.visited_pages):
                # retrieve memory of seen page from earlier time step
                page_mem = self.visited_pages[self.env.page.url]
                logger.debug(f"Page '{page_mem.name}' already visited, reload page_mem")
                page_mem.similar = True
                if self.env.page.content() != page_mem.html:
                    page_mem = self.update_pagemem(website_mem, page_mem, page_mem)
            else:
                is_list_item = prev_state['observation']['page_mem'].is_list_item
                website_mem, page_mem = self.get_mem(is_list_item=is_list_item)
            self.page_obs = page_mem
            self.browser_tabs_obs[self.browser_tab_index] = page_mem
            prev_action['env_change'] = f'Page url changed to "{get_sim_url(self.env.page.url)}".'
        elif (self.check_dialog(check_only=True)):
            prev_action['env_change'] = f'Dialog message'
        elif (self.env.page.content() == prev_state['observation']['html']):
            if prev_state['observation']['scroll_height'] != self.page_scroll_height():
                prev_scroll = prev_state['observation']['scroll_height']
                new_scroll = self.page_scroll_height()
                prev_action['env_change'] = f"Scrolled from y={prev_scroll} to y={new_scroll}."
            elif prev_action['action_type'] not in ['enter_input', 'choose_select_option']:
                prev_action['env_change'] = "No change."
            else:
                prev_action['env_change'] = prev_action['action_summary']
        else:
            # page html changed but no url change
            page_mem = self.create_page_mem(website_mem, full_url=True)
            page_mem = self.divide_page(website_mem, page_mem, screenshot=False)
            self.page_obs = self.update_pagemem(website_mem, page_mem, prev_page)


        if (self.page_obs.n_missing_sections > 1) or (self.page_obs.empty_sections >= 3):
            logger.debug(f"missing_sections: {self.page_obs.n_missing_sections}, empty_sections: {self.page_obs.empty_sections}")
            page_mem = self.create_page_mem(website_mem, full_url=True)
            page_mem = self.divide_page(website_mem, page_mem, True)
            page_mem.add_bookmark = False
            page_mem = self.update_pagemem(website_mem, page_mem, page_mem, new=True)
            self.page_obs = page_mem
        
        if (prev_page.name != "Homepage") and (self.pages_match(prev_page)) and (prev_page.url in self.page_obs.url):
            self.page_obs.similar = True
        if self.check_alerts():
            prev_action['env_change'] = f"Page message after action: {self.page_obs.alerts}"
        if (element) and (not element.no_dropdown) and (not prev_action['env_change']):
            if (prev_action['action_type'] in ['click_element', 'click_coords']):
                self.screen_change = self.screen_diff(prev_action)
        
        print_action(prev_action, step=self.time_step)

        self.visited_pages[self.env.page.url] = self.page_obs
        if self.changed_website:
            self.bookmarks = self.relevant_bookmarks(website_mem)
            self.changed_website = False

        return


    def screen_diff(self, action: AgentAction=None, action_str=None) -> str:
        """Compare screenshot before and after last action."""

        if self.cfg.llm_only:
            return ""

        logger.debug(f"VLM describe screen change")
        website_name = site_name(self.env.page.url)
        if action:
            action_str = action['action_summary']
        if action_str:
            action_str = action_str[0].lower() + action_str[1:]
            vlm_prompt = self.prompts.screen_diff_action.format(
                website_name=website_name,
                action_str=action_str
            )
        else:
            vlm_prompt = self.prompts.screen_diff
        
        if action_str.startswith('entered value of ""'):
            return ""

        # Prompt VLM
        vlm_out = self.model_manager.multi_img_vlm_call(
            images=[self.prev_screenshot, self.post_screenshot],
            vlm_prompt=vlm_prompt
        )
        
        if action:
            action['env_change'] = f"Result: {vlm_out}"

        return vlm_out
    

    def get_dialog_section(self, dialog_loc, element: Element) -> PageSection:
        """"""
        
        if not dialog_loc:
            logger.warning(f"Can't locate dialog window")
            self.key_press_action([], 'Escape')
            return None

        iframe_id = None
        if element:
            iframe_id = element.iframe_id
        dialog_section = self.create_section(dialog_loc, iframe_id, is_dialog=True)
        dialog_section, elem_diff = self.update_section(dialog_section, resummarize=True)
        if not dialog_section:
            return None

        dialog_actions = []
        n_buttons = 0
        n_inputs = 0
        for elem in dialog_section.elements:
            if (elem.tag=='button'):
                elem_loc = self.get_elem_locator(elem)
                if (elem_loc.count() != 1) or (elem.is_disabled):
                    logger.debug(f"Skip {elem.get_name()}, count={elem_loc.count()}")
                    continue
                n_buttons += 1
            if (elem.tag in ['input', 'textarea', 'select']):
                n_inputs += 1
            if elem.contained_in != 'form':
                elem.contained_in = 'dialog'
            dialog_actions.append(elem)
        dialog_section.elements = dialog_actions
        dialog_section.print_section()

        if (not dialog_actions):
            logger.warning(f"No dialog actions")
            self.key_press_action([], 'Escape')
            return None

        return dialog_section
    

    def check_alertdialog(self, element: Element=None, check_only=False):
        """"""

        if element:
            base_locator = self.get_base_locator(element)
        else:
            return None

        dialog_loc = None
        try:
            dialog_loc = base_locator.locator('div[role="alertdialog"]').filter(visible=True).first
            aria_modal = dialog_loc.get_attribute('aria-modal', timeout=1000)=='true'
        except Error as e:
            logger.warning(f"{str(e)}")
            dialog_loc = None
        if not dialog_loc:
            return None

        if (check_only) and (aria_modal):
            return True
        
        iframe_id = None
        if element:
            iframe_id = element.iframe_id
        dialog_section = self.create_section(dialog_loc, iframe_id, is_dialog=aria_modal)
        self.update_section(dialog_section, resummarize=True)

        if (not aria_modal) and (not check_only):
            self.page_obs.html_sections.insert(1, dialog_section)
            return None
        else:
            return dialog_section


    def check_dialog(self, element: Element=None, check_only=False):
        """"""

        global last_dialog_msg
        logger.info(f"{last_dialog_msg}")
        
        time.sleep(2.0)

        # check chrome dialog message
        self.page_obs.browser_dialog = None
        if last_dialog_msg:
            self.page_obs.browser_dialog = last_dialog_msg
            logger.debug(f"{self.page_obs.browser_dialog}")
            last_dialog_msg = ""
            if check_only:
                return True
        
        if (not element) and (self.trajectory):
            if 'meta_data' in self.trajectory[-1]:
                element = self.trajectory[-1]['meta_data']['element']
        
        # check alert dialog
        alertdialog_section = self.check_alertdialog(element, check_only=check_only)
        if alertdialog_section:
            logger.debug(f"new alertdialog")
            if check_only:
                return True
            return alertdialog_section

        # check for modal dialog window
        page_dialog = self.check_modal(element, close_popup=False)
        if page_dialog:
            logger.success(f"Dialog window opened")
            if check_only:
                return True
            dialog_section = self.get_dialog_section(page_dialog, element)
            self.page_obs.dialog = True
            self.page_obs.dialog_section = dialog_section
            return dialog_section
        else:
            self.page_obs.dialog = False
            self.page_obs.dialog_section = None
            self.page_obs.dialog_field_edited = None
            return None

    
    # Analyze observation
    def top_page_sections(self, section_list: list[PageSection]=None, done_obs=False):
        """"""

        if (not self.cfg.split_page_sections):
            return self.page_obs.html_sections, []
        if section_list == None:
            page_sections = []
            for section in self.page_obs.html_sections:
                if (section.summary) or (section.list_item_type):
                    page_sections.append(section)
            if self.page_obs.list_section:
                page_sections.append(self.page_obs.list_section)
        else:
            page_sections = section_list
        if len(page_sections) == 0:
            return [], []
        
        # identify promising sections
        relevant_sections = self.choose_top_sections(page_sections, done_obs=done_obs)
        if (len(relevant_sections) > 4) and (len(page_sections) < 10):
            if (len(relevant_sections) == len(page_sections)) and (relevant_sections[-1].bbox.get_abs_px_height()<250) and (relevant_sections[-1].bbox.y1_abs_px>1000):
                return relevant_sections[:-1], relevant_sections[-1:]
            if (len(relevant_sections) < len(page_sections)):
                relevant_sections = self.choose_top_sections(relevant_sections)
        self.page_obs.task_sections = relevant_sections


        for i in range(len(relevant_sections)):
            section = relevant_sections[i]
            if i == 0:
                section.is_primary_section = True
            if (len(section.elements)==1) and (section.n_dropdowns==1):
                element = section.elements[0]
                if element.tag != 'a':
                    self.click_element(element, section)
                    section.elements = element.dropdown_elements
                    section.n_dropdowns -= 1
                    self.update_section(section, inputs_only=True, resummarize=True)
                    section.updated = True

        other_sections = []
        for section in page_sections:
            if section not in relevant_sections:
                other_sections.append(section)
        
        if (not section_list) and (not self.manual):
            if (len(relevant_sections)==1) and (len(other_sections)>3) and (self.time_step>0):
                other_relevant, other_sections = self.top_page_sections(other_sections)
                relevant_sections += other_relevant

        return relevant_sections, other_sections


    def extract_details(self, section: PageSection, details: str=None):
        """"""

        if not details:
            logger.warning(f"No details for section: {section.type} {section.class_name}")
            return ""
        if (section.extracted_attr_info):
            section.task_details = section.relevant_items_str
            return section.task_details
        if (section.task_details) and (details == section.section_text):
            if (section.bbox.get_abs_px_height()<100) and (section.bbox.y1_abs_px>720):
                logger.debug(f"Re-use relevant details")
                return section.task_details
        if (section.type=='aside') and ('dialog' in details) and (len(details)<200):
            section.summary = ""
            section.task_details = details
            return details
        logger.debug(f"Extract relevant section details")

        sys_prompt = self.prompts.extract_section_details
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        if (self.page_obs.similar) and (self.page_obs.list_section) and (self.page_obs.list_section.sort_order):
            current_page += f'\n- Summary: {self.page_obs.list_section.sort_order}'
        else:
            current_page += f"\n- Summary: {self.page_obs.page_summary}"
            if (section.sort_order):
                current_page += f" {section.sort_order}"
        
        if section.class_name:
            class_str = clean_text(section.class_name, max_lines=1)
        else:
            class_str = ""
        section_str = f'SECTION: {section.type} class="{class_str}"'
        content = f"CONTENT: '''\n{details}\n'''"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n{section_str}\n\n{content}"

        logger.info(get_sim_url(f"\n{current_page}\n\n{section_str}\n\n{content}"))
        if self.manual:
            return
        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=False)
        summary = self.extract_llm_answer(llm_out, '**SUMMARY**:', line_only=True)
        if any(c in summary for c in ['no further action', 'no additional action', 'task is complete']):
            logger.debug(f"remove premature conclusion")
            summary = ""
        extracted = self.extract_llm_answer(llm_out, '**RELEVANT DETAILS**:', line_only=False)
        if any(extracted.startswith(s) for s in ['None', '- None', '- No relevant']):
            extracted = ""
        if ('{' in extracted) or ('}' in extracted):
            extracted = extracted.replace('{', '')
            extracted = extracted.replace('}', '')

        section.task_summary = summary
        section.task_details = extracted

        return extracted


    def summarize_page_obs(self, page_context: str, vlm_context=False) -> str:
        """"""

        logger.debug(f"Summarize page observation")
        show_prompt = True

        sys_prompt = self.prompts.summarize_obs
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        page_observation = f"PAGE OBSERVATION:"
        if vlm_context:
            page_observation += f" '''{page_context}'''"
        else:
            page_observation += f"\n{page_context}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n{page_observation}"

        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        if self.manual:
            print(user_prompt)
            return "[Summary placeholder]"
        llm_out = self.model_manager.llm_call(messages, show_prompt=show_prompt)
        obs_summary = self.extract_llm_answer(llm_out, '**OBSERVATION SUMMARY**:', line_only=False)

        return obs_summary

    
    def summarize_page_screen(self, detailed=False) -> str:
        """"""

        if (self.page_obs.page_summary) and ((self.manual) or (self.time_step==0)):
            return self.page_obs.page_summary
        
        if (self.manual) or (not self.cfg.filter_page_info):
            return ''
        if self.cfg.llm_only:
            dom_snapshot = self.env.page.locator('body').aria_snapshot()
            page_summary = self.summarize_page_obs(dom_snapshot)
            return page_summary

        if not detailed:
            prompt = self.prompts.site_page_report.format(
                website_name=site_name(self.page_obs.url), 
                page_name=self.page_obs.name)
            max_h = 1280 if self.page_obs.page_height<4000 else 720
        else:
            prompt = self.prompts.site_page_describe.format(
                website_name=site_name(self.page_obs.url), 
                page_name=self.page_obs.name)
            max_h = 1280

        page_summary = self.describe_screen(prompt=prompt, max_height=max_h)
        if (page_summary) and ('inactive' in page_summary):
            page_summary = self.page_obs.short_summary

        return page_summary

    
    def analyze_page(self, relevant_sections: list[PageSection]):
        """"""

        if not self.cfg.split_page_sections:
            dom_snapshot = self.env.page.locator('body').aria_snapshot()
            self.page_obs.dom_snapshot = dom_snapshot
            if self.cfg.summarize_state:
                self.page_obs.page_summary = self.summarize_page_obs(dom_snapshot)
            else:
                self.page_obs.page_summary = dom_snapshot
            return

        # Analyze relevant page sections
        all_section_details: list[(PageSection, str)] = []
        for i in range(len(relevant_sections)):
            section = relevant_sections[i]
            if (section.type=='table') or (section.list_type=='table'):
                self.table_action(section)
            elif (section.list_type in ['list', 'grid']):
                self.list_action(section)
            if section.not_relevant: continue
            section_details = self.section_info(section, caption_img=self.cfg.caption_img, vlm_info=True)
            if section.table_filters:
                filters_info = f"Table Actions:\n{self.elems_context(section.table_filters, numbered=False)}\n\n"
                section_details = filters_info + section_details
            all_section_details.append((section, section_details))

        if self.cfg.filter_page_info:
            relevant_info = []
            for section, details in all_section_details:
                relevant_details = self.extract_details(section, details)
                if not relevant_details: continue
                info_bullets = relevant_details.splitlines()
                for line in info_bullets:
                    line = line.strip()
                    if (section.list_type) or (':' in line):
                        relevant_info.append(line)

        if self.manual: return


        # Summarize page
        logger.debug(f"Summarize page observation")
        sys_prompt = self.prompts.summarize_obs
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        if (len(self.browser_tabs_obs)>1):
            current_page += f"\n- Open tabs:"
            for tab_page in self.browser_tabs_obs:
                current_page += f'\n	- "{tab_page.name}" ("{get_sim_url(tab_page.url)}")'
                if tab_page == self.page_obs:
                    current_page += f" [focused]"
        
        page_info = f''
        if self.page_obs.alerts:
            page_info += f'**Alert message**: {self.page_obs.alerts[0]}\n\n'
        self.page_obs.alerts = []
        if self.screen_change:
            page_info += f'{self.screen_change}\n\n'
        if self.clipboard:
            page_info += f'**Clipboard**: "{self.clipboard}"\n\n'

        if (self.cfg.filter_page_info) and (not relevant_info):
            page_info += f'{self.page_obs.page_summary}\n\n'
        page_info += f'Here is an overview of the different sections of the "{self.page_obs.name}" page:\n'
        if relevant_sections:
            page_info += f'{self.page_sections_context(relevant_sections, task_details=True, add_title=False)}'
        else:
            page_info += f'{self.page_sections_context(self.page_obs.html_sections, task_summary=False)}'
        
        page_observation = f"PAGE OBSERVATION:\n{page_info}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n{page_observation}"
        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=True)
        obs_summary = self.extract_llm_answer(llm_out, '**OBSERVATION SUMMARY**:', line_only=False)

        # Add info to pagemem
        if (self.cfg.filter_page_info):
            self.page_obs.task_summary = obs_summary
            self.page_obs.task_info = "\n".join(relevant_info)
        self.page_obs.page_summary = obs_summary

        return


    def get_obs(self) -> PageMem:
        """"""

        dialog_section = self.check_dialog()
        if dialog_section:
            relevant_sections = [dialog_section]
        else:
            relevant_sections, other_sections = self.top_page_sections()
        
        self.analyze_page(relevant_sections)

        return self.page_obs

    
    # Get relevant actions
    def relevant_elements(self, element_list: list[Element], section: PageSection=None):
        """"""

        element_list = self.useful_elements(element_list, section)
        if not element_list:
            return []
        if (not self.cfg.filter_page_info):
            return element_list
        if (section):
            if (section.is_dialog) and (len(element_list)<20):
                return element_list
            if ('file' in str(section.class_name)) and (len(section.elements)<5):
                section, _ = self.update_section(section, summarize=False)
                element_list = section.elements
        logger.debug(f"Identify relevant elements in section")

        elems_info = self.elems_context(element_list, section)
        elem_str_list = self.elem_str_list(element_list, section)

        if (len(element_list) <= 10):
            sys_prompt = self.prompts.relevant_elements_a
            use_letters = True
        else:    
            sys_prompt = self.prompts.relevant_elements
            use_letters = False
        
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        current_page += f"\n- Summary: {self.page_obs.page_summary}"
        
        if section:
            if (section.list_items) and (len(section.list_items)<10):
                current_page += f"\n{self.indent_str(section.relevant_items_str)}"
            elif (section.section_text) and (len(section.section_text)<2500):
                current_page += f"\n\nPAGE CONTENT: '''{section.section_text}'''"
            elif section.desc:
                current_page += f"\n\nDESCRIPTION: '''{section.desc}'''"
        
        user_prompt = f"{task}\n\n{history}\n\n{current_page}"
        user_prompt += '\n\n**PAGE ELEMENTS**:\n{context}'

        if self.slow_mode:
            print(current_page)
        else:
            logger.info(current_page)
        chosen_elems = self.llm_choose_list(
            element_list, 
            elem_str_list, 
            sys_prompt, 
            user_prompt, 
            keyword='**SELECT ELEMENTS**:',
            backups=['**SELECT ELEMENTS:**', 'SELECT ELEMENTS:'],
            inline=True,
            show_prompt=False,
            show_options=True,
            ret_all_on_err=False,
            use_letters=use_letters
        )
        if section:
            section.task_elems = chosen_elems

        return chosen_elems

    
    def nav_actions(self):
        """"""

        nav_options: list[tuple[str, Any]] = []

        # switch tab
        for i in range(len(self.browser_tabs_obs)):
            if i == self.browser_tab_index:
                continue
            tab_page = self.browser_tabs_obs[i]
            tab_url = get_sim_url(tab_page.url)
            action_str = f'Switch to tab {i+1}: "{tab_page.name}" href="{tab_url}"'
            action_str += f'\n		- Info: {tab_page.short_summary}'
            nav_options.append((action_str, i))
        
        accounts_url_info = {}
        if self.cfg.benchmark == 'webarena':
            accounts_url_info = {
                os.environ.get('SHOPPING'): "the user does their online shopping for various products on this site",
                os.environ.get('REDDIT'): "the user's social media where they discuss posts related to their interests",
                os.environ.get('GITLAB'): "the user collaborates on various software projects here",
                os.environ.get('SHOPPING_ADMIN'): "the user manages their e-commerce website here"
            }

        if (len(self.task_websites)>1):
            other_websites = [url for url in self.task_websites if url not in self.env.page.url]
            for other_url in other_websites:
                for saved_url in self.memory.websites.keys():
                    if (other_url.startswith(saved_url)) or (saved_url.startswith(other_url)):
                        site_mem = self.memory.websites[saved_url]
                website_page = list(site_mem.pages.values())[0]
                website_name = site_name(other_url)
                action_str = f'Navigate to website: "{website_name}"'
                if other_url in accounts_url_info:
                    website_info = accounts_url_info[other_url]
                    action_str += f'\n		- Info: {website_info}'
                nav_options.append((action_str, website_page))

        # goto bookmark
        for bookmark in self.bookmarks:
            if bookmark.url in self.visited_pages:
                continue
            action_str = f'Navigate to page: "{bookmark.name}" href="{get_sim_url(bookmark.url)}"'
            action_str += f'\n		- Info: {bookmark.short_summary}'
            nav_options.append((action_str, bookmark))

        # if not start, option to go back to previous pages
        if (self.time_step > 0) and (len(self.trajectory)>=2):
            prev_page_urls = [self.env.page.url]
            prev_page_urls += [t.url for t in self.browser_tabs]
            go_back_actions = []
            for state in self.prev_states:
                step = state['nth']
                page_mem = state['observation']['page_mem']
                if page_mem.url not in prev_page_urls:
                    prev_page_urls.append(page_mem.url)
                    action_str = f'Go back to previous page: "{page_mem.name}" href="{get_sim_url(page_mem.url)}"'
                    action_str += f'\n		- Info: Last visited during step {step}'
                    go_back_actions.append((action_str, page_mem))
            go_back_actions.reverse()
            nav_options += go_back_actions

        # Option to edit url directly
        nav_options.append((f"Type a URL", None))
        nav_options.append((f"None", self.page_obs))

        return nav_options
    

    def relevant_nav_actions(self, nav_actions):
        """"""

        if (not self.cfg.filter_page_info):
            return nav_actions[:-1]
        if (len(nav_actions) == 1):
            return []
        logger.debug(f"Identify relevant navigation actions")

        opt_str_list = []
        page_list = []
        for nav_str, page in nav_actions:
            if nav_str in self.failed_actions:
                continue
            opt_str_list.append(nav_str)
            page_list.append(page)

        sys_prompt = self.prompts.relevant_nav_actions
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        current_page += f"\n- Summary: {self.page_obs.page_summary}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n"
        user_prompt += "BROWSER ACTIONS:\n"

        numbered_list = self.add_list_numbers(opt_str_list)
        options_context = '\n'.join(numbered_list)
        user_prompt += f"{options_context}"
        
        logger.info(f"\n{options_context}")
        if self.manual:
            answer = input("Answer: ")
        else:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=False)
            answer = self.extract_llm_answer(llm_out, keyword='**ANSWER**:', backups=['**ANSWER:**', 'ANSWER:'])
        if answer.startswith('None'):
            return []
        choices_str_list = answer.split(',')

        chosen_options = []
        for choice in choices_str_list:
            try:
                index = int(choice.strip()) - 1
                nav_str = opt_str_list[index].split('\n')[0]
                if nav_str == 'None':
                    continue
                chosen_options.append((nav_str, page_list[index]))
            except:
                continue

        return chosen_options
    

    # Perform action
    def choose_action(self, action_list, opt_str_list, bookmarks=[], dropdown=False, another_action=False, incomplete=False):
        """"""

        if (len(action_list) == 1) and (opt_str_list[0].startswith('Mark task as complete')):
            logger.debug(f"No actions remaining, end task")
            self.no_actions_left = True
            return action_list[0]
        
        actions = f"ACTIONS:"
        if dropdown:
            actions = f"DROPDOWN ACTIONS:"
        actions += f"\n{self.format_list_to_str(opt_str_list, numbered=True)}"

        # Format prompt
        sys_prompt = self.prompts.choose_action
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        if (not self.cfg.split_page_sections):
            current_page += f"\n\n{self.page_obs.dom_snapshot}\n"
        if (not incomplete) and (not another_action) and (self.cfg.summarize_state):
            current_page += f"\n- Summary: {self.page_obs.page_summary}"
        elif (another_action and self.screen_change):
            current_page += f"\n- **Update**: {self.screen_change}"
        elif incomplete:
            current_page += f'\n{self.page_sections_context(self.page_obs.task_sections, task_details=True, add_title=False)}'
        
        if self.page_obs.browser_dialog:
            current_page += f"\n- **NOTE**: the browser displayed the following dialog message after the previous action was clicked:"
            current_page += f"\n	- {self.page_obs.browser_dialog}"
        if bookmarks:
            bookmark_info = ", ".join(bookmarks)
            current_page += f"\n- Available bookmarks: {bookmark_info}"
        if ('Navigate to website:' in actions):
            current_page += f"\n- Relevant websites: {self.task_websites}"
        if (len(self.browser_tabs_obs)>1):
            current_page += f"\n- Open tabs:"
            for tab_page in self.browser_tabs_obs:
                current_page += f'\n	- "{tab_page.name}" ("{get_sim_url(tab_page.url)}")'
                if tab_page == self.page_obs:
                    current_page += f" [focused]"

        if self.clipboard:
            current_page += f'\n- Clipboard: "{self.clipboard}"'
        
        if (not incomplete) and (len(self.page_obs.list_info) < 15000) and (self.cfg.filter_page_info):
            if self.page_obs.list_section:
                if (0 < len(self.page_obs.list_section.list_items) < 15):
                    current_page += f"\n\n{self.indent_str(self.page_obs.list_info)}"
            elif (self.page_obs.list_info) and ('list item' in actions):
                current_page += f"\n\n{self.indent_str(self.page_obs.list_info)}"
        
        if dropdown:
            current_page += f"\n\n* Thought: {self.prev_thought}"
            current_page += f"\n* Action: {self.prev_action}"
            current_page += f"\n* Result: Dropdown menu opened"

        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n{actions}"

        # LLM chooses action
        answer = ""
        if self.manual:
            answer = input(f"{user_prompt}\n\nManual choice: ")
        if not answer:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages)
            reason = self.extract_llm_answer(llm_out, keyword='**REASON**:')
            # reason = self.extract_llm_answer(llm_out, keyword='**PLAN**:')
            self.current_subgoal = reason
            answer = self.extract_llm_answer(llm_out, keyword='**SELECT ACTION**:')
        if answer.startswith('None'):
            return None
        if len(answer.split(' ')) > 1:
            answer = answer.split(' ')[0]
        if answer.endswith(')'):
            answer = answer[:-1]
        try:
            index = int(answer) - 1
            selected_action = action_list[index]
        except Exception as e:
            logger.error(f"{e}\n{traceback.format_exc()}")
            if len(opt_str_list) > 200:
                logger.debug(f"Retry with truncated options list")
                return self.choose_action(action_list, opt_str_list[:200], bookmarks, dropdown, another_action, incomplete)
            else:
                return None
        if self.slow_mode: self.wait_user_input()

        return selected_action


    def multi_row_action(self, row_inputs: list[Element]):
        """"""

        logger.success(f"Perform bulk action on table rows")
        start_page = self.page_obs
        print([e.get_name() for e in row_inputs])
        if self.slow_mode: self.wait_user_input()

        for input_elem in row_inputs:
            self.element_action(input_elem, input_elem.section)
            if self.env.page.url != start_page.url:
                break
        
        self.current_subgoal = f'I will now select all relevant items in the list.'

        return


    def exec_action(self, action_candidates, nav_candidates=[], another_action=False, incomplete=False):
        """"""

        logger.success(f"Choose next action to execute")

        action_list = []
        opt_str_list = []
        bookmarks = []
        row_inputs = []
        for action in (action_candidates + nav_candidates):
            if (action[0] in self.failed_actions) or (action[0] in self.invalid_elems):
                continue
            if (action[0] in self.repeated_actions) and (len(self.repeated_actions) >= self.max_repeat):
                continue
            if isinstance(action[0], Element):
                element = action[0]
                if (element.is_disabled):
                    continue
                if (element.href) and (element.href.startswith('tel:')):
                    continue
                if (element.tag=='button'):
                    if (self.page_obs.dialog_field_edited) and (element.get_name() in ['Close', 'Exit']):
                        continue
                if (another_action) and (element.clicked):
                    if (element.tag in ["input", "textarea", "select"]) or (element.role=='combobox'):
                        continue
                action_str = f'{self.get_elem_str(element, max_options=5)}'
                if (action_str in self.repeated_actions) and (len(self.repeated_actions) >= self.max_repeat):
                    continue
                if len(action_list) < 50:
                    action_str = f'Click element: {self.get_elem_str(element, max_options=30)}'
                opt_str_list.append(action_str)
                if (element.row_info) or (element.table_row):
                    if (element.tag=='input') or (element.input_type=='submit'):
                        row_inputs.append(element)
            elif isinstance(action[1], PageMem):
                logger.debug(f"{action[0]}")
                if action[0].startswith('Navigate to page:'):
                    page = action[1]
                    if f'Click element: link "{page.name}" href="{page.url}"' in opt_str_list:
                        logger.debug(f"same link in options list")
                        continue
                    action_str = f'Go to bookmarked page: "{page.name}" href="{page.url}"'
                    bookmarks.append(f'"{page.name}" ("{page.url}")')
                else:
                    action_str = action[0]
                if action_str in opt_str_list:
                    logger.debug(f"same action in options list")
                    continue
                opt_str_list.append(action_str)
            else:
                opt_str_list.append(action[0])
            action_list.append(action)
        if (not self.page_obs.dialog) and (not incomplete):
            action_list.append(('End task', None))
            opt_str_list.append('Mark task as complete.')

        
        selected_action = self.choose_action(action_list, opt_str_list, bookmarks, another_action=another_action, incomplete=incomplete)
        if not selected_action:
            return None, opt_str_list


        # Execute action
        if isinstance(selected_action[0], Element):
            element = selected_action[0]
            page_section = selected_action[1]
            action_str = self.get_elem_str(element)
            if (not page_section) or (not self.cfg.compound_actions):
                self.element_action(element)
            elif (element in row_inputs) and (len(row_inputs)>1):
                # page_section.input_modified=True
                self.multi_row_action(row_inputs)
            elif page_section.list_type in ['list', 'grid']:
                if element in row_inputs: page_section.input_modified=True
                self.element_action(element, element.section)
            elif ((page_section.type=='form') or (element.contained_in=='form')) and ((element.tag in ['input', 'textarea', 'select']) or (element.role=='combobox') or (element.dropdown_elements)):
                if element.get_name().startswith('Search'):
                # if element.input_type=='search':
                    self.element_action(element, page_section)
                else:
                    if another_action: page_section.task_elems = [element]
                    self.submit_form(page_section)
            else:
                success = self.element_action(element, page_section)
                if not success:
                    logger.error(f"Action execution failed")
                    # raise Exception
                    return None, opt_str_list
            dialog = self.check_dialog(element, check_only=True)
        else:
            nav_str = selected_action[0]
            nav_page = selected_action[1]
            action_str = nav_str
            if (nav_str.startswith('Type a URL')):
                self.enter_url()
            elif nav_str.startswith('Navigate to page:'):
                self.goto_page(nav_page)
            elif nav_str.startswith('Go back to previous page:'):
                self.return_to_page(nav_page)
            # switch tab
            elif nav_str.startswith('Switch to tab'):
                self.switch_tab(nav_page)
            elif nav_str.startswith('Navigate to website:'):
                self.goto_website(nav_page)
            elif nav_str.startswith('End task'):
                self.task_complete = True
                return create_action('stop'), opt_str_list
        
        # Add to history
        agent_action = None
        if len(self.action_stack) == 1:
            self.trajectory.append(self.action_stack[0])
            agent_action = self.action_stack[0]
        elif len(self.action_stack) > 1:
            compound_action = create_action('compound')
            compound_action['meta_data']['element'] = element
            compound_action['actions'] = self.action_stack.copy()
            self.trajectory.append(compound_action)
            agent_action = self.action_stack[-1]
        else:
            logger.error(f"No action executed")
            self.failed_actions.append(selected_action[0])
            return None, opt_str_list
        
        if action_str in self.repeated_actions:
            self.repeated_actions.append(action_str)
            logger.info(f"Action repeated: {len(self.repeated_actions)}")
        else:
            self.repeated_actions = [action_str]

        self.action_stack = []
        self.failed_actions = []
        self.update_history(another_action=another_action)

        return agent_action, opt_str_list
    

    def check_task_completion(self, relevant_sections: list[PageSection], opt_str_list: list[str]) -> bool:
        """"""

        logger.success(f"Check task completion")
        if self.manual:
            return True

        # Format prompt
        sys_prompt = self.prompts.check_task_complete
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        if self.page_obs.browser_dialog:
            current_page += f"\n- **NOTE**: the browser displayed the following dialog message after the previous action was clicked:"
            current_page += f"\n    - {self.page_obs.browser_dialog}"
        
        if (len(relevant_sections) <= 5):
            current_page += f'\n{self.page_sections_context(relevant_sections, task_details=True, add_title=False)}'
        else:
            current_page += f"\n- Summary: {self.page_obs.page_summary}"
            if self.page_obs.list_section:
                if (0 < len(self.page_obs.list_section.list_items) < 15): # and (self.page_obs.list_section.task_elems):
                    current_page += f"\n\n{self.indent_str(self.page_obs.list_info)}"

        actions = f"ACTIONS:"
        actions += f"\n{self.format_list_to_str(opt_str_list[:-1], numbered=True)}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}\n\n{actions}"

        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=True)
        answer = self.extract_llm_answer(llm_out, keyword='**TASK COMPLETE**:')
        if answer.startswith('False'):
            return False

        return True


    def wait_user_input(self):
        """"""

        user_input = input(f"Press 'Enter' to proceed (type '0' to stop, '1' to finish task, 'm' to manually take over): ")
        
        if not user_input:
            return
        elif user_input == '0':
            self.task_complete = True
        elif user_input == '1':
            self.slow_mode = False

        return


    def page_action(self, website_mem: WebsiteMem, max_actions=5) -> str:
        """"""
        
        logger.success(f"======== Step {self.time_step} ========")

        # Observation
        try:
            self.update_state()
        except Error as e:
            logger.error(f"Error while updating state:\n{repr(e)}\n{traceback.format_exc()}")
            logger.debug(f"Wait and retry...")
            time.sleep(5.0)
            self.update_state()

        start_url = self.env.page.url
        start_html = self.env.page.content()
        self.page_obs.page_summary = self.summarize_page_screen()

        dialog_section = self.check_dialog()
        if (dialog_section) and (self.cfg.split_page_sections):
            relevant_sections, other_sections = [dialog_section], []
            self.page_obs.task_sections = relevant_sections
            dialog_loc = self.get_section_locator(dialog_section)
            dialog_context = self.section_content(dialog_loc)['text']
            if not dialog_context:
                dialog_context = self.section_content_md(dialog_loc)['text']
            self.page_obs.task_summary = self.summarize_page_obs(dialog_context, vlm_context=True)
            self.page_obs.page_summary = self.page_obs.task_summary
        else:
            relevant_sections, other_sections = self.top_page_sections()
            self.analyze_page(relevant_sections)


        # Plan
        if self.slow_mode: self.wait_user_input()
        
        logger.success(f"Get relevant actions")
        # navigation actions
        nav_candidates: list[tuple[str, Any]] = []
        nav_options = self.nav_actions()
        nav_candidates += self.relevant_nav_actions(nav_options)
        # page actions
        action_candidates: list[tuple[str, Any]] = []
        selected_elems = []
        other_elems = []
        for section in relevant_sections:
            if section.not_relevant: continue
            elems = self.relevant_elements(section.elements, section)
            selected_elems += elems
            for elem in elems:
                action_candidates.append((elem, section))
            if (not elems) and (len(section.elements)<=10) and (section.type!='form'):
                other_elems += section.elements
        
        if (not self.manual) and (self.cfg.filter_page_info):
            logger.debug("Check other actions")
            for other_section in other_sections[:5]:
                if (len(other_section.elements)>20) or (other_section.bbox.y1_abs_px>500) or (other_section.type=='form'):
                    continue
                other_elems += other_section.elements
            relevant_elems = self.relevant_elements(other_elems)
            for elem in relevant_elems:
                if (elem.input_type=='search') and (self.page_obs.filter_table):
                    continue
                action_candidates.append((elem, elem.section))

        # (maybe) next_subgoal
        self.page_obs.relevant_actions = action_candidates
        self.record_state_v1()


        # Action
        agent_action, opt_str_list = self.exec_action(action_candidates, nav_candidates)
        if (self.task_complete) and (self.task_checks < 1):
            # double-check task completion
            if (action_candidates or nav_candidates):
                self.task_checks += 1
                complete = self.check_task_completion(relevant_sections, opt_str_list)
                if (not complete):
                    logger.info(f"Task incomplete, continuing")
                    self.task_complete = False
                    agent_action, _ = self.exec_action(action_candidates, nav_candidates, incomplete=True)
                    if not agent_action:
                        self.task_complete = True
                        return
        

        # Retry failed action
        retries = 0
        while (not agent_action) and (retries < self.max_repeat):
            retries += 1
            logger.debug(f"Retry action selection: {retries}")
            agent_action, opt_str_list = self.exec_action(action_candidates, nav_candidates)
        if not agent_action:
            logger.error(f"Exceeded max action retries: {self.max_repeat}, terminating")
            self.task_complete = True


        action_count = 1
        while (action_count<max_actions) and (self.env.page.url==start_url) and (self.cfg.compound_actions):
            if (dialog_section) or (self.check_dialog()):
                break
            if (not self.page_obs.browser_dialog):
                if (not agent_action) or (self.page_obs.filled_form) or (agent_action['action_type'] not in ['enter_input', 'choose_select_option', 'copy']):
                    break
            
            logger.debug(f"Choose another action")
            self.screen_change = self.screen_diff(action_str=self.prev_action)

            page_sections = []
            for section in self.page_obs.html_sections[1:5]:
                if (len(section.elements)>20) or (section.type in ['header', 'footer']):
                    continue
                page_sections.append(section)
            for section in relevant_sections:
                if (section not in page_sections) and (section.type not in ['header', 'footer']):
                    page_sections.append(section)

            action_candidates = []
            for section in page_sections:
                if section.type!='table':
                    section, elem_diff = self.update_section(section)
                if not section:
                    continue
                elements = self.useful_elements(section.elements)
                for elem in elements:
                    action_candidates.append((elem, section))

            action_count += 1
            agent_action, _ = self.exec_action(action_candidates, nav_candidates=[], another_action=True)


        logger.debug(f"Completed Step {self.time_step}\n\n\n\n")
        self.time_step += 1

        return




    # Navigation
    def saved_pages(self, website_mem: WebsiteMem, prev_urls=[], add_path=False):
        """"""

        all_urls = []
        logins = 0
        self.use_bookmarks = True

        for url in website_mem.pages:
            all_urls.append(url)
            if 'login' in url:
                logins += 1
        if (len(all_urls) < 15) and (logins > 2):
            self.use_bookmarks = False
        if (len(all_urls) < 50):
            self.use_bookmarks = False

        all_page_mem = []
        all_page_str = []
        for url in sorted(all_urls):
            if len(all_page_mem) > 400:
                break
            if (url in prev_urls) or (url == self.page_obs.url):
                continue  # avoid repeating previous pages
            page_mem = website_mem.pages[url]
            if (not page_mem.add_bookmark) or ((page_mem.list_page) and (not page_mem.list_section)):
                # logger.debug(f"Skip: {page_mem.url}")
                continue
            all_page_mem.append(page_mem)
            page_title = clean_text(page_mem.name, max_words=7, max_length=200)
            page_str = f"{page_title}"
            if add_path:
                path = url_path(url)
                page_str += f": {path}"
            all_page_str.append(page_str)

        return all_page_mem, all_page_str
    

    def relevant_bookmarks(self, website_mem: WebsiteMem, nav_subgoal=None, debug=True) -> list[PageMem]:
        """"""

        if not self.cfg.use_bookmarks:
            return []

        logger.success(f"Look for relevant bookmarks")
        if not nav_subgoal:
            nav_subgoal = self.intent

        all_page_mem, all_page_str = self.saved_pages(website_mem, [self.page_obs.url], add_path=False)
        if len(all_page_mem) < 500:
            chunk_size = len(all_page_mem)
        else:
            chunk_size = 250

        if len(all_page_mem) > 20:
            sys_prompt = self.prompts.page_candidates
            user_prompt = f'USER: {nav_subgoal}\n\n'
            if (self.time_step > 0):
                user_prompt += f'HISTORY:\n{self.episode_history()}\n\n'
                user_prompt += f'CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})\n\n'
            user_prompt += 'ALL PAGES:\n{context}'
            backups = ['RELEVANT PAGES:']
            candidate_pages = self.llm_choose_list(
                all_page_mem, 
                all_page_str, 
                sys_prompt, 
                user_prompt, 
                keyword='**RELEVANT PAGES**:', 
                backups=backups,
                chunk_size=chunk_size, 
                inline=True,
                show_prompt=debug
            )
            if (not candidate_pages) or (len(candidate_pages)>20):
                if self.slow_mode: self.wait_user_input()
                return []
        else:
            candidate_pages = all_page_mem
        
        page_str_list = []
        for page_mem in candidate_pages:
            page_title = clean_text(page_mem.name, max_words=7, max_length=200)
            page_str = f"{page_title}: {url_path(page_mem.url)}"
            page_str += f"\n\t\t- {page_mem.short_summary}"
            page_str_list.append(page_str)
        sys_prompt = self.prompts.page_candidates
        user_prompt = f'USER: {nav_subgoal}\n\n'
        if (self.time_step > 0):
            user_prompt += f'HISTORY:\n{self.episode_history()}\n\n'
            user_prompt += f'CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})\n\n'
        user_prompt += 'ALL PAGES:\n{context}'
        backups = ['RELEVANT PAGES:']
        top_pages = self.llm_choose_list(
            candidate_pages, 
            page_str_list, 
            sys_prompt, 
            user_prompt, 
            keyword='**RELEVANT PAGES**:', 
            backups=backups,
            chunk_size=20, 
            inline=True,
            show_prompt=True
        )
        if not top_pages:
            logger.warning(f"No page chosen")

        self.page_obs.bookmarks = top_pages
        if self.slow_mode: self.wait_user_input()

        return top_pages


    def switch_tab(self, tab_index: int) -> AgentAction:
        """"""

        logger.success(f"Switch to tab {tab_index}")
        start_i = self.browser_tab_index
        start_page = self.page_obs
        self.browser_tabs_obs[start_i] = start_page

        if tab_index not in range(len(self.browser_tabs)):
            logger.error(f"Tab index {tab_index} not in range {len(self.browser_tabs)}")
            return None
        
        self.env.page = self.browser_tabs[tab_index]
        self.env.page.bring_to_front()
        self.page_obs = self.browser_tabs_obs[tab_index]
        self.browser_tab_index = tab_index

        website_url = get_website_url(self.page_obs.url)
        self.bookmarks = self.website_bookmarks.get(website_url, [])

        action = create_action('switch_tab')
        action['action_summary'] = f"Switched from tab {start_i} ({get_sim_url(start_page.url)}) to tab {tab_index} ({get_sim_url(self.page_obs.url)})."
        action['state_updated'] = True
        self.record_action(action)
        
        return action
    

    def goto_website(self, page: PageMem):
        """"""

        nav_action = create_action('goto_page')
        logger.debug(f"Change page url")
        self.go_to_url_action([], page.url)

        website_url = get_website_url(page.url)
        website_mem = self.memory.websites[website_url]
        website_name = site_name(page.url)
        nav_action['action_summary'] = f'Navigated to the "{website_name}" website ({get_sim_url(page.url)}).'
        self.record_action(nav_action)

        # login if needed
        _, success = self.login_action([])
        _, success = self.go_to_url_action([], page.url)

        self.changed_website = True

        return
    

    def goto_page(self, page: PageMem):
        """"""

        nav_action = create_action('goto_page')
        logger.debug(f"Change page url")
        self.go_to_url_action([], page.url)
        nav_action['action_summary'] = f'Navigated to page "{page.name}" ({get_sim_url(page.url)}).'
        self.record_action(nav_action)

        return
    

    def return_to_page(self, page: PageMem):
        """"""

        nav_action = create_action('goto_page')
        logger.debug(f"Change page url")
        self.go_to_url_action([], page.url)

        self.page_obs = page
        nav_action['action_summary'] = f'Returned to the "{page.name}" page.'
        nav_action['state_updated'] = True
        page.similar = True

        self.record_action(nav_action)

        return
    

    def enter_url_value(self) -> str:
        """"""

        # Format context
        sys_prompt = self.prompts.enter_url
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.page_obs.name} ({get_sim_url(self.page_obs.url)})"
        current_page += f"\n- Summary: {self.page_obs.page_summary}"
        user_prompt = f"{task}\n\n{history}\n\n{current_page}"
        
        # Prompt LLM
        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages)
        answer = self.extract_llm_answer(llm_out, keyword='**ANSWER**:')

        if self.cfg.benchmark in ['webarena', 'visualwebarena']:
            answer = revert_sim_url(answer)

        return answer


    def enter_url(self, max_retries=1):
        """"""

        logger.success(f"Manually enter a url")
        start_url = self.env.page.url

        tries = 0
        while tries < max_retries:
            tries += 1
            if self.manual:
                url = input("Enter url: ")
            else:
                url = self.enter_url_value()
            
            if not self.check_url_allowed(url):
                success = False
            else:
                _, success = self.go_to_url_action([], url)
            
            if success:
                break
            elif (tries == max_retries):
                self.go_to_url_action([], start_url)
                return
            else:
                logger.warning(f"Invalid url, retry")
        

        # add to history
        action = create_action('edit_url')
        action['action_summary'] = f'Navigated to URL: "{url}"'
        self.record_action(action)

        return




    # Planning
    def check_alerts(self) -> bool:
        """Check for page alert messages"""

        new_alert = False
        
        all_alert_loc = self.env.page.locator(f'[id="messages"], [role="alert"]').filter(visible=True).all()
        for alert_loc in all_alert_loc:
            alert_text = self.section_content_md(alert_loc, add_quotes=False)['text']
            alert_text = alert_text.strip()
            if (alert_text) and (alert_text not in self.page_obs.alerts):
                new_alert = True
                self.page_obs.alerts.append(f'{alert_text}')
                logger.warning(f"Alert message: {alert_text}")

        return new_alert


    def subgoal_actions(self) -> list:
        """Get list of actions taken since last completed subgoal."""
        
        # Trajectory: [State, Action, State, ... , Action, State]
        subgoal_actions = []
        i = -2
        while -(i) <= len(self.trajectory):
            action = self.trajectory[i]
            if action['action_type'] != 'eval':
                subgoal_actions.append(action)
            if action['nth'] <= self.last_milestone:
                break
            i -= 2

        subgoal_actions.reverse()
        return subgoal_actions
    

    def action_history(self) -> str:
        """Get the string with list action summaries for current subgoal."""

        if self.cfg.version == 4:
            actions_str = ""
            for action in self.action_stack:
                action_summary = action['action_summary']
                actions_str += f'\n* {action_summary}'
            return actions_str

        actions_str = ""

        subgoal_actions = self.subgoal_actions()
        if not subgoal_actions:
            return ""
        
        # subgoal_actions.reverse()
        for n in range(len(subgoal_actions)):
            action = subgoal_actions[n]
            action_summary = action['action_summary']
            if (len(action_summary.split('\n')) > 1):
                if action['action_type'] == 'enter_input':
                    action_summary = f'Entered message'
            actions_str += f'{n+1}. {action_summary}\n'

        # Add result of last action
        env_change = action['env_change']
        if (env_change.startswith('Page url changed')) or (env_change.startswith('Was the action')):
            actions_str += f'    - Result: {env_change}'

        return actions_str.strip()


    def progress_str(self, add_current=False, subgoal_actions=False, title=True, subgoal=False) -> str:
        """"""

        if self.cfg.version == 4:
            progress = f'PROGRESS:'
            if subgoal:
                progress += f'\n- [ ] (Current subgoal) {self.current_subgoal}'
                for action in self.action_stack:
                    action_summary = action['action_summary']
                    progress += f'\n    - [x] {action_summary}'
            else:
                for action in self.action_stack:
                    action_summary = action['action_summary']
                    progress += f'\n- [x] {action_summary}'
            
            return progress


        progress = f'PROGRESS:'
        if not title:
            progress = f''
        if self.history_str:
            progress += f'\n{self.history_str.strip()}'
        
        for subgoal in self.completed_subgoals:
            # If subgoal is multiline, indent the list items
            lines = subgoal.split('\n')
            if len(lines) > 1:
                indented_lines = []
                for line in lines[1:]:
                    if line.startswith('- '):
                        line = '- [x] '+line[2:]
                    indented = f"    {line}"
                    indented_lines.append(indented)
                subgoal = '\n'.join([lines[0]]+indented_lines)
            # Add completed subgoal to progress
            progress += f'\n- [x] {subgoal}'

        if add_current and self.current_subgoal:
            progress += f'\n- [ ] (Current subgoal) {self.current_subgoal}'

        if subgoal_actions and add_current:
            actions_list = self.subgoal_actions()
            for action in actions_list:
                action_summary = action['action_summary']
                progress += f'\n    - [x] {action_summary}'

        return progress
    

    def update_history(self, another_action=False, full_details=False) -> str:
        """"""

        history = self.history_str
        
        if another_action:
            # replace prev step 'Action' with 'Actions'
            lines = history.splitlines(keepends=True)
            last_line = lines[-1]
            if "Action:" in last_line:
                last_action = last_line.split(':')[-1].strip()
                last_line = f"  * Actions:\n    - {last_action}\n"
                lines[-1] = last_line
            self.history_str = "".join(lines)
            # add another action
            action = self.trajectory[-1]
            if action['action_type'] == 'compound':
                for subaction in action['actions']:
                    action_str = subaction['action_summary']
                    self.history_str += f'    - {action_str}\n'
            else:
                action_str = action['action_summary']
                self.history_str += f'    - {action_str}\n'
            return self.history_str
        
        last_state = self.trajectory[-2]
        timestep = last_state['nth']+1
        step = f"* Step {timestep}:\n"

        state_obs = last_state['observation']
        browser_tabs = state_obs['browser_tabs']
        page_mem = state_obs['page_mem']
        observation = f"* Observation: {page_mem.name} ({get_sim_url(page_mem.url)})"
        observation += f"\n  - Summary: {page_mem.page_summary}"
        detailed_observation = observation
        if (page_mem.task_info):
            detailed_observation += f"\n  - Relevant info:\n{self.indent_str(page_mem.task_info, 2, indent_first=True)}"

        thought = f""

        last_action = self.trajectory[-1]
        if last_action['action_type'] == 'compound':
            all_actions = last_action['actions']
            subgoal = all_actions[0]['reason']
            if subgoal:
                thought = f"\n* Reason for Action: " + subgoal
            action_info = f"* Actions:"
            for subaction in all_actions:
                action_info += '\n  - '
                action_info += subaction['action_summary']
        else:
            thought = f"\n* Reason for Action: " + last_action['reason']
            action_summary = last_action['action_summary']
            action_info = f"* Action: {action_summary}"
        
        step_info = f"{step}{observation}{thought}\n{action_info}"
        if not self.cfg.summarize_state:
            step_info = f"* Step {timestep}:{thought}\n{action_info}"

        step_info = self.indent_str(step_info)
        history += f"{step_info}\n"
        self.history_str = history
        detailed_step_info = self.indent_str(f"{step}{detailed_observation}{thought}\n{action_info}")
        self.detailed_history += f"{detailed_step_info}\n"

        return history


    def episode_history(self, full_details=False) -> str:
        """"""

        if self.history_str:
            history = self.history_str
        else:
            website_url = get_website_url(self.env.page.url)
            if (website_url in self.memory.accounts):
                account_details = self.memory.accounts[website_url]
                username = account_details['username']
                website_name = site_name(website_url)
                if website_name == "Onestopmarket":
                    history = f"* Step 0: Task started (date: 2023, location: US) - I have logged in to my {website_name} account ({username}) and handed over control of the browser for you to complete the task. You have full permission to pay for orders on my behalf.\n"
                else:
                    history = f"* Step 0: Task started (date: 2023, location: US) - I have logged in to my {website_name} account ({username}) and handed over control of the browser for you to complete the task.\n"
            elif self.cfg.current_date:
                date_str = date.today().strftime("%B %d, %Y")
                history = f"* Step 0: Task started (current date: {date_str}) - the has user has handed off control of the browser for you to complete their task.\n"
            else:
                history = f"* Step 0: Task started - the has user has handed off control of the browser for you to complete their task.\n"

            self.history_str = history

        return history.strip()
    

    def retrieve_history_details(self) -> str:
        """"""

        if not self.cfg.filter_page_info:
            return ""

        logger.debug(f"Retrieve information from history")
        
        # Format prompt
        sys_prompt = self.prompts.retrieve_history_details
        task = f"TASK: {self.intent}"
        history = f"HISTORY:\n{self.detailed_history}"
        current_page = f"CURRENT PAGE: {self.get_page_info(self.page_obs, full_details=True)}"
        user_prompt = f'{task}\n\n{history}\n\n{current_page}'

        if self.manual:
            print(user_prompt)
            relevant_details = input("Relevant details: ")
        else:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages, show_prompt=True)
            relevant_details = self.extract_llm_answer(llm_out, keyword='**RELEVANT DETAILS**:', line_only=False)
        
        if relevant_details:
            self.page_obs.task_info = relevant_details
            self.page_obs.retrieved_history = True

        return relevant_details


    def format_answer(self, task_complete: bool=True) -> str:
        """"""

        if self.manual:
            answer = input("Final answer: ")
            self.answer = answer
            return answer
        logger.success(f"Format final answer")

        sys_prompt = self.prompts.final_answer
        user = f"USER: {self.intent}"
        history = f"HISTORY:\n{self.episode_history()}"
        current_page = f"CURRENT PAGE: {self.get_page_info(self.page_obs, full_details=True)}"
        user_prompt = f"{user}\n\n{history}\n\n{current_page}"

        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=True)
        message = self.extract_llm_answer(llm_out, keyword='**MESSAGE**:', line_only=False)
        answer = self.extract_llm_answer(llm_out, keyword='**ANSWER**:', line_only=False)

        if (len(message)>500 and len(message.splitlines())>3) and (len(answer.split(' ')) > 1):
            answer = message

        if self.saved_info["saved_links"]:
            for info in self.saved_info["saved_links"]:
                answer += f"\n{info}"

        if self.cfg.benchmark in ['webarena', 'visualwebarena']:
            answer = revert_sim_url(answer)
        if answer.startswith("`") and answer.endswith("`"):
            answer = answer[1:-1]

        self.answer = answer
        if answer:
            self.log_trajectory()

        return answer
        
    


    def page_image_intent(self) -> str:
        """"""

        user_prompt = f'''Describe what the following user is asking for based on their request message that references both the first image and the web page screenshot.\\n\\n**User request message**: "{self.intent}"'''
        screenshot, bbox = self.section_screenshot(self.page_obs.list_section, max_height=5120)
        images = self.input_images + [screenshot]
        image_paths = self.input_image_paths + ["screenshot.png"]
        vlm_out = self.model_manager.multi_img_vlm_call(images, user_prompt, image_paths=image_paths)

        # Summarize VLM output
        sys_prompt = self.prompts.summarize_image_details
        task = f'USER TASK REQUEST: "{self.intent}"\n- **Note*: image file attached to message'
        website = f'WEBSITE: '
        start_website = get_website_url(self.env.page.url)
        if 'http://localhost:7770/' in start_website:
            website += f'OneStopMarket (online shopping site)'
        elif 'http://localhost:9980/' in start_website:
            website += f'OSClass (classifieds site)'
        elif 'http://localhost:9999/' in start_website:
            website += f'Reddit (forum site)'
        website += f'\n- Current page: {self.page_obs.list_section.summary}'
        task_analysis = f"TASK ANALYSIS: '''{vlm_out}'''"
        user_prompt  = f"{task}\n\n{website}\n\n{task_analysis}"
        messages = self.format_llm_prompt(sys_prompt, user_prompt)
        llm_out = self.model_manager.llm_call(messages, show_prompt=True)
        summary = self.extract_llm_answer(llm_out, keyword='**ANSWER**:', line_only=False)

        self.intent += f"\n- An image has been provided as context for the task: {summary}"
        self.image_task_context = summary

        return summary
    

    def image_intent(self) -> str:
        """"""

        logger.debug(f"Analyze input image")

        if (self.env.page.url not in ['http://localhost:7770/', 'http://localhost:9980/', 'http://localhost:9999/']):
            if (len(self.input_images)==1) and (self.page_obs.list_section):
                return self.page_image_intent()

        if len(self.input_images) == 1:
            sys_prompt = self.prompts.image_details
            instruction = f"Please analyze the provided image and clearly describe the relevant details of the user's task."
        else:
            sys_prompt = self.prompts.multi_img_details
            instruction = f"Please analyze the provided images and clearly describe the relevant details of the user's task."

        task = f'USER: "{self.intent}"'
        website = f'WEBSITE: '
        start_website = get_website_url(self.env.page.url)
        if 'http://localhost:7770/' in start_website:
            website += f'OneStopMarket (online shopping site)'
        elif 'http://localhost:9980/' in start_website:
            website += f'OSClass (classifieds site)'
        elif 'http://localhost:9999/' in start_website:
            website += f'Reddit (forum site)'
        if self.env.page.url not in ['http://localhost:7770/', 'http://localhost:9980/', 'http://localhost:9999/']:
            website += f'\n- Current page: {self.page_obs.name}'
            if (len(self.page_obs.name.split(' '))<3):
                website += f' ({get_sim_url(self.page_obs.url)})'
        
        user_prompt = f"{website}\n\n{task}\n\n{instruction}"
        if len(self.input_images) == 1:
            vlm_out = self.model_manager.vlm_call(self.input_images[0], user_prompt, sys_prompt, image_path=self.input_image_paths[0])
            self.intent += f"\n- An image has been provided as context for the task: {vlm_out}"
        else:
            vlm_out = self.model_manager.multi_img_vlm_call(self.input_images, user_prompt, sys_prompt, image_paths=self.input_image_paths)
            self.intent += f"\n- {len(self.input_images)} images have been provided as context for the task: {vlm_out}"
        
        self.image_task_context = vlm_out
        
        return vlm_out


    def check_multi_site(self):
        """"""

        if self.cfg.benchmark == 'visualwebarena':
            return

        start_website = get_website_url(self.env.page.url)
        self.task_websites = [start_website]

        start_name = site_name(start_website)
        website_list = ["OneStopMarket", "Reddit", "GitLab", "OpenStreetMap", "Wikipedia", "Magento"]
        accounts_info = {
            "OneStopMarket": "the user does their online shopping for various products on this site",
            "Reddit": "the user's social media where they discuss posts related to their interests",
            "GitLab": "the user collaborates on various software projects here",
            "Magento": "the user manages their e-commerce website here"
        }

        sys_prompt = self.prompts.check_multi_site
        user = f'USER: {self.intent}'
        current_page = f'CURRENT PAGE: {self.page_obs.name} {get_sim_url(self.page_obs.url)}'
        current_page += f'\n  - {self.page_obs.page_summary}'
        websites = f'WEBSITES:'
        for site in website_list:
            websites += f'\n* {site}'
            if start_name.lower() == site.lower():
                websites += f' (current)'
            if site in accounts_info:
                info = accounts_info[site]
                websites += f'\n	* {info}'
        user_prompt = f"{user}\n\n{current_page}\n\n{websites}"

        if self.manual:
            chosen_websites = input(f"{user_prompt}\n\nWebsites: ")
        else:
            messages = self.format_llm_prompt(sys_prompt, user_prompt)
            llm_out = self.model_manager.llm_call(messages)
            chosen_websites = self.extract_llm_answer(llm_out, keyword='**ANSWER**:')

        website_urls = []
        if "OneStopMarket" in chosen_websites:
            website_urls.append(os.environ.get('SHOPPING'))
        if "Reddit" in chosen_websites:
            website_urls.append(os.environ.get('REDDIT'))
        if "GitLab" in chosen_websites:
            website_urls.append(os.environ.get('GITLAB'))
        if "OpenStreetMap" in chosen_websites:
            website_urls.append(os.environ.get('MAP'))
        if "Wikipedia" in chosen_websites:
            website_urls.append(os.environ.get('WIKIPEDIA'))
        if "Magento" in chosen_websites:
            website_urls.append(os.environ.get('SHOPPING_ADMIN'))
        if start_website not in website_urls:
            website_urls.append(start_website)
        
        self.task_websites = website_urls

        return


    def complete_task(self, intent: str=None, max_steps: int=None):
        """Agent executes actions in the environment until task is complete."""

        if intent:
            self.intent = intent
        logger.success(f"Start new task")
        logger.debug(f"Agent settings:\n{str(self.cfg)}\n")
        logger.success(f"[Intent]: {self.intent}")

        # Initialize trajectory files
        self.log_trajectory(initialize=True)

        # get environment memory for all pages
        self.browser_tabs = self.env.context.pages
        self.browser_tabs_obs = []
        for i in range(len(self.env.context.pages)):
            self.env.page = self.env.context.pages[i]
            self.env.page.bring_to_front()
            time.sleep(1.0)
            website_mem, page_mem = self.get_mem()
            self.page_obs = page_mem
            self.browser_tabs_obs.append(page_mem)
            self.browser_tab_index = i

            self.page_obs.page_summary = self.summarize_page_screen()
            if i == 0:
                if (len(self.browser_tabs_obs)==1) and (len(self.allowed_websites) > 1):
                    self.check_multi_site()
                if self.input_images:
                    self.image_intent()
                if self.slow_mode: self.wait_user_input()

            # Check relevant bookmarks
            if self.cfg.use_bookmarks:
                website_url = get_website_url(page_mem.url)
                homepage = next(iter(website_mem.pages.values()))
                if ((page_mem.url==homepage.url) or (page_mem.url.endswith('all'))) and (website_url not in self.website_bookmarks):
                    self.bookmarks = self.relevant_bookmarks(website_mem)
                    self.homepage_start = True
                else:
                    self.bookmarks = []
                    self.homepage_start = False
                self.website_bookmarks[website_url] = self.bookmarks
        
        self.env.page = self.browser_tabs[0]
        self.env.page.bring_to_front()
        self.page_obs = self.browser_tabs_obs[0]
        self.browser_tab_index = 0
        self.bookmarks = self.website_bookmarks.get(get_website_url(self.page_obs.url), [])


        if not max_steps:
            max_steps = self.max_steps
        if self.cfg.max_steps:
            max_steps = self.cfg.max_steps
        if self.cfg.benchmark=='workarena':
            self.max_steps = 10
        if self.slow_mode: self.wait_user_input()

        n_page_actions = 0
        while True:
            if (n_page_actions >= self.max_page_actions):
                logger.warning(f"Max page actions: {self.max_page_actions}")
                break
            if (self.time_step >= self.cfg.max_steps):
                logger.warning(f"Max steps reached: {self.max_steps}")
                break
            
            self.page_action(website_mem)

            if self.task_complete == True:
                logger.success(f"Task completed.")
                break
            n_page_actions += 1

        stop_action = create_action('stop')
        final_answer = self.format_answer()
        stop_action['answer'] = final_answer
        logger.success(f"Final answer: {final_answer}")
        
        self.trajectory.append(stop_action)
    
        return
    

    def setup_webarena_task(self, config_file):
        """Initialize agent and environment to evaluate a WebArena or VisualWebArena (vwa) task"""

        logger.info(f"[Task config file]: {config_file}")

        # reset agent state
        self.reset_agent()
        if self.cfg.benchmark == 'visualwebarena':
            self.allowed_websites = [
os.environ.get('CLASSIFIEDS'), 
os.environ.get('SHOPPING'), 
os.environ.get('REDDIT'), 
os.environ.get('WIKIPEDIA')
            ]
        else:
            self.allowed_websites = [
os.environ.get('SHOPPING'), 
os.environ.get('REDDIT'), 
os.environ.get('WIKIPEDIA'),
os.environ.get('SHOPPING_ADMIN'), 
os.environ.get('GITLAB'),
os.environ.get('MAP')
            ]

        # load task config
        with open(config_file) as f:
            _c = json.load(f)
            start_url = _c["start_url"]
            self.intent = _c["intent"]
            self.init_intent = _c["intent"]
            task_id = _c["task_id"]
            image_paths = _c.get("image", None)
            images = []
            # Load images if any
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for i in range(len(image_paths)):
                    image_path = image_paths[i]
                    if image_path.startswith("http"):
                        input_image = Image.open(BytesIO(requests.get(image_path).content)).convert('RGB')
                    else:
                        input_image = Image.open(image_path).convert('RGB')
                    self.save_img(image=input_image, image_path=f'user_task_image_{i}.png', is_input_file=True)
                    if (input_image.size[0] > 1000) or (input_image.size[1] > 1000):
                        input_image.thumbnail(size=(960, 1440), resample=Image.LANCZOS)
                        print(input_image.size)
                    images.append(input_image)
                self.input_image_paths = image_paths
            self.input_images = images
        
        # setup task environment
        obs, info = self.env.reset(options={"config_file": config_file})
        self.env.context.on("dialog", handle_dialog)
        global last_dialog_msg
        last_dialog_msg = ""

        # sign in
        start_urls = start_url.split(" |AND| ")
        for i in range(len(self.env.context.pages)):
            page = self.env.context.pages[i]
            self.env.page = page
            self.env.page.bring_to_front()
            url = start_urls[i]

            _, success = self.login_action([])
            if success is False:
                logger.error(f"Login failed")
                return
            logger.debug(f"Change page url")
            _, success = self.go_to_url_action([], url)
            time.sleep(2.0)
            self.env.page.wait_for_load_state("domcontentloaded", timeout=20000)

        return


    def eval_webarena(self, task_folder=None, resume=0):
        """Run agent on benchmark tasks."""
        
        caption_image_fn = None
        if self.cfg.benchmark == 'visualwebarena':
            import torch
            caption_image_fn = get_captioning_fn(
                "cpu", torch.float32, "Salesforce/blip2-flan-t5-xl"
            )

        self.env = WebChallengerEnv(
            headless=self.headless,
            observation_type='image',
            save_trace_enabled=True,
            browser=self.browser,
            # start_url=start_url,
        )
        self.save_mem = False
        try:
            self.env.page.evaluate("navigator.clipboard.writeText('')")
        except:
            logger.warning(f"clear clipboard failed")

        if 'shopping' in self.result_dir:
            self.data_dir += f"/test_env/shopping"
        elif 'classifieds' in self.result_dir:
            self.data_dir += f"/test_env/classifieds"
        elif 'reddit' in self.result_dir:
            self.data_dir += f"/test_env/reddit"

        # Get test configs
        test_config_base_dir = self.task_config_dir
        if task_folder:
            test_config_base_dir = task_folder
        config_file_list = []
        if self.start_idx and self.end_idx:
            for i in range(self.start_idx, self.end_idx):
                config_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
        else:
            task_file_list = [file for file in os.listdir(test_config_base_dir) if file.endswith('.json')]
            for task_file in task_file_list:
                config_file_list.append(os.path.join(test_config_base_dir, task_file))
        
        sorted_config_files = []
        for i in range(999):
            for path in config_file_list:
                file = path.split('/')[-1]
                if (file.split('.')[0]==str(i)) or (file.split('_')[0]==str(i)):
                    sorted_config_files.append(path)
                    break
        config_file_list = sorted_config_files

        config_file_list = config_file_list[resume:]
        print(config_file_list)

        # Complete test tasks
        scores = []
        total_steps = 0

        for i in range(len(config_file_list)):
            config_file = config_file_list[i]
            task_id = os.path.basename(config_file).split(".")[0]
            self.task_id = task_id
            output_folder = f"{self.result_dir}/task_{task_id}"
            log_file = f"{output_folder}/system_logs.log"
            task_log_id = logger.add(log_file)
            os.makedirs(output_folder, exist_ok=True)
            self.trajectory_log_dir = output_folder
            self.trajectory_screenshot_dir = f"{self.trajectory_log_dir}/trajectory"
            os.makedirs(self.trajectory_screenshot_dir, exist_ok=True)
            
            try:
                logger.success(f"Task {i}")
                self.setup_webarena_task(config_file)
                self.complete_task()
                print(f"\n\n#### ---- Completed Task {task_id}: {self.intent} ---- ####\n")
                total_steps += self.time_step
                prompts = self.model_manager.n_llm_requests + self.model_manager.n_vlm_requests
                logger.info(f"Total tokens: {self.model_manager.total_tokens}, prompts: {prompts}")
                avg_prompt_tokens = self.model_manager.total_tokens / (prompts)
                logger.info(f"Max prompt tokens: {self.model_manager.max_req_tokens}, Avg prompt tokens: {avg_prompt_tokens}")

                evaluator = evaluator_router(
                    config_file, captioning_fn=caption_image_fn
                )
                score = evaluator(
                    trajectory=self.trajectory, 
                    config_file=config_file, 
                    page=self.env.page, 
                    client=self.env.get_page_client(self.env.page)
                )
                scores.append(score)
                if score == 1:
                    logger.success(f"[Result] (PASS)")
                else:
                    logger.error(f"[Result] (FAIL)")
                logger.info(f"Task config:\n{config_file}\n\n")

                self.log_trajectory(score=score)
                if self.slow_mode: self.wait_user_input()

            except Exception as e:
                logger.error(f"[Unhandled Error] {repr(e)}]")
                with open(f"{output_folder}/error.txt", "a") as f:
                    f.write(f"[Config file]: {config_file}\n")
                    f.write(f"[Unhandled Error] {repr(e)}\n")
                    f.write(traceback.format_exc())
                if self.slow_mode: self.wait_user_input()
            
            logger.remove(task_log_id)  # stop writing to task log file

        self.env.close()
        if len(scores):
            logger.info(f"Score: {sum(scores)} / {len(scores)}, Avg={sum(scores)/len(scores)}")

        print(f"Total steps: {total_steps}")
        print(f"Average steps per task: {total_steps / len(config_file_list)}")


    def eval_workarena(self, level: Literal[1, 2, 3]=1, start_i=0, end_i=-1, step=1):
        """"""

        from browsergym.workarena import ATOMIC_TASKS, ALL_WORKARENA_TASKS, get_all_tasks_agents


        self.cfg.benchmark = "workarena"
        self.cfg.max_steps = 10
        self.cfg.caption_img = False


        self.env = WebChallengerEnv(
            headless=self.headless,
            observation_type='image',
            save_trace_enabled=True,
            browser=self.browser,
        )
        self.save_mem = False


        if level == 1:
            all_tasks = get_all_tasks_agents(filter='l1')
            self.result_dir += f"/workarena/L1"
        elif level == 2:
            all_tasks = get_all_tasks_agents(filter='l2')
            self.result_dir += f"/workarena/L2"
        elif level == 3:
            all_tasks = get_all_tasks_agents(filter='l3')
            self.result_dir += f"/workarena/L3"
        logger.info(f"WorkArena L{level}: {len(all_tasks)} tasks")

        
        all_tasks = all_tasks[start_i:end_i]

        # Complete test tasks
        scores = []
        for i in range(0, len(all_tasks), step):
            task_entrypoint, seed = all_tasks[i]
            task_num = start_i + i
            self.task_id = task_num
            output_folder = f"{self.result_dir}/task_{task_num}"
            log_file = f"{output_folder}/system_logs.log"
            task_log_id = logger.add(log_file)
            os.makedirs(output_folder, exist_ok=True)
            self.trajectory_log_dir = output_folder
            self.trajectory_screenshot_dir = f"{self.trajectory_log_dir}/trajectory"
            os.makedirs(self.trajectory_screenshot_dir, exist_ok=True)


            bgym_env = BrowserEnv(task_entrypoint=task_entrypoint,
                    headless=self.headless,
                    viewport={"width": 1280, "height": 1280}
            )
            bgym_env.reset()
            task = bgym_env.task
            task_goal = bgym_env.goal_object[0].get("text")
            self.env.context = bgym_env.context
            self.env.page = bgym_env.page

            page = self.env.page

            time.sleep(2.0)
            self.env._wait_dom_loaded()
            self.check_load_screen()
            self.env.context.on("dialog", handle_dialog)
            logger.info(f"\nTask {task_num}: ", task_entrypoint)
            task_goal = task_goal.replace('{', '')
            task_goal = task_goal.replace('}', '')
            logger.info(task_goal)

            global last_dialog_msg
            last_dialog_msg = ""
            self.reset_agent()

            self.env.page.mouse.move(500, 350)
            self.env.page.mouse.wheel(0, 5000)
            time.sleep(2.0)
            self.env.page.mouse.wheel(0, -5000)

            try:
                chat_messages = [{'role': "assistant", "message": ""}]
                reward, stop, message, info = task.validate(page, chat_messages)  # Check for validation bug
                self.intent = task_goal

                self.log_trajectory(initialize=True)
                self.browser_tabs = self.env.context.pages
                self.browser_tabs_obs = []
                website_mem, page_mem = self.get_mem()
                self.page_obs = page_mem
                self.browser_tabs_obs.append(page_mem)
                self.browser_tab_index = 0

                n_actions = 0
                reward = 0
                while n_actions <= 10:
                    self.page_action(website_mem)
                    if self.task_complete == True:
                        logger.success(f"Task completed.")
                        break
                    n_actions += 1
                    reward, stop, message, info = task.validate(page, chat_messages)
                    logger.info(message)
                    if (stop) or (self.page_obs.filled_form):
                        break
                
                stop_action = create_action('stop')
                if self.task_complete == True:
                    final_answer = self.format_answer()
                    stop_action['answer'] = final_answer
                    logger.success(f"Final answer: {final_answer}")
                self.trajectory.append(stop_action)

                answer = self.answer
                chat_messages = [{'role': "assistant", "message": answer}]
                
                if not reward:
                    reward, stop, message, info = task.validate(page, chat_messages)
                    # Hacky way to validate all subtasks at once (since compositional tasks expect validation after each subtask)
                    while (not stop) and (message.endswith('has been completed successfully.')):
                        reward, stop, message, info = task.validate(self.env.page, chat_messages)
                
                if reward == 1:
                    logger.success(f"[Result] (PASS)\n{info}")
                else:
                    logger.error(f"[Result] (FAIL)\n{info}")
                scores.append(reward)
                logger.info(f"Task type: {task_entrypoint}\n\n")
                
                self.log_trajectory(score=reward)
                if self.slow_mode: self.wait_user_input()
            
            except Exception as e:
                logger.error(f"[Unhandled Error] {repr(e)}]\n{traceback.format_exc()}")
                with open(f"{output_folder}/error.txt", "a") as f:
                    f.write(f"[Task type]: {task_entrypoint}\n")
                    f.write(f"[Unhandled Error] {repr(e)}\n")
                    f.write(traceback.format_exc())

            try:
                bgym_env.close()
            except Exception as e:
                pass
            logger.remove(task_log_id)
        
        self.env.close()
        if len(scores):
            logger.info(f"Score: {sum(scores)} / {len(scores)}, Avg={sum(scores)/len(scores)}")


    def eval_mind2web(self, start_i=0, end_i=-1):
        """"""

        self.env = WebChallengerEnv(
            headless=self.headless,
            observation_type='image',
            save_trace_enabled=True,
            browser=self.browser
        )
        self.env.viewport_size = {"width": 1280, "height": 1280}
        try:
            self.env.page.evaluate("navigator.clipboard.writeText('')")
        except:
            logger.warning(f"clear clipboard failed")
        

        benchmark_file_path = "webchallenger/benchmarks/visualwebarena/config_files/Online_Mind2Web.json"
        with open(benchmark_file_path, 'r') as f:
            all_tasks = json.load(f)
        if end_i == -1:
            all_tasks = all_tasks[start_i:]
        else:
            all_tasks = all_tasks[start_i:end_i]
        print(f"{len(all_tasks)} tasks")
        

        for i in range(len(all_tasks)):
            task_dict = all_tasks[i]
            confirmed_task = task_dict['confirmed_task']
            website = task_dict['website']
            site = site_name(website)
            self.allowed_websites = [get_website_url(website)]

            self.env.reset()
            _, success = self.go_to_url_action([], website)
            if (not success) and (website not in self.env.page.url):
                continue
            time.sleep(2.0)
            self.env._wait_dom_loaded()
            self.env.context.on("dialog", handle_dialog)
            global last_dialog_msg
            last_dialog_msg = ""
            self.reset_agent()

            self.task_id = task_dict['task_id']
            output_folder = f"{self.result_dir}/{site}/{self.task_id}"
            log_file = f"{output_folder}/system_logs.log"
            task_log_id = logger.add(log_file)
            os.makedirs(output_folder, exist_ok=True)
            self.trajectory_log_dir = output_folder
            self.trajectory_screenshot_dir = f"{self.trajectory_log_dir}/trajectory"
            os.makedirs(self.trajectory_screenshot_dir, exist_ok=True)

            try:
                logger.success(f"Task {i}")
                self.intent = confirmed_task
                self.complete_task()
                print(f"\n\n#### ---- Completed Task {i}: {self.intent} ---- ####\n\n\n\n")
                if self.slow_mode: self.wait_user_input()

            except Exception as e:
                logger.error(f"[Unhandled Error] {repr(e)}]\n{traceback.format_exc()}")
                with open(f"{output_folder}/error.txt", "a") as f:
                    f.write(f"[Task type]: {task_dict}\n")
                    f.write(f"[Unhandled Error] {repr(e)}\n")
                    f.write(traceback.format_exc())
            
            logger.remove(task_log_id)  # stop writing to task log file

        self.env.close()

        return


    


    def explore_websites(self, websites):
        self.env = WebChallengerEnv(
            headless=self.headless,
            observation_type='image',
            save_trace_enabled=True,
            browser=self.browser,
        )
        self.env.viewport_size = {"width": 1280, "height": 1280}
        

        if websites == "webarena":
            self.env.setup()
            env_list = ['SHOPPING', 'REDDIT', 'GITLAB', 'WIKIPEDIA', 'MAP', 'SHOPPING_ADMIN']
            url_list = [os.environ.get(env_name) for env_name in env_list]
        elif websites == "visualwebarena":
            self.env.setup()
            env_list = ['CLASSIFIEDS', 'SHOPPING', 'REDDIT', 'WIKIPEDIA']
            url_list = [os.environ.get(env_name) for env_name in env_list]
        elif websites == "workarena_l1":
            from browsergym.workarena import get_all_tasks_agents
            all_tasks = get_all_tasks_agents(filter='l1')
            bgym_env = BrowserEnv(
                    task_entrypoint=all_tasks[0][0],
                    headless=True,
                    viewport={"width": 1280, "height": 1280}
            )
            bgym_env.reset()
            self.env.context = bgym_env.context
            self.env.context.grant_permissions(['clipboard-read', 'clipboard-write'])
            self.env.page = bgym_env.page
            url_list = [self.env.page.url]
        elif websites == "online_mind2web":
            self.env.setup()
            self.cfg.exploration_depth = 1
            self.cfg.exploration_time_limit = 3600

            benchmark_file_path = "webchallenger/benchmarks/visualwebarena/config_files/Online_Mind2Web.json"
            with open(benchmark_file_path, "r") as f:
                data = json.load(f)
            url_list = list(dict.fromkeys(item["website"] for item in data))
        else:
            self.env.setup()
            url_list = [url.strip() for url in websites.split(',')]

        # explore all websites
        errors = []
        for url in url_list:
            _, success = self.go_to_url_action([], url)
            _, success = self.login_action([])
            _, success = self.go_to_url_action([], url)
            if not success:
                errors.append(url)
                continue
            output_folder = f"webchallenger/memory/saved_files/exploration_log/{url}"
            log_file = os.path.join(output_folder, "system_logs.log")
            task_log_id = logger.add(log_file)
            self.env.page.wait_for_load_state("domcontentloaded", timeout=20000)
            try:
                website_mem, page_mem = self.get_mem()
                self.summarize_all_pages(website_mem)
            except Exception as e:
                errors.append(url)
                logger.error(repr(e))
                logger.error(traceback.format_exc())
                with open(os.path.join(output_folder, "error.txt"), "a") as f:
                    f.write(f"[Unhandled Error] {repr(e)}\n")
                    f.write(traceback.format_exc())
            logger.remove(task_log_id)


    def quickstart(self, intent: str, start_url: str=""):
        """"""
        
        if not start_url:
            start_url = self.start_url
        self.env = WebChallengerEnv(
            headless=False,
            observation_type='image',
            save_trace_enabled=True,
            browser=self.browser,
            start_url=start_url,
        )
        self.env.viewport_size = {"width": 1280, "height": 1280}
        self.env.setup()
        self.env.context.on("dialog", handle_dialog)

        self.browser_tabs = self.env.context.pages
        self.browser_tab_index = 0

        logger.remove()
        logger.add(sys.stdout, level="INFO")

        # Sign in
        _, success = self.login_action([])
        if success is False:
            logger.error(f"Login failed")
            return
        logger.debug(f"Change page url")
        _, success = self.go_to_url_action([], start_url)
        self.env.page.wait_for_load_state("domcontentloaded", timeout=20000)

        try:
            self.env.page.evaluate("navigator.clipboard.writeText('')")
        except:
            logger.warning(f"clear clipboard failed")
        self.env.page.set_default_timeout(5000)


        if not intent:
            self.intent = input("Instruction for agent: ")
        else:
            self.intent = intent

        website_mem, page_mem = self.get_mem()
        self.page_obs = page_mem
        self.browser_tabs_obs = [page_mem]


        self.trajectory_log_dir = f"{self.result_dir}/test_trajectory"
        self.trajectory_screenshot_dir = f"{self.trajectory_log_dir}/trajectory"
        os.makedirs(f"{self.trajectory_log_dir}/trajectory", exist_ok=True)
        self.log_trajectory(initialize=True)
        

        if (self.time_step == 0) and (self.cfg.use_bookmarks) and (website_mem.pages):
            homepage = next(iter(website_mem.pages.values()))
            if (page_mem.url == homepage.url) or (page_mem.url.endswith('all')):
                self.bookmarks = self.relevant_bookmarks(website_mem, debug=True)

        for i in range(30):
            self.page_action(website_mem)
            if self.task_complete:
                break
        
        stop_action = create_action('stop')
        if self.task_complete == True:
            final_answer = self.format_answer()
            stop_action['answer'] = final_answer
            logger.success(f"Final answer: {final_answer}")
        else:
            stop_action['answer'] = "N/A"
            logger.warning(f"Unable to complete task")
        self.trajectory.append(stop_action)


        input("Done task, press any key to exit.")
        website_mem.set_page_mem(page_mem)
        self.memory.save_website_mem(website_mem, replace=True)

        self.env.close()





class CFG:
    version: int = 4
    current_date: bool = False

    exploration_depth: int = 2
    exploration_time_limit: int = 43200

    max_steps: int = 30

    suggest_nav: bool = True
    use_bookmarks: bool = True

    scout_sections: bool = True
    only_relevant_sections: bool = True
    analyze_full_page: bool = False
    caption_img: bool = True

    table_chunk_size: int = 50
    list_chunk_size: int = 25


    llm_only: bool = False
    filter_page_info: bool = True
    compound_actions: bool = True
    split_page_sections: bool = True
    summarize_state: bool = True

    benchmark: str = None
    # benchmark: str='webarena'
    # benchmark: str='visualwebarena'
    # benchmark: str='workarena'
    # benchmark: str='mind2web'

    def __str__(self):
        return "\n".join(f"{key} = {value}" 
            for key, value in self.__class__.__dict__.items()
            if not key.startswith("__"))


def run_agent(vision: str, planning: str, browser: str, args: SimpleNamespace):
    agent = Agent(vision, planning, browser, load_mem=True, args=args, config=CFG())
    model_manager.init(vision, planning, args)

    # Explore
    if args.explore_websites:
        agent.explore_websites(args.explore_websites)
    
    # Benchmark
    if args.benchmark == "webarena":
        agent.cfg.benchmark = 'webarena'
        agent.result_dir += f"/webarena"
        agent.eval_webarena("webchallenger/benchmarks/visualwebarena/config_files/test_webarena")
    elif args.benchmark == "visualwebarena":
        agent.cfg.benchmark = 'visualwebarena'
        agent.result_dir += f"/visualwebarena"
        base_dir = agent.result_dir

        agent.result_dir = f"{base_dir}/shopping"
        agent.eval_webarena("webchallenger/benchmarks/visualwebarena/config_files/test_shopping")

        agent.result_dir = f"{base_dir}/reddit"
        agent.eval_webarena("webchallenger/benchmarks/visualwebarena/config_files/test_reddit")

        agent.result_dir = f"{base_dir}/classifieds"
        agent.eval_webarena("webchallenger/benchmarks/visualwebarena/config_files/test_classifieds")
    
    elif args.benchmark == "workarena_l1":
        agent.cfg.benchmark = 'workarena'
        agent.eval_workarena(level=1)

    elif args.benchmark == "online_mind2web":
        agent.cfg.benchmark = 'mind2web'
        agent.cfg.caption_img = False
        agent.result_dir += f"/online_mind2web"
        agent.eval_mind2web()


    # Custom task
    else:
        agent.quickstart(intent=args.intent)

