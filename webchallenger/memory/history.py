import json
import os
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Union

from loguru import logger
from PIL import Image
from playwright.sync_api import Locator, Page

from webchallenger.memory import Element, PageMem
from webchallenger.utils import Singleton, BBox, url_to_name, is_state_class
from webchallenger.visualwebarena import (
    Action,
    ActionTypes,
    StateInfo,
    Trajectory,
)




class AgentAction(Action):
    # types:
        # compound
        # switch_tab, google_search, goto_website, goto_page, edit_url
        # click_element, click_coords, choose_select_option, enter_input, create_file, upload_file
        # take_note, download, copy
        # eval, stop
    action_type: str

    url: str
    nth: int
    answer: str
    subgoal: str

    meta_data: dict[str, Any]
    actions: list[Any]

    reason: str
    action_summary: str
    env_change: str
    # error
    state_updated: bool


def create_action(
    action_type: str, 
    element: Element=None, 
    elem_desc: str=None, 
) -> AgentAction:
    """"""

    action: AgentAction = dict()

    action['action_type'] = action_type
    action['answer'] = ''

    action['meta_data'] = {
        "element": element,
        "elem_desc": elem_desc,
        "input_value": None,
        "click_coords": None,
        "click_button": None,
        "elem_crop": None,
        
        "subgoal_completed": False,
    }
    action['actions'] = []

    action['reason'] = ""
    action['action_summary'] = ""
    action['env_change'] = ""
    action['state_updated'] = False

    return action


def print_action(action: AgentAction, step: int=None):
    """Print action info to terminal."""

    logger.success(f"Step {step}: thoughts = '{action['reason']}'")

    action_str = f"Action type: {action['action_type']}"
    
    answer = action['answer']
    if answer:
        action_str += f"\nanswer: {answer}"
    element = action['meta_data']['element']
    if element:
        action_str += f"\nelement: <{element.tag}> {element.get_name()}"
    elem_desc = action['meta_data']['elem_desc']
    if elem_desc:
        action_str += f"\nelem_desc: {elem_desc}"
    env_change = action['env_change']
    if env_change:
        action_str += f"\nenv_change: {env_change}"

    if action['action_type'] == 'take_note':
        action_str += action['action_summary']
    if action['action_type'] == 'eval':
        subgoal_completed = action['meta_data']['subgoal_completed']
        action_str += f"\nSubgoal completed: {subgoal_completed}"

    logger.debug(f"{action_str}\n")

    return


def save_action(action: AgentAction) -> dict:
    """"""

    meta_data = action['meta_data']
    
    # Don't save Image
    meta_data['elem_crop'] = None

    action_element: Element = meta_data['element']
    if action_element:
        elem_dict = action_element.to_dict()
        meta_data['element'] = elem_dict
    
    action_str_list = []
    for subaction in action['actions']:
        action_str_list.append(subaction['action_summary'])
    action['actions'] = action_str_list

    return action






"""
self.saved_info = {
    "files": {
        "input_files": [],
        "saved_files": []
    },
    "saved_links": [],
    "notes": [],
    "required_info": {}
}
self.task_status = {
    "intent": None,
    "task_complete": False,
    "requirements": [],
    "completed_subgoals": [],
    "last_milestone": None,
    "current_subgoal": None,
}
"""


class SystemState(StateInfo):
    nth: int
    url: str
    observation: dict[str, Any] = {
        "html": None,
        "screenshot": None,
        "browser_tabs": [],
        "page_mem": None,
        "focused_elem": None,
        "scroll_height": None,

        "dialog": None,
        "clipboard": None,
    }

    # Agent
    saved_info: dict[str, Any]
    task_status: dict[str, Any]


def save_state(state: SystemState, output_file: str=None) -> dict:
    """"""

    try:
        full_obs = state['observation']
        step = state['nth']
        
        screenshot: Image = full_obs['screenshot']
        if screenshot and output_file:
            # TODO: save screenshot
            pass
        
        browser_tabs: list[Page] = full_obs['browser_tabs']
        browser_tab_urls = [page.url for page in browser_tabs]

        page_mem: PageMem = full_obs['page_mem']
        website_url = page_mem.website_url
        page_url = page_mem.url

        focused_elem: Element = full_obs['focused_elem']
        if focused_elem:
            foc_elem = focused_elem.to_dict()
        else:
            foc_elem = None

        observation_info = copy(full_obs)
        observation_info['html'] = ""
        observation_info['browser_tabs'] = browser_tab_urls
        observation_info['page_mem'] = {"website_url": website_url, "page_url": page_url}
        observation_info['focused_elem'] = foc_elem
        observation_info['screenshot'] = step

        state['observation'] = observation_info
    
    except Exception as e:
        logger.error(f"Error while saving trajectory state: {repr(e)}")
        state = {}

    return state


def save_trajectory(trajectory: Trajectory, output_file: str, config_file: str=None):
    """Save agent trajectory to file"""

    trajectory_dict = []

    for i in range(len(trajectory)):
        x = trajectory[i]
        if i % 2 == 0:
            serialized = save_state(x, output_file)
        else:
            serialized = save_action(x)
        trajectory_dict.append(serialized)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory_dict, f, indent=4)
    
    return