import json
import os
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Union

from loguru import logger
from PIL import Image
from playwright.sync_api import Locator

from webchallenger.utils import Singleton, BBox, url_to_name, is_state_class, clean_text
from webchallenger.visualwebarena import (
    Action,
    ActionTypes,
    StateInfo,
    Trajectory,
)




class Element:
    def __init__(self, saved_element_map: dict[str, Any] = None, load_mem: bool=True) -> None:
        self.tag = None
        self.bbox = None
        self.text = ""
        self.iframe_id = None
        
        # Attributes
        self.attributes_dict = dict()
        self.id = None
        self.data_testid = None
        self.data_qa = None
        self.data_value = None
        self.data_index = None
        self.data_track_property = None
        self.data_attributes: list[tuple[str, str]] = []
        self.role = None
        self.label = None
        self.aria_label = None
        self.aria_labelledby = None
        self.aria_controls = None
        self.aria_haspopup = None
        self.aria_autocomplete = None
        self.aria_hidden = None
        self.aria_invalid = None
        self.aria_expanded = None
        self.aria_selected = None
        self.open = None
        self.required = None
        self.is_disabled = False
        self.title = None
        self.name = None
        self.class_name = ""
        self.href = None
        self.placeholder = None


        self.parent_tag = None
        self.parent_class = ""

        self.input_type = None
        self.date_format = None
        

        self.has_suggest = None
        self.inner_html = None
        self.fieldset = None
        self.contained_in = None
        self.explored = False
        self.meta_data = None

        # Ignore element
        self.skip = False
        self.click_only = False
        # Tab iteration order on page
        self.index = None
        # Description of what happens when element is clicked
        self.func_desc = None
        self.vlm_info = ""
        self.more_info = ""
        self.table_row = None
        self.row_info = ""

        # List of options for <select> and <input> elements
        self.options = []
        self.multiple = None
        # List of elements contained inside dropdown menu
        self.dropdown_elements = []
        self.parent_menu = None
        self.tab_sections: list[PageSection] = []  # list of sections controlled by tab
        self.collapse = None
        self.dialog = False
        self.next_element = None
        self.iter_key = 'Tab'
        # Page after clicking element
        self.destination_page: str = None
        self.new_page: str = None

        # variable
        self.no_dropdown = False
        self.dropdown_updated = False
        self.nth = None
        self.section = None
        self.input_value = None
        self.input_edited = False
        self.input_truncated = False
        self.clicked = False

        # Load from json if provided
        if saved_element_map != None and load_mem:
            self.id = saved_element_map["id"]
            self.data_testid = saved_element_map["data_testid"]
            self.data_qa = saved_element_map["data_qa"]
            if "data_value" in saved_element_map:
                self.data_value = saved_element_map["data_value"]
            if "data_track_property" in saved_element_map:
                self.data_track_property = saved_element_map["data_track_property"]

            self.label = saved_element_map["label"]
            self.aria_label = saved_element_map["aria_label"]
            self.title = saved_element_map["title"]
            if "role" in saved_element_map:
                self.role = saved_element_map["role"]
            if "aria_haspopup" in saved_element_map:
                self.aria_haspopup = saved_element_map["aria_haspopup"]
            if "has_suggest" in saved_element_map:
                self.has_suggest = saved_element_map["has_suggest"]
            if "name" in saved_element_map:
                self.name = saved_element_map["name"]
            if "input_value" in saved_element_map:
                self.input_value = saved_element_map["input_value"]
            if "placeholder" in saved_element_map:
                self.placeholder = saved_element_map["placeholder"]
            
            self.class_name = saved_element_map["class_name"]
            self.text = saved_element_map["text"]
            self.href = saved_element_map["href"]
            self.tag = saved_element_map["tag"]

            saved_bbox = saved_element_map["bbox"]
            self.bbox = BBox(saved_bbox=saved_bbox)

            self.input_type = saved_element_map["input_type"]
            self.inner_html = saved_element_map["inner_html"]
            if "fieldset" in saved_element_map:
                self.fieldset = saved_element_map["fieldset"]
                self.contained_in = saved_element_map["contained_in"]
            if "explored" in saved_element_map:
                self.explored = saved_element_map["explored"]
            if "meta_data" in saved_element_map:
                self.meta_data = saved_element_map["meta_data"]
            
            if "collapse" in saved_element_map:
                for saved_section_map in saved_element_map["tab_sections"]:
                    self.tab_sections.append(PageSection(saved_section_map))
                self.collapse = saved_element_map["collapse"]
            if "parent_class" in saved_element_map:
                self.parent_tag = saved_element_map["parent_tag"]
                self.parent_class = saved_element_map["parent_class"]
                self.dialog = saved_element_map["dialog"]
            if "iframe_id" in saved_element_map:
                self.iframe_id = saved_element_map["iframe_id"]
            if "new_page" in saved_element_map:
                self.new_page = saved_element_map["new_page"]
            if "iter_key" in saved_element_map:
                self.iter_key = saved_element_map["iter_key"]

            self.skip = saved_element_map["skip"]
            self.click_only = saved_element_map["click_only"]
            self.index = saved_element_map["index"]
            
            self.func_desc = saved_element_map["func_desc"]
            self.options = saved_element_map["options"]

            for element_map in saved_element_map["dropdown_elements"]:
                self.dropdown_elements.append(Element(element_map))

            next_element_map = saved_element_map["next_element"]
            if next_element_map:
                self.next_element = Element(next_element_map)

            self.destination_page = saved_element_map["destination_page"]


    def to_dict(self, save_big=False, max_recurse=3) -> dict[str, Any]:
        """Returns Element serialized as a dict"""
        next_elem = None
        if self.next_element != None:
            next_elem = self.next_element.to_dict()
        
        bbox = None
        if self.bbox != None:
            bbox = self.bbox.to_dict()

        if max_recurse == 0:
            dropdown_elements = []
        else:
            dropdown_elements = [elem.to_dict(max_recurse=max_recurse-1) for elem in self.dropdown_elements]
        if (len(dropdown_elements) > 100) and (not save_big):
            dropdown_elements = []
            self.func_desc = "big_element"

        element_map = {
            "id": self.id,
            "label": self.label,
            "data_testid": self.data_testid,
            "data_qa": self.data_qa,
            "data_value": self.data_value,
            "data_track_property": self.data_track_property,
            "aria_label": self.aria_label,
            "role": self.role,
            "aria_haspopup": self.aria_haspopup,
            "has_suggest": self.has_suggest,
            "title": self.title,
            "name": self.name,
            "class_name": self.class_name,
            "parent_tag": self.parent_tag,
            "parent_class": self.parent_class,
            "iframe_id": self.iframe_id,
            "text": self.text,
            "bbox": bbox,
            "href": self.href,
            "tag": self.tag,
            "input_type": self.input_type,
            "input_value": self.input_value,
            "placeholder": self.placeholder,
            "inner_html": self.inner_html,
            "fieldset": self.fieldset,
            "contained_in": self.contained_in,
            "explored": self.explored,
            "meta_data": self.meta_data,
            "click_only": self.click_only,
            "skip": self.skip,
            "index": self.index,
            "func_desc": self.func_desc,
            "options": self.options,
            "dropdown_elements": dropdown_elements,
            "tab_sections": [section.to_dict() for section in self.tab_sections],
            "collapse": self.collapse,
            "dialog": self.dialog,
            "next_element": next_elem,
            "iter_key": self.iter_key,
            "destination_page": self.destination_page,
            "new_page": self.new_page
        }
        return element_map
    

    def print_element(self):
        """Print metadata of element"""
        element_str = f"{self.tag}, class={self.class_name}, id={self.id}"
        if self.name:
            element_str += f"\nname: {self.name}"
        element_str += f"\ntext: {self.text}"
        element_str += f"\nhref: {self.href}"
        element_str += f"\nparent: {self.parent_tag}, class={self.parent_class}"
        if self.input_type:
            element_str += f"\ninput_type: {self.input_type}"
        if self.options:
            element_str += f"\noptions: {len(self.options)}"
        if self.dropdown_elements:
            element_str += f"\ndropdown menu: {len(self.dropdown_elements)} items"
        if self.func_desc:
            element_str += f"\nfunc_desc: {self.func_desc}"
        if self.iframe_id:
            element_str += f"\niframe_id: {self.iframe_id}"

        logger.debug(f"{element_str}")
        if self.bbox:
            self.bbox.print_abs_page_coords()
    

    def get_str(self) -> str:
        """Return name of Element."""

        elem_text = ""
        if self.text:
            if len(self.text) < 500:
                elem_text = clean_text(self.text, max_lines=2)
                if self.tag == 'button':
                    if self.aria_label:
                        return self.aria_label
                    return elem_text

        if self.title:
            return self.title
        if self.aria_label:
            return self.aria_label
        if (self.name) and (len(self.name)>1):
            return self.name
        if self.placeholder:
            return self.placeholder
        if self.text:
            if len(self.text) < 500:
                return clean_text(self.text, max_lines=2)
        
        if self.func_desc:
            return self.func_desc
        
        if self.id:
            return self.id
        if self.data_testid:
            return self.data_testid
        if self.data_qa:
            return self.data_qa
        if self.class_name:
            return self.class_name

        return ""
    

    def get_name_label(self) -> str:
        """Return name of Element."""
        
        name = self.get_str()

        if (self.label):
            label = clean_text(self.label, max_words=25, max_lines=1)
            if label not in name:
                name = f"{label} - {name}"
        
        return name
    

    def get_name(self) -> str:
        """Return name of Element."""

        elem_text = ""
        if self.text:
            if len(self.text) < 500:
                elem_text = clean_text(self.text, max_lines=1)
                if (self.tag == 'button') or (self.role == 'button'):
                    if self.aria_label:
                        return self.aria_label
                    return elem_text
        
        if self.tag in ['input', 'textarea', 'select']:
            if (self.name == 'sPattern'):
                return 'Search query'
            if (self.name) and (any(x in self.name.lower() for x in ['min', 'max'])):
                return self.name
            if (self.label) and (len(self.label.strip()) > 3) and (not self.placeholder):
                return self.label

        if self.title:
            return self.title
        if self.aria_label:
            return self.aria_label
        if self.placeholder:
            return self.placeholder
        if (self.name) and (len(self.name)>1):
            return self.name
        
        if self.text:
            return clean_text(self.text, max_lines=2)
        if self.label:
            return self.label
        
        if self.func_desc:
            return self.func_desc
        

        if self.id:
            return self.id
        if self.data_testid:
            return self.data_testid
        if self.data_qa:
            return self.data_qa
        if (self.class_name) and (not self.href):
            return self.class_name

        return ""
    


    def is_text_input(self) -> bool:
        """Checks if element is a text input field."""
        if self.tag == 'input':
            if self.input_type in ['text', 'email', 'password', 'search', 'tel', 'url']:
                return True
        elif self.tag == 'textarea':
            return True
        return False
    

    def is_skip(self) -> bool:
        """Checks if element is a 'Skip to main/content' accessibility option."""
        
        if (self.tag=='a') and (self.bbox.get_abs_px_width() == 1):
            return True
        
        skip_text = False
        if self.text:
            if 'skip ' in self.text.lower(): skip_text=True
            if 'jump ' in self.text.lower(): skip_text=True
            if 'main ' in self.text.lower(): skip_text=True
            if ' content' in self.text.lower(): skip_text=True
        
        skip_href = False
        if self.href:
            if (self.href[0] == '#') or (self.href[0] == 'j'):
                skip_href=True

        if skip_text and skip_href:
            return True
        else:
            return False
    

    def class_equals(self, other_element, parent_class=False) -> bool:
        """Returns True if both elements have the same non-dynamic classes."""
        
        if parent_class:
            class_str = self.parent_class
            other_class = other_element.parent_class
        else:
            class_str = self.class_name
            other_class = other_element.class_name

        if class_str == other_class:
            return True

        classes = [] if class_str == None else class_str.split()
        other_classes = [] if other_class == None else other_class.split()

        diff_classes = list(set(classes) ^ set(other_classes))

        for c in diff_classes:
            if not is_state_class(c):
                return False

        return True

    
    def equals(self, other_element, class_only=False, exact=False) -> bool:
        """Check if the other element is the same"""

        if other_element == None:
            return False

        if self.tag != other_element.tag:
            return False
        if (self.tag == 'input') and ((self.label != other_element.label) or (self.id != other_element.id)):
            return False
        if (self.parent_tag and other_element.parent_tag) and (self.parent_tag != other_element.parent_tag):
            return False
        if (self.parent_class and other_element.parent_class) and (not self.class_equals(other_element, parent_class=True)):
            return False

        if class_only:
            if self.class_equals(other_element):
                return True
        if (self.class_equals(other_element)) and ((self.id) and (self.id==other_element.id)) and (self.get_name() == other_element.get_name()):
            if (self.aria_label) and (self.aria_label == other_element.aria_label):
                return True
            if (self.role) and (self.role == other_element.role):
                return True
        
        if (not exact) and (self.bbox and other_element.bbox) and (self.bbox.get_abs_px_coords() == other_element.bbox.get_abs_px_coords()):
            if ((self.role) and (self.role==other_element.role)) or ((self.aria_label) and (self.aria_label==other_element.aria_label)):
                if (self.id != None) and (self.id == other_element.id):
                    return True
                if (self.text != None) and (self.text == other_element.text):
                    return True
                if (self.class_name != None) and (other_element.class_name != None):
                    classes = self.class_name.split(" ")
                    other_classes = other_element.class_name.split(" ")
                    for i in range(min(len(classes), len(other_classes))):
                        if classes[i] != other_classes[i]: return False
                    return True


        if self.data_testid != other_element.data_testid:
            return False
        if self.data_qa != other_element.data_qa:
            return False
        if self.data_value != other_element.data_value:
            return False
        if self.data_track_property != other_element.data_track_property:
            return False
        if self.aria_label != other_element.aria_label:
            return False
        if (self.title and other_element.title) and (self.title != other_element.title):
            return False
        if self.name != other_element.name:
            return False
        if not self.class_equals(other_element):
            return False

        if self.text != other_element.text:
            return False
        if self.href != other_element.href:
            return False
        
        return True
    

    def in_list(self, element_list, class_only=False, check_coord=False, exact=False):
        """Returns True if element is contained in the list."""
        
        if not element_list:
            return None

        for e in element_list:
            if self.equals(e, class_only, exact=exact):
                return e
            if (not self.bbox) or (not e.bbox):
                continue
            if (check_coord) and (self.bbox.center_in_bbox(e.bbox) or (e.bbox.center_in_bbox(self.bbox))):
                action_tags = ['input', 'textarea', 'select', 'label', 'button', 'a']
                if (e.tag in action_tags) or (self.tag in action_tags):
                    continue
                if (abs(self.bbox.get_abs_px_width() - e.bbox.get_abs_px_width()) < 10) and (abs(self.bbox.get_abs_px_height() - e.bbox.get_abs_px_height()) < 10):
                    return e

        return None


class PageSection:
    def __init__(self, saved_section_map: dict[str, Any] = None, load_mem: bool=True) -> None:
        self.desc = None  # full VLM description of section
        self.summary = None  # summary of section by LLM
        self.bbox = None  # defines absolute pixel coordinates on the page
        self.screenshot_file = None
        self.image_captions = {}

        # for sections that contain a long list of elements that should be traversed
        # using a list iteration function.
        self.list_type = None  # 'list' OR 'grid' OR 'table'
        self.row_length = 1
        self.explored_end = False
        self.has_images = False
        self.toggle = False

        # Section html element locator
        self.class_name = ""
        self.id = None
        self.data_index = None
        self.parent_tag = None
        self.parent_class = ""
        self.iframe_id = None
        self.role = None
        self.type = None  # 'div', 'form', 'ol', 'ul', 'list', 'table', 'article', 'p', 'details'
        self.list_item_tag = None  # html tag of list item elements
        self.li_class = ""
        
        # for locating list items
        self.list_section_tag = None
        self.list_section_class = None
        self.data_level = None
        self.nth = None

        # Elements in section (for list/table sections: actions for each entry)
        self.elements = []
        self.n_dropdowns = 0
        self.item_actions_template = []
        self.list_actions = ""

        # variable
        self.is_dialog = False
        self.is_form = False
        self.done_form = False
        self.updated = None
        self.is_primary_section = False
        self.in_main = False
        self.not_relevant = None
        self.reselect_items = False
        self.n_items = 0
        self.section_plan = ""
        self.input_modified = False
        self.elem_changes = 0
        self.table_filters = []
        self.column_labels = []
        self.screenshot = None
        self.inner_html = ""
        self.section_text = ""
        self.images = []
        self.sort_order = ""
        self.list_items_url = ""
        self.list_items_dict = {}
        self.list_item_type = ""
        self.list_item_attributes = []
        self.relevant_attributes = []
        self.extracted_attr_info = ""
        self.list_items = []  # matching items
        self.relevant_item_indices = []
        self.relevant_items_str = ""
        self.chart_info = ""
        self.task_details = ""
        self.task_summary = ""
        self.task_elems = []
        

        # Load from json if provided
        if saved_section_map != None and load_mem:
            self.desc = saved_section_map["desc"]
            self.summary = saved_section_map["summary"]
            self.screenshot_file = saved_section_map["screenshot_file"]
            if "image_captions" in saved_section_map:
                self.image_captions = saved_section_map["image_captions"]
            self.list_type = saved_section_map["list_type"]
            self.explored_end = saved_section_map["explored_end"]
            self.has_images = saved_section_map["has_images"]

            if "toggle" in saved_section_map:
                self.toggle = saved_section_map["toggle"]

            if "class_name" in saved_section_map:
                self.class_name = saved_section_map["class_name"]
                self.id = saved_section_map["id"]
                self.parent_tag = saved_section_map["parent_tag"]
                self.parent_class = saved_section_map["parent_class"]
                self.list_item_tag = saved_section_map["list_item_tag"]
                self.type = saved_section_map["type"]
            if "role" in saved_section_map:
                self.role = saved_section_map["role"]
            if "iframe_id" in saved_section_map:
                self.iframe_id = saved_section_map["iframe_id"]
            if "data_level" in saved_section_map:
                self.data_level = saved_section_map["data_level"]
            if "data_index" in saved_section_map:
                self.data_index = saved_section_map["data_index"]

            if "elements" in saved_section_map:
                for saved_element_map in saved_section_map["elements"]:
                    self.elements.append(Element(saved_element_map))
            if "n_dropdowns" in saved_section_map:
                self.n_dropdowns = saved_section_map["n_dropdowns"]
                self.list_actions = saved_section_map["list_actions"]
            
            saved_bbox = saved_section_map["bbox"]
            self.bbox = BBox(saved_bbox=saved_bbox)
    

    def to_dict(self) -> dict[str, Any]:
        """Returns PageSection serialized as a dict"""
        section_map = {
            "desc": self.desc,
            "summary": self.summary,
            "screenshot_file": self.screenshot_file,
            "image_captions": self.image_captions,
            "list_type": self.list_type,
            "explored_end": self.explored_end,
            "has_images": self.has_images,
            "toggle": self.toggle,
            "class_name": self.class_name,
            "id": self.id,
            "parent_tag": self.parent_tag,
            "parent_class": self.parent_class,
            "iframe_id": self.iframe_id,
            "list_item_tag": self.list_item_tag,
            "data_level": self.data_level,
            "data_index": self.data_index,
            "role": self.role,
            "type": self.type,
            "nth": self.nth,
            "bbox": self.bbox.to_dict(),
            "elements": [elem.to_dict() for elem in self.elements],
            "n_dropdowns": self.n_dropdowns,
            "list_actions": self.list_actions
        }
        return section_map
    

    def add_element(self, element: Element):
        """Add Element to section."""
        if not element.in_list(self.elements):
            if element.dropdown_elements:
                self.n_dropdowns += 1
            self.elements.append(element)
        else:
            print("Element already added to section")

    
    def print_section(self):
        """"""
        element_str = f'{self.type}:'
        if self.list_item_tag:
            element_str += f' [{self.list_item_tag}]'
        if self.class_name:
            element_str += f' class="{self.class_name}"'
        if self.id:
            element_str += f' id="{self.id}"'
        
        if self.bbox:
            element_str += f'\nbbox: {self.bbox.get_abs_px_coords()}'
        else:
            element_str += f'\nbbox: {self.bbox}'
        if self.elements:
            element_str += f'\nelements: {len(self.elements)}'
        if self.iframe_id:
            element_str += f"\niframe_id: {self.iframe_id}"

        logger.debug(f'{element_str}')
        return
    

    def equals(self, other_section) -> bool:
        """Check if the other section is the same"""

        if other_section == None:
            return False

        if self.class_name != other_section.class_name:
            return False
        if self.id != other_section.id:
            return False
        if self.data_index != other_section.data_index:
            return False
        if self.parent_tag != other_section.parent_tag:
            return False
        if self.parent_class != other_section.parent_class:
            return False
        if self.iframe_id != other_section.iframe_id:
            return False
        if self.role != other_section.role:
            return False
        if self.type != other_section.type:
            return False

        return True

    
    def copy_section(self):
        """"""

        copied_section = copy(self)
        elem_list = []
        for elem in self.elements:
            elem_list.append(copy(elem))
        copied_section.elements = elem_list

        return copied_section


class PageMem:
    def __init__(self, saved_page_map: dict[str, Any] = None, load_mem: bool=True) -> None:
        self.url = None
        self.website_url = None
        self.screenshot_dir = None  # directory where page screenshots are saved

        # v3
        self.html_sections: list[PageSection] = []
        self.list_section: PageSection = None
        self.n_missing_sections = 0  # section_loc errors upon reload
        self.missing_sections: list[PageSection] = []
        self.empty_sections = 0  # no elements
        
        self.is_list_item = False
        self.similar_urls = []
        self.list_page = None
        self.name = ""
        self.vlm_desc = ""
        self.page_summary = ""
        self.short_summary = ""
        self.add_bookmark = True

        # variable
        self.table_subgoal = ""
        self.table_need_info = False
        self.filter_table = False
        self.filled_form = False
        self.page_height = 1280
        self.html = ""
        self.dom_snapshot = ""
        self.alerts = []
        self.similar = None
        self.dialog = None
        self.dialog_section = None
        self.dialog_field_edited = None
        self.popup_info = None
        self.browser_dialog = None
        self.task_sections = []
        self.task_summary = ""
        self.task_info = ""
        self.retrieved_history = False
        self.list_info = ""
        self.relevant_actions = []
        
        # (maybe not here)
        self.bookmarks = []
        self.bookmark_info = ""


        # Elements that are not contained in a section bbox
        self.outer_elements = []

        # Load from json if provided
        if saved_page_map != None and load_mem:
            self.url = saved_page_map["url"]
            self.website_url = saved_page_map["website_url"]
            self.screenshot_dir = saved_page_map["screenshot_dir"]
            self.name = saved_page_map["name"]
            self.page_summary = saved_page_map["page_summary"]
            self.short_summary = saved_page_map["short_summary"]
            if "add_bookmark" in saved_page_map:
                self.add_bookmark = saved_page_map["add_bookmark"]

            if "html_sections" in saved_page_map:
                for saved_section_map in saved_page_map["html_sections"]:
                    self.html_sections.append(PageSection(saved_section_map))

            if "outer_elements" in saved_page_map:
                for saved_element_map in saved_page_map["outer_elements"]:
                    self.outer_elements.append(Element(saved_element_map))

            if "list_section" in saved_page_map:
                list_section_map = saved_page_map["list_section"]
                if list_section_map:
                    self.list_section = PageSection(saved_page_map["list_section"])
            if "is_list_item" in saved_page_map:
                self.is_list_item = saved_page_map["is_list_item"]
            if "similar_urls" in saved_page_map:
                self.similar_urls = saved_page_map["similar_urls"]
            if "list_page" in saved_page_map:
                self.list_page = saved_page_map["list_page"]


    def to_dict(self) -> dict[str, Any]:
        """Returns PageMem serialized as a dict"""
        list_section = None
        if self.list_section != None:
            list_section = self.list_section.to_dict()
        
        page_map = {
            "url": self.url,
            "website_url": self.website_url,
            "screenshot_dir": self.screenshot_dir,
            "name": self.name,
            "page_summary": self.page_summary,
            "short_summary": self.short_summary,
            "add_bookmark": self.add_bookmark,
            "outer_elements": [elem.to_dict() for elem in self.outer_elements],
            "html_sections": [section.to_dict() for section in self.html_sections],
            "list_section": list_section,
            "is_list_item": self.is_list_item,
            "similar_urls": self.similar_urls,
            "list_page": self.list_page
            # "extra_links": self.extra_links
        }
        return page_map

    
    def add_section(self, section: PageSection):
        """Add a PageSection object to the page memory."""

        self.html_sections.append(section)


class WebsiteMem:
    """Memory database of a website. Initially constructed by explore_page_v3 in agent.py"""

    def __init__(self, saved_website_map: dict[str, Any] = None, load_mem: bool=True) -> None:
        self.url = None
        self.pages = {}  # Maps from url to corresponding PageMem obj
        self.page_visits = {}  # increments once for each agent trajectory that visits the url

        # list page types
        self.list_pages = []

        # Set of all Element objects encountered on website during exploration
        self.all_elements: list[Element] = []
        # Save Elements that take up lots of memory (many dropdown items) so we don't have to repeat
        self.big_elements: list[Element] = []
        # Error occured when exploring element
        self.error_elements = []
        self.error_pages = []

        # Load from json if provided
        if saved_website_map != None and load_mem:
            self.url = saved_website_map["url"]
            
            # initialize pages
            saved_pages = saved_website_map["pages"]
            for saved_page_url in list(saved_pages.keys()):
                saved_page_map = saved_pages[saved_page_url]
                self.pages[saved_page_url] = PageMem(saved_page_map)
            for saved_list_map in saved_website_map["list_pages"]:
                self.list_pages.append(PageMem(saved_list_map))
            
            # load all_elements
            if "all_elements" in saved_website_map:
                for saved_element_map in saved_website_map["all_elements"]:
                    self.all_elements.append(Element(saved_element_map))
            if "big_elements" in saved_website_map:
                for saved_element_map in saved_website_map["big_elements"]:
                    self.big_elements.append(Element(saved_element_map))
            if "error_elements" in saved_website_map:
                for saved_element_map in saved_website_map["error_elements"]:
                    self.error_elements.append(Element(saved_element_map))
                self.error_pages = saved_website_map["error_pages"]
            if "page_visits" in saved_website_map:
                self.page_visits = saved_website_map["page_visits"]

    
    def to_dict(self) -> dict[str, Any]:
        """Returns WebsiteMem serialized as a dict"""
        # Convert PageMem objects to dictionaries
        pages_map = {}
        for page_url in list(self.pages.keys()):
            pages_map[page_url] = self.pages[page_url].to_dict()
        
        list_page_maps = []
        for list_page in self.list_pages:
            list_page_map = list_page.to_dict()
            list_page_maps.append(list_page_map)

        website_map = {
            "url": self.url,
            "pages": pages_map,
            "list_pages": list_page_maps,
            "all_elements": [elem.to_dict() for elem in self.all_elements],
            "big_elements": [elem.to_dict(save_big=True) for elem in self.big_elements],
            "error_elements": [elem.to_dict() for elem in self.error_elements],
            "error_pages": self.error_pages,
            "page_visits": self.page_visits
        }
        return website_map
    

    def set_page_mem(self, page_mem: PageMem):
        """Add a new page memory to the website memory."""
        self.pages[page_mem.url] = page_mem


class EnvMemory(metaclass=Singleton):
    """Long-term agent memory database that saves a structured knowledge-graph for every
    website visited.
    Agent can reference and update the database.
    """

    def __init__(self, mem_dir: str = f'{os.path.dirname(__file__)}', load_mem: bool=True):
        self.mem_dir = mem_dir
        self.reset()
    

    def reset(self):
        self.websites = dict()
        self.accounts = dict()

        # Load user account login details
        logger.info(f"LOADING ACCOUNT LOGIN INFO...")
        with open(os.path.join(f'{os.path.dirname(__file__)}', 'accounts.json')) as acc_file:
            self.accounts = json.load(acc_file)

        # Initialize website knowledge graphs from saved json files
        logger.info(f"LOADING AGENT MEMORY...")
        mem_files = os.listdir(self.mem_dir)
        for save_file in mem_files:
            if save_file.endswith('.json'):
                with open(os.path.join(self.mem_dir, save_file)) as f:
                    saved_website_map = json.load(f)
                    url = saved_website_map["url"]
                    website_mem = WebsiteMem(saved_website_map, True)
                    self.websites[url] = website_mem
                    if len(mem_files) < 50:
                        logger.info(f"MEMORY: loaded {url_to_name(url)}")

        logger.info(f"Memory initialized\n")


    def save_website_mem(self, website_mem: WebsiteMem, replace = False) -> None:
        """Adds a new WebsiteMem object to the websites map of the agent's memory
        and saves the website in a json file."""
        
        website_url = website_mem.url
        # Don't overwrite existing website if replace is set to False
        if website_url in self.websites:
            if replace == False:
                logger.error(f"Website {website_url} already exists.")
                return
        
        # add website_mem object to memory dictionary
        self.websites[website_url] = website_mem

        # Serialize WebsiteMem as dictionary
        website_map = website_mem.to_dict()

        # Save website json file
        try:
            os.makedirs(f"{self.mem_dir}", exist_ok=True)
        except OSError as error:
            logger.error(f"An error occurred while creating the directory: {error}")
        
        json_file = f"{url_to_name(website_mem.url)}.json"
        json_path = os.path.join(f"{self.mem_dir}", json_file)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(website_map, f, indent=4)



