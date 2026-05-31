import argparse
import copy
import os
import re
import sys
import threading
from typing import Any, List, Tuple, Optional
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup, Tag
from loguru import logger
from PIL import Image, ImageChops, ImageDraw, ImageFont

from webchallenger.visualwebarena import Action


# max height and width value for BBox
MAX = 1000
# VisualWebArena urls
ENV_LIST = ['CLASSIFIEDS', 'SHOPPING', 'REDDIT', 'WIKIPEDIA', 'HOMEPAGE']
# # WebArena urls
ENV_LIST += ['SHOPPING_ADMIN', 'GITLAB', 'MAP']


SHOPPING = os.environ.get("SHOPPING", "")
if not SHOPPING:
    ENV_LIST = []


lock = threading.Lock()
class Singleton(type):
    _instances = {}

    # thread-safe singleton
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(
                        Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Trajectory: [Obs, Action, Obs, Action, Obs, FinalAction]
# ActionInfo: [Obs, Fail, Fail, Success, Obs, Success, Obs, Success]
class ActionInfo:
    def __init__(
            self,
            prompt: list[dict[str, str]],
            output: str,
            success: bool,
            error_msg: str = None
        ):
        self.prompt = self._format_prompt(prompt)
        self.output = output
        self.success = success
        self.error_msg = error_msg

    def _format_prompt(self, prompt):
        """Convert prompt system and user dicts into combined string."""
        sys_prompt = prompt[0]["content"]
        user_prompt = prompt[1]["content"]
        prompt_str = f"{sys_prompt}\n{user_prompt}"
        return prompt_str
    
    def to_dict(self) -> dict[str, Any]:
        action_info_map = {
            "prompt": self.prompt,
            "output": self.output,
            "success": self.success,
            "error_msg": self.error_msg
        }
        return action_info_map
    

def save_action_history(action_hist: list[ActionInfo], path: str):
    """Serialize list of agent action execution outputs as dict and save it in
    the path as a json file."""
    # TODO:
    return




class BBox:
    def __init__(self, x1=0, y1=0, x2=MAX, y2=MAX, saved_bbox: dict[str, Any] = None):
        # fix coordinate outliers
        if y1 < 0 and y2 < 0:
            y1 = 0
            y2 = y2 - y1 if y2 > y1 else 0
        else:
            self.y1 = self._check_and_round(y1)
            self.y2 = self._check_and_round(y2)
        x1 = self._check_and_round(x1)
        x2 = self._check_and_round(x2)

        # Coordinates relative to screenshot at 0-1000 scale
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # Region that the BBox is inside of.
        self.parent_bbox = None
        # Absolute pixel coordinates of BBox on browser page
        self.x1_abs_px = 0
        self.y1_abs_px = 0
        self.x2_abs_px = 1280
        self.y2_abs_px = 960

        if saved_bbox != None:
            self.x1 = saved_bbox["x1"]
            self.y1 = saved_bbox["y1"]
            self.x2 = saved_bbox["x2"]
            self.y2 = saved_bbox["y2"]
            self.x1_abs_px = saved_bbox["x1_abs_px"]
            self.y1_abs_px = saved_bbox["y1_abs_px"]
            self.x2_abs_px = saved_bbox["x2_abs_px"]
            self.y2_abs_px = saved_bbox["y2_abs_px"]

    
    @classmethod
    def from_string(cls, coord_str):
        """Constructor that returns BBox from VLM coord output string 
        Args:
            coord_str: '[[x1,y1,x2,y2]]' format
        """

        no_brackets = coord_str.strip('[[]]')
        coords = no_brackets.split(',')

        return cls(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    

    @classmethod
    def from_playwright_bbox(cls, bounding_box: dict[str, int], scroll_height: int = 0):
        """Converts bounding box returned by Locator.bounding_box() into BBox."""

        if not bounding_box:
            return None

        bbox = cls()
        bbox.x1_abs_px = int(bounding_box['x'])
        bbox.y1_abs_px = int(bounding_box['y'] + scroll_height)
        bbox.x2_abs_px = int(bounding_box['x'] + bounding_box['width'])
        bbox.y2_abs_px = int(bounding_box['y'] + bounding_box['height'] + scroll_height)

        return bbox


    def _check_and_round(self, value):
        """Ensures the coordinate is within the range [0, MAX]."""
        return max(0, min(MAX, int(value)))
    
    
    def copy(self):
        """Return a copy of the BBox."""
        return copy.copy(self)
    

    def set_parent(self, parent_bbox):
        """Sets the parent and scales coordinates to match the parent scale.
        E.g. If parent bbox has width 500-1000 and child bbox has width
        0-500, then child bbox is scaled to 500-750."""

        if not isinstance(parent_bbox, BBox):
            raise TypeError("parent_bbox must be an instance of BBox")
        self.parent_bbox = parent_bbox

        # Scale coordinates with respect to parent
        parent_width_ratio = self.parent_bbox.get_width() / MAX
        self.x1 = self.parent_bbox.x1 + int(self.x1 * parent_width_ratio)
        self.x2 = self.parent_bbox.x1 + int(self.x2 * parent_width_ratio)
        parent_height_ratio = self.parent_bbox.get_height() / MAX
        self.y1 = self.parent_bbox.y1 + int(self.y1 * parent_height_ratio)
        self.y2 = self.parent_bbox.y1 + int(self.y2 * parent_height_ratio)


    def get_image_px_coords(self, image_width, image_height) -> tuple[int, int, int, int]:
        """Get bbox coordinates scaled from 0-1000 to image pixel dimensions.
        Returns: x1, y1, x2, y2"""
        scale_x = image_width / MAX
        scale_y = image_height / MAX
        return (
            int(self.x1 * scale_x),
            int(self.y1 * scale_y),
            int(self.x2 * scale_x),
            int(self.y2 * scale_y)
        )


    def set_abs_page_coords(self, screen_width_px, screen_height_px, scroll_height_px):
        """Set absolute pixel coordinates of bbox relative to entire browser page."""
        scaled_coords = self.get_image_px_coords(screen_width_px, screen_height_px)
        self.x1_abs_px = scaled_coords[0]
        self.x2_abs_px = scaled_coords[2]
        self.y1_abs_px = scaled_coords[1] + scroll_height_px
        self.y2_abs_px = scaled_coords[3] + scroll_height_px
    
    
    def set_abs_subs_coords(self, parent_section):
        """Set absolute pixel coordinates of bbox based on pixel coords of it's parent."""
        section_width = parent_section.get_abs_px_width()
        section_height = parent_section.get_abs_px_height()

        self.x1_abs_px = parent_section.x1_abs_px + int((self.x1/MAX) * section_width)
        self.x2_abs_px = parent_section.x1_abs_px + int((self.x2/MAX) * section_width)
        self.y1_abs_px = parent_section.y1_abs_px + int((self.y1/MAX) * section_height)
        self.y2_abs_px = parent_section.y1_abs_px + int((self.y2/MAX) * section_height)


    def shift_bbox_right(self, shift_px: int):
        """Shift the bbox right."""
        self.x1_abs_px += shift_px
        self.x2_abs_px += shift_px

    
    def shift_bbox_down(self, shift_px: int, max=64000):
        """Shift the bbox down."""
        self.y1_abs_px += shift_px
        self.y2_abs_px += min(shift_px, max - self.y2_abs_px)

    
    def get_abs_px_coords(self):
        """Returns the absolute bbox coordinates relative to entire page."""
        return (
            self.x1_abs_px,
            self.y1_abs_px,
            self.x2_abs_px,
            self.y2_abs_px
        )
    

    def get_abs_px_str(self):
        """Returns bbox string with absolute coordinates."""
        return f"[[{self.x1_abs_px},{self.y1_abs_px},{self.x2_abs_px},{self.y2_abs_px}]]"


    def get_abs_px_height(self):
        """Returns the height of the bounding box in pixels."""
        return self.y2_abs_px - self.y1_abs_px
    

    def get_abs_px_width(self):
        """Retruns the wdith of the bounding box in pixels."""
        return self.x2_abs_px - self.x1_abs_px


    def print_abs_page_coords(self):
        """Prints absolute pixel coordinates relative to page."""
        logger.debug("[[{},{},{},{}]]\n", self.x1_abs_px, self.y1_abs_px, self.x2_abs_px, self.y2_abs_px)


    def get_width(self):
        """Returns the width of the bounding box."""
        return self.x2 - self.x1


    def get_height(self):
        """Returns the height of the bounding box."""
        return self.y2 - self.y1
    

    def get_center_xy(self):
        """Returns the center coordinates of the bbox, rounded down"""
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return center_x, center_y
    

    def get_center_xy_abs(self):
        """Returns the absolute center coordinates of the bbox"""
        center_x = (self.x1_abs_px + self.x2_abs_px) // 2
        center_y = (self.y1_abs_px + self.y2_abs_px) // 2
        return center_x, center_y
    

    def not_in_view(self) -> bool:
        """Returns True if element bbox is outside viewport."""
        if (self.x1_abs_px < 0) and (self.x2_abs_px < 0):
            return True
        if (self.y1_abs_px < 0) and (self.y2_abs_px < 0):
            return True

        return False


    def matches_x_coords(self, other_bbox, threshold=10) -> bool:
        """Checks whether or not the bbox has the same x coords as another bbox within threshold."""
        if abs(self.x1 - other_bbox.x1) <= threshold and abs(self.x2 - other_bbox.x2) <= threshold:
            return True
        return False


    def contained_in_x(self, other_bbox, threshold=50) -> bool:
        """Checks if x coordinates of the bbox overlap."""
        if max(self.x1, other_bbox.x1) <= min(self.x2, other_bbox.x2) - threshold:
            return True
        return False
    

    def center_in_bbox(self, other_bbox) -> bool:
        """Checks if center of current bbox is contained in other_bbox."""
        if not other_bbox:
            return False
        x, y = self.get_center_xy_abs()
        in_x = False
        in_y = False
        if x > other_bbox.x1_abs_px and x < other_bbox.x2_abs_px:
            in_x = True
        if y > other_bbox.y1_abs_px and y < other_bbox.y2_abs_px:
            in_y = True
        
        if in_x and in_y:
            return True
        return False


    def print_bbox(self):
        """Prints coordinates using 0-1000 scale"""
        logger.debug("[[{},{},{},{}]]", self.x1, self.y1, self.x2, self.y2)
        

    def to_dict(self) -> dict[str, Any]:
        """Returns BBox serialized as a dict"""
        bbox_map = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "x1_abs_px": self.x1_abs_px,
            "y1_abs_px": self.y1_abs_px,
            "x2_abs_px": self.x2_abs_px,
            "y2_abs_px": self.y2_abs_px
        }
        return bbox_map




#### ---- Image manipulation ---- ####

def crop_img(screenshot: Image, bbox: BBox=None, abs=False, add_margin=False, top=0) -> Image:
    """Returns cropped screenshot"""

    if not screenshot:
        logger.warning(f"screenshot=None")
        return None
    
    width, height = screenshot.size
    if bbox is None:
        bbox = BBox(0, top, width, height)
    
    if abs:
        # Get absolute bbox coordinates
        x1, y1, x2, y2 = bbox.get_abs_px_coords()
        if add_margin:
            x1 -= 25
            y1 -= 40
            x2 += 25
            y2 += 10
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, height)
    else:
        # Get coordinates relative to current image
        x1, y1, x2, y2 = bbox.get_image_px_coords(width, height)

    if (x2 <= x1) or (y2 <= y1):
        logger.warning(f"crop coords invalid: {(x1, y1, x2, y2)}")
        return None
    
    # Crop the image and return
    return screenshot.crop((x1, y1, x2, y2))


def draw_outlined_text(draw, text, position, font, outline_color, fill_color):
    """Function to draw outlined text"""
    x, y = position
    OL = 2

    # Draw the outline
    draw.text((x-OL, y-OL), text, font=font, fill=outline_color)
    draw.text((x+OL, y-OL), text, font=font, fill=outline_color)
    draw.text((x-OL, y+OL), text, font=font, fill=outline_color)
    draw.text((x+OL, y+OL), text, font=font, fill=outline_color)
    # Draw the fill
    draw.text((x, y), text, font=font, fill=fill_color)


def stack_images(img_1: Image, img_2: Image) -> Image:
    """Stacks img_1 on top of img_2 and returns the combined image.
    Both images must be same size"""

    width, height = img_1.width, img_1.height

    if img_1.size != img_2.size:
        logger.warning("Input image dimensions don't match, resizing second")
        img_2 = img_2.resize((width, height))

    # Create a new image with the combined height and the same width
    combined_height = height * 2
    new_img = Image.new('RGB', (width, combined_height))

    # Paste img_1 at the top
    new_img.paste(img_1, (0, 0))
    # Paste img_2 directly below img_1
    new_img.paste(img_2, (0, height))
    
    # Annotate combined image for comparison
    draw = ImageDraw.Draw(new_img)
    draw.line([(0, height), (width, height)], fill='red', width=5)
    # Label top image
    M = 10
    font = ImageFont.load_default(size=60)
    draw_outlined_text(draw, "A", (0+M, 0+M), font, outline_color='black', fill_color='white')
    # Label bottom image
    draw_outlined_text(draw, "B", (0+M, height+M), font, outline_color='black', fill_color='white')

    return new_img


def images_identical(img_1: Image, img_2: Image, threshold_px: int = 50) -> bool:
    """Returns true if img_1 and img_2 are the same.
    threshold_px determines max number of different pixels between two images."""

    if (not img_1) or (not img_2):
        logger.warning(f"One of the images is 'None'")
        return False

    diff = ImageChops.difference(img_1, img_2)

    diff_pixels = len(set(diff.getdata()))
    logger.info(f"pixel diff count: {diff_pixels}")

    if diff_pixels < threshold_px:
        return True

    return False


def draw_bbox_on_image(image: Image, bbox: BBox, index: int = None, abs=False, add_margin=False) -> Image:
    """
    Draws a red bounding box on an image.

    Args: 
        image: PIL Image object
        bbox: BBox object with coordinates scaled from 0 to 1000
    
    Returns:
        Image: PIL Image with drawn bounding box
    """
    
    if (not image) or (not bbox):
        return image

    width, height = image.size

    if abs:
        # Get absolute bbox coordinates
        x1, y1, x2, y2 = bbox.get_abs_px_coords()
    else:
        # Get coordinates relative to current image
        x1, y1, x2, y2 = bbox.get_image_px_coords(width, height)
    
    if add_margin:
        x1 -= 5
        y1 -= 5
        x2 += 5
        y2 += 5
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    if (y2 < y1) or (x2 < x1):
        # logger.warning(f"Can't draw rectangle {x1, y1, x2, y2}")
        return image

    # Draw bbox rectangle
    draw = ImageDraw.Draw(image)
    draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
    # If index provided, draw the number at the center of the bbox
    if index:
        font = ImageFont.load_default(size=40)
        draw.text(((x1+x2)//2, (y1+y2)//2), str(index), font=font, fill='red')

    return image


def draw_point_on_image(image: Image, x: float, y: float, index=None) -> Image:
    """Draws dot on point location. x, y are 0-1 scale."""

    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Get image dimensions
    width, height = image.size
    
    # Convert x, y from 0-1 scale to pixel coordinates
    point_x = int(x * width)
    point_y = int(y * height)
    
    # Define the point radius (adjustable for visibility)
    radius = 5
    
    # Draw dot
    draw.ellipse(
        [(point_x-radius, point_y-radius), (point_x+radius, point_y+radius)],
        fill=(240, 8, 252)
    )
    # If index provided, draw the number to the right of point
    if index:
        font = ImageFont.load_default(size=15)
        draw.text((point_x+10, point_y-10), str(index), font=font, fill=(240, 8, 252))

    return image




#### ---- URL parsing ---- ####

def get_website_url(url):
    """Get the url of the current website."""
    if url is None:
        return None
    
    site_url = ""

    for env_name in ENV_LIST:
        env_url = os.environ.get(env_name)
        # if url in env_url or env_url in url:
        if urlparse(url).netloc == urlparse(env_url).netloc:
            site_url =  env_url
            if env_name != "WIKIPEDIA":
                site_url += '/'
    
    if (not site_url):
        site_url = urlparse(url).netloc

    return site_url


def convert_local_url(url):
    """Converts local address of WebArena site into more readable name."""
    parsed_url = urlparse(url)
    local_hostname = parsed_url.netloc
    env_name = url_to_name(local_hostname)
    path = parsed_url.path
    return env_name + path


def get_full_url(url, page_url):
    """If url is relative, return full url."""

    if url.startswith('http'):
        return url

    absolute_url = urljoin(page_url, url)

    return absolute_url


def get_sim_url(url: str) -> str:
    """Convert localhost address into hostname of simulated site."""

    if not ENV_LIST:
        return url

    if os.environ.get('SHOPPING') in url:
        sim_url = url.replace(os.environ.get('SHOPPING'), "https://www.onestopmarket.com")
        return sim_url
    if os.environ.get('CLASSIFIEDS') in url:
        sim_url = url.replace(os.environ.get('CLASSIFIEDS'), "https://osclass-classifieds.com")
        return sim_url
    if "/f/" in url:
        url = url.replace("/f/", "/r/")
    if os.environ.get('REDDIT') in url:
        sim_url = url.replace(os.environ.get('REDDIT'), "https://www.reddit.com")
        return sim_url
    if os.environ.get('WIKIPEDIA') in url:
        sim_url = url.replace(os.environ.get('WIKIPEDIA'), "https://en.wikipedia.org")
        return sim_url
    if os.environ.get('SHOPPING_ADMIN') in url:
        sim_url = url.replace(os.environ.get('SHOPPING_ADMIN'), "http://magento.site/")
        return sim_url
    if "@localhost:2222" in url:
        url = url.replace("@localhost:2222", "@gitlab.com")
    if os.environ.get('GITLAB') in url:
        sim_url = url.replace(os.environ.get('GITLAB'), "https://gitlab.com")
        return sim_url
    if os.environ.get('MAP') in url:
        sim_url = url.replace(os.environ.get('MAP'), "http://openstreetmap.org")

    return url


def revert_sim_url(url: str) -> str:
    """"""

    if "https://www.onestopmarket.com" in url:
        sim_url = url.replace("https://www.onestopmarket.com", os.environ.get('SHOPPING'))
        return sim_url
    if "https://osclass-classifieds.com" in url:
        sim_url = url.replace("https://osclass-classifieds.com", os.environ.get('CLASSIFIEDS'))
        return sim_url
    if "/r/" in url:
        url = url.replace("/r/", "/f/")
    if "https://www.reddit.com" in url:
        sim_url = url.replace("https://www.reddit.com", os.environ.get('REDDIT'))
        return sim_url
    if "https://en.wikipedia.org" in url:
        sim_url = url.replace("https://en.wikipedia.org", os.environ.get('WIKIPEDIA'))
        return sim_url
    if "http://magento.site/" in url:
        sim_url = url.replace("http://magento.site/", os.environ.get('CLASSIFIEDS'))
        return sim_url
    if "@gitlab.com" in url:
        url = url.replace("@gitlab.com", "@localhost:2222")
    if "https://gitlab.com" in url:
        sim_url = url.replace("https://gitlab.com", os.environ.get('GITLAB'))
        return sim_url

    return url


def site_name(url: str, sim_url=True) -> str:
    """"""
    
    if sim_url:
        url = get_sim_url(url)

    netloc = urlparse(url).netloc
    if len(netloc.split('.')) >= 2:
        site = netloc.split('.')[-2]
        return site.capitalize()

    return netloc


def url_to_name(url: str) -> str:
    # Convert WebArena local address to env name
    for env_name in ENV_LIST:
        env_url = os.environ.get(env_name)
        if (env_url) and ((url in env_url) or (env_url in url)):
            return env_name.capitalize()
    
    # For real websites, return normal url 
    return url


def url_path(url: str) -> str:

    parsed_url = urlparse(url)
    relative_path = parsed_url.path
    if parsed_url.query:
        relative_path += f"?{parsed_url.query}"

    return relative_path


#### ---- HTML text parsing ---- ####

def clean_text(text: str, max_words=100, max_length=1000, max_lines=2) -> str:
    """Clean and truncate element text extracted from html."""

    lines = text.split('\n')
    if not max_lines:
        max_lines = len(lines)
    
    if len(lines) == max_lines:
        stripped = []
        for line in lines:
            stripped.append(line.strip())
        # return ' '.join(stripped)
        text = ' '.join(stripped)
    elif len(lines) > max_lines:
        lines = lines[:max_lines]
        stripped = []
        for line in lines:
            stripped.append(line.strip())
        # return ' '.join(stripped)
        text = ' '.join(stripped)
    
    if len(text) > max_length:
        text = text[:max_length]

    words = text.split(' ')
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words])
        return truncated

    return text.strip()


def process_aria(snapshot: str, remove_tags: list[str]=[]) -> str:
    """"""

    lines = snapshot.splitlines(keepends=True)

    for i in range(len(lines)):
        line = lines[i]
        leading_spaces = len(line) - len(line.lstrip(' '))

        # Remove '- '
        lines[i] = line[:leading_spaces] + line[leading_spaces+2:]

        element = line[leading_spaces+2:]
        if element.strip().endswith('img'):
            # Remove empty img
            lines[i] = ""
        elif element.startswith('checkbox'):
            if ('checked' not in element):
                lines[i] = line.strip() + ' [empty]\n'
        elif element.startswith('link'):
            # Remove link same text
            link_text = element[6:-3]
            for j in range(i+1, len(lines)):
                next_line = lines[j]
                next_spaces = len(next_line) - len(next_line.lstrip(' '))
                if next_spaces <= leading_spaces:
                    break
                next_elem = next_line.lstrip(' ')
                if (next_elem.startswith('- text') or next_elem.startswith('- strong')) and (link_text in next_line):
                    lines[j] = ""
        elif (i > 0) and (any(element.startswith(t) for t in remove_tags)):
            # Remove nested
            lines[i] = ""
            for j in range(i+1, len(lines)):
                next_line = lines[j]
                next_spaces = len(next_line) - len(next_line.lstrip(' '))
                if (next_spaces <= leading_spaces):
                    break
                lines[j] = ""
    
    cleaned = "".join(lines)

    return cleaned


def extract_url_snapshot(snapshot: str, filename: str) -> str:
    """"""

    lines = snapshot.splitlines()

    for line in lines:
        if ('url: ' in line) and (filename in line):
            img_url = line.split('url: ')[-1].strip()
            print(img_url)
            return img_url

    return filename


def replace_urls_full(text: str, all_urls: list[str], page_url: str) -> str:
    """Replace relative urls in text with full url"""

    all_urls = list(set(all_urls))

    for url in all_urls:
        if url.startswith('/'):
            full_url = get_full_url(url, page_url)
            text = text.replace(url, full_url)

    return text


def remove_markdown_urls(text: str) -> str:
    """Finds all markdown links and replaces them with only the link text."""
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)


def remove_repeated_urls(text: str) -> str:
    """Remove repeated urls from text (keep first)"""
    seen_urls = set()
    pattern = r"(?<=\])\(([^)]*)\)"

    def replacer(match: re.Match) -> str:
        url = match.group(1)

        if not url:
            return match.group(0)

        if (url in seen_urls) and (len(url)>10):
            return "()"
        else:
            seen_urls.add(url)
            return match.group(0)

    return re.sub(pattern, replacer, text)



def remove_html_tags(html: str, tags_to_remove: List[str]) -> str:
    """Remove specified HTML tags from an HTML document."""
    
    # Parse the HTML content
    soup = BeautifulSoup(html, "html.parser")
    
    # Find and remove all elements with specified tags
    for tag_name in tags_to_remove:
        for element in soup.find_all(tag_name):
            element.decompose()
    
    # Return the modified HTML as a string
    return str(soup)


def remove_nested_html(html: str, tag: str, elem_class: str=None) -> str:
    """"""

    if (f"<{tag}" not in html) or (tag == 'div'):
        return html

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    find_kwargs = {"name": tag}
    
    # Find specified elements in the HTML
    for nested_elem in soup.find_all(**find_kwargs):
        if (isinstance(nested_elem, Tag)):
            nested_elem.decompose()
        else:
            print(f"None")
    
    # Return the modified inner HTML as a string
    return str(soup)


def remove_link_newlines(markdown_text: str) -> str:
    """"""

    link_pattern = r'\[([^\]]*?)\]\((.*?)\)'

    # Replace line breaks within the square brackets
    def replace_linebreaks(match):
        link_text = match.group(1).replace('\n\n', '\n').strip()
        link_text = link_text.replace('\n', ' ')
        url = match.group(2)
        return f"[{link_text}]({url})"
    
    updated_markdown = re.sub(link_pattern, replace_linebreaks, markdown_text)

    return updated_markdown


def extract_urls_md(markdown: str) -> list[str]:
    """Extract list of all URLs in the markdown string"""

    url_list = []
    pattern = r"\[[^\]]+\]\((.+?)\)"
    extracted_urls = re.findall(pattern, markdown)
    url_list.extend(extracted_urls)

    return url_list


def process_md(markdown: str, page_url: str=None, remove_links=False) -> str:
    """Post-process raw markdown converted text before inputting to LLM."""
    
    # Replace 3 or more newlines with exactly 2 newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # strip spaces
    cleaned_lines = []
    lines = cleaned_text.split('\n\n')
    for line in lines:
        line = line.strip()
        # if line
        cleaned_lines.append(line)
    cleaned_text = "\n\n".join(cleaned_lines)
    cleaned_text = cleaned_text.strip()

    if remove_links:
        cleaned_text = remove_markdown_urls(cleaned_text)
    else:
        # Remove newlines in links
        cleaned_text = remove_link_newlines(cleaned_text)

        # Replace relative urls with full url
        all_urls = extract_urls_md(markdown)
        cleaned_text = replace_urls_full(cleaned_text, all_urls, page_url)

        # Remove repeated urls
        cleaned_text = remove_repeated_urls(cleaned_text)

    return cleaned_text


def extract_file_url_md(markdown: str, filename: str):
    """Extract the url that ends in filename from the markdown."""

    # Track current position in string
    pos = 0
    
    while True:
        # Find next markdown link pattern
        start_idx = markdown.find('](', pos)
        if start_idx == -1:  # No more links found
            return None
            
        # Move to start of URL
        start_idx += 2
        
        # Find closing parenthesis
        end_idx = markdown.find(')', start_idx)
        if end_idx == -1:  # Malformed markdown
            return None
            
        # Extract URL
        url = markdown[start_idx:end_idx]
        
        # Check if this URL contains our image file
        if filename in url:
            return url
            
        # Move position to continue search
        pos = end_idx + 1


def parse_class_string(structured_string: str) -> Tuple[Optional[str], List[str]]:
    """
    Helper function for parsing list item class string
    """
    if not structured_string or structured_string.isspace():
        return (None, [])

    lines = structured_string.strip().splitlines()

    if not lines:
        return (None, [])

    # --- Process the first line (class name) ---
    first_line = lines[0].strip()
    if not first_line.endswith(':'):
        logger.warning(f"Invalid format: First line '{lines[0]}' must end with a colon ':'.")

    class_name = first_line[:-1].strip()
    if not class_name:
        logger.warning("Invalid format: Class name cannot be empty.")

    # --- Process subsequent lines (attributes) ---
    attributes = []
    expected_prefix = "  - "

    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        if not line.startswith(expected_prefix):
            continue
        # Extract attribute name, removing prefix and stripping whitespace
        attribute_name = line[len(expected_prefix):].strip()
        if not attribute_name:
            continue

        attributes.append(attribute_name)

    return (class_name, attributes)


def is_one_replacement(word, valid):
    """
    Returns True if word and valid have the same length and differ by exactly one character.
    """
    if len(word) != len(valid):
        return False
    diff_count = 0
    for c1, c2 in zip(word, valid):
        if c1 != c2:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

def is_one_deletion(word, valid):
    """
    Returns True if word can be transformed into valid by deleting exactly one character.
    That is, word is one character longer than valid.
    """
    if len(word) != len(valid) + 1:
        return False
    i = j = 0
    while i < len(word) and j < len(valid):
        if word[i] == valid[j]:
            i += 1
            j += 1
        else:
            # Allow one deletion: if indices differ, then more than one deletion is needed.
            if i != j:
                return False
            i += 1
    return True

def is_one_addition(word, valid):
    """
    Returns True if word can be transformed into valid by adding exactly one character.
    Equivalently, valid is one character longer than word.
    """
    if len(valid) != len(word) + 1:
        return False
    i = j = 0
    while i < len(word) and j < len(valid):
        if word[i] == valid[j]:
            i += 1
            j += 1
        else:
            # Allow one addition: if indices differ, then more than one addition is needed.
            if i != j:
                return False
            j += 1
    return True

def correct_off_by_one(text, valid_words):
    """"""

    words = text.split(' ')
    corrected_words = []
    
    for word in words:
        if (len(word)<4) or (word.lower() in valid_words):
            corrected_words.append(word)
            continue
        
        correction = None
        
        # First, try to find a valid word that is off-by-one via a single character replacement.
        for valid in valid_words:
            if is_one_replacement(word, valid):
                correction = valid
                break
        
        # Next, if no replacement was found, check for a deletion.
        if correction is None:
            for valid in valid_words:
                if is_one_deletion(word, valid):
                    correction = valid
                    break
        
        # Finally, if still no correction found, check for an addition.
        if correction is None:
            for valid in valid_words:
                if is_one_addition(word, valid):
                    correction = valid
                    break

        # Use the correction if found; otherwise, keep the original word.
        corrected_words.append(correction if correction is not None else word)
    
    return " ".join(corrected_words)


def autocorrect_text(text, ground_truth):
    """"""

    if not ground_truth:
        return text
    words_list = ground_truth.split(' ')
    words_list = [word.lower() for word in words_list]

    text = correct_off_by_one(text, words_list)

    return text


def is_state_class(class_str: str) -> str:
    """Return True if class is for dynamic element state."""

    class_str = class_str.lower()
    
    if class_str.startswith('is-'):
        return True
    if class_str.startswith('can-'):
        return True
    if class_str.endswith('active'):
        return True
    if class_str.endswith('show'):
        return True
    if class_str.endswith('current'):
        return True
    if class_str.endswith('expanded'):
        return True
    if class_str.endswith('collapsed'):
        return True
    if class_str.endswith('visible'):
        return True
    if class_str.endswith('is-open'):
        return True
    if class_str.endswith('hover'):
        return True

    return False


try:
    import webchallenger.benchmarks.visualwebarena
except ImportError:
    logger.error("Please run `make prepare` to setup the environment")
    raise

