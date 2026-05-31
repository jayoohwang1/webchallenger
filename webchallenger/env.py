import json
import numpy as np
import os
import requests
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from PIL import Image
from playwright.sync_api import Playwright, ViewportSize, Error, sync_playwright

from webchallenger.memory import PageMem
from webchallenger.visualwebarena import (
    Action,
    DetachedPage,
    Observation,
    ScriptBrowserEnv,
    StateInfo,
    execute_action,
)

CLASSIFIEDS = os.environ.get('CLASSIFIEDS')
CLASSIFIEDS_RESET_TOKEN = os.environ.get('CLASSIFIEDS_RESET_TOKEN')


class WebChallengerEnv(ScriptBrowserEnv):
    def __init__(self, browser: str = "chromium", start_url: str = None, **kwargs: Any) -> None:
        self.browser = browser
        if start_url:
            self.start_url = urlparse(start_url, scheme="https").geturl()
        else:
            self.start_url = None
        super().__init__(**kwargs)

    def setup(self, config_file: Path | None = None) -> None:
        """Setting up the environment, simple and easy.
        """
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}
        

        # Reset site if needed. Currently only supported for Classifieds.
        # TODO(jykoh): Add reset functionality for Shopping/Reddit.
        if instance_config.get("require_reset", False):
            if "classifieds" in instance_config["sites"]:
                # Send POST request to __CLASSIFIEDS__/index.php?page=reset with token=CLASSIFIEDS_TOKEN
                response = requests.post(
                    f"{CLASSIFIEDS}/index.php?page=reset",
                    data={"token": CLASSIFIEDS_RESET_TOKEN},
                )

                # Check if the request was successful
                if response.status_code == 200:
                    print("Reset Classifieds site.")
                else:
                    print(
                        "Failed to reset Classifieds site:",
                        response.status_code,
                    )
            else:
                print(
                    "WARNING: Reset is not supported for this site. Please manually reset the site."
                )

        viewport_size = self.viewport_size.copy()
        config_viewport_size = instance_config.get("viewport_size", {})
        if (config_viewport_size) and (config_viewport_size['width'] != 1280):
            viewport_size.update(config_viewport_size)
        self.observation_handler.viewport_size = viewport_size

        # storage_state = instance_config.get("storage_state", None)
        config_start_url = instance_config.get("start_url", None)

        self.context = self.browser.new_context(
            viewport=viewport_size,
            # storage_state=storage_state,
            device_scale_factor=1,
        )

        if config_start_url:
            start_urls = config_start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                client = page.context.new_cdp_session(page)
                # client.send("Accessibility.enable")
                # client.detach()
                page.client = client
                page.goto(url, timeout=60000)
            # set the first page as the current page
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            client = self.page.context.new_cdp_session(self.page)
            self.page.client = client
        
        if self.start_url:
            browser_response = self.page.goto(self.start_url, timeout=60000, wait_until='domcontentloaded')
            print(browser_response.status)

        self.context.grant_permissions(['clipboard-read', 'clipboard-write'])
        self.page.set_default_timeout(10000)


    def _wait_dom_loaded(self, networkidle=False):
        """Wait for all pages to load
        https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/env.py#L476"""
        
        loadstate = "domcontentloaded"
        if networkidle:
            loadstate = "networkidle"
        
        for page in self.context.pages:
            try:
                page.wait_for_load_state(loadstate, timeout=3000)
            except Error:
                pass
            for frame in page.frames:
                try:
                    frame.wait_for_load_state(loadstate, timeout=3000)
                except Error:
                    pass


    def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")
        
        scroll_height = self.page.evaluate("window.pageYOffset || document.documentElement.scrollTop")

        success = False
        fail_error = ""
        try:
            self.page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
            )
            success = True
        except Exception as e:
            fail_error = str(e)

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, self.page.content()),
            "scroll_height": scroll_height,
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
        }
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg


    def get_screenshot(self, max_height: int=None, iframe_id: str=None) -> Image:
        """Return screenshot of current page"""

        try:
            full_page = False
            if (max_height) and (max_height > 1280):
                full_page = True

            screenshot_arr = np.array(Image.open(BytesIO(self.page.screenshot(full_page=full_page, timeout=30000))))
            screenshot = Image.fromarray(screenshot_arr)

            max_width = self.viewport_size["width"]
            if not max_height:
                max_height = screenshot.size[1]
            
            # screenshot = screenshot.crop((0, 0, max_width-15, max_height))
            screenshot = screenshot.crop((0, 0, max_width, max_height))
        except Error as e:
            logger.error(f"{repr(e)}")
            screenshot = None

        return screenshot
    

