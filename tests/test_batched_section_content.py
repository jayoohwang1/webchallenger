import os
"""Exercise section evidence against real HTML and SVG browser nodes."""
from types import SimpleNamespace as NS
import unittest
from playwright.sync_api import sync_playwright
from webchallenger.batched_sections import BatchedSectionsMixin

class SectionContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(**({'executable_path': os.environ['WEBCHALLENGER_CHROMIUM']} if os.environ.get('WEBCHALLENGER_CHROMIUM') else {}), headless=True)
    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
    def setUp(self):
        self.page = self.browser.new_page()
        self.agent = BatchedSectionsMixin()
        self.agent.get_section_locator = lambda section: self.page.locator('.section')
        self.agent.elems_context = lambda *args, **kwargs: 'Controls'
        self.section = NS(elements=[])
    def tearDown(self):
        self.page.close()
    def test_svg_reproduces_original_failure_and_preserves_labels(self):
        self.page.set_content('<svg class="section"><text>Route 1.7 km</text></svg>')
        self.assertEqual(self.page.locator('.section').all_inner_texts(), [None])
        with self.assertRaises(TypeError):
            '\n'.join(self.page.locator('.section').all_inner_texts())
        self.assertEqual(self.agent.batch_section_content(self.section), 'Route 1.7 km\n\nControls')
    def test_html_hidden_text_not_added_and_svg_empty_safe(self):
        self.page.set_content('<div class="section">Visible<span hidden>Hidden</span></div><svg class="section"><path d="M0 0"/></svg>')
        self.assertEqual(self.agent.batch_section_content(self.section), 'Visible\n\n\nControls')
    def test_missing_section_is_empty_evidence(self):
        self.page.set_content('<div>No section</div>')
        self.assertEqual(self.agent.batch_section_content(self.section), '\n\nControls')
