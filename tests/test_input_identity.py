import os
from types import SimpleNamespace as NS
import unittest
from playwright.sync_api import sync_playwright
from webchallenger.input_identity import capture_row_identity, resolve_input, validated_input_handle

class InputIdentityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(**({'executable_path': os.environ['WEBCHALLENGER_CHROMIUM']} if os.environ.get('WEBCHALLENGER_CHROMIUM') else {}),headless=True)
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop()
 def setUp(self):
  self.page=self.browser.new_page()
  self.page.set_content('<table><tbody><tr><td><input type="checkbox" value="123"></td><td><input name="position" class="input-text" value="1"></td></tr><tr><td><input type="checkbox" value="456"></td><td><input name="position" class="input-text" value="2"></td></tr></tbody></table>')
  self.base=self.page.locator('body')
  self.e=NS(tag='input',name='position',class_name='input-text',input_type='text',attributes_dict={},row_identity=capture_row_identity(self.page.locator('input[name=position]').nth(1)))
 def tearDown(self):self.page.close()
 def test_row_reorder_keeps_product_identity(self):
  self.page.evaluate('document.querySelector("tbody").prepend(document.querySelectorAll("tr")[1])')
  loc=resolve_input(self.base,self.e);h=validated_input_handle(loc,self.e)
  self.assertIsNotNone(h);h.fill('9');h.dispose()
  self.assertEqual(self.page.locator('tr').filter(has=self.page.locator('input[value="456"]')).locator('[name=position]').input_value(),'9')
  self.assertEqual(self.page.locator('tr').filter(has=self.page.locator('input[value="123"]')).locator('[name=position]').input_value(),'1')
 def test_checkbox_substitution_rejected(self):
  loc=self.page.locator('[name=position]').nth(1)
  loc.evaluate('el => el.type="checkbox"')
  self.assertEqual(resolve_input(self.base,self.e).count(),0)
  self.assertIsNone(validated_input_handle(loc,self.e))
 def test_duplicate_or_missing_row_rejected(self):
  self.page.evaluate('document.querySelector("tbody").append(document.querySelectorAll("tr")[1].cloneNode(true))')
  self.assertEqual(resolve_input(self.base,self.e).count(),0)
  self.e.row_identity={'kind':'checkbox','value':'999'}
  self.assertEqual(resolve_input(self.base,self.e).count(),0)
 def test_no_anchor_cannot_choose_among_duplicate_inputs(self):
  self.e.row_identity=None
  self.assertIsNone(validated_input_handle(resolve_input(self.base,self.e),self.e))
 def test_pinned_handle_does_not_retarget_after_replacement(self):
  h=validated_input_handle(resolve_input(self.base,self.e),self.e)
  self.page.locator('[name=position]').nth(1).evaluate('el => el.replaceWith(el.cloneNode(true))')
  with self.assertRaises(Exception):h.fill('BAD',timeout=100)
  h.dispose();self.assertEqual(self.page.locator('[name=position]').nth(1).input_value(),'2')
 def test_readonly_rejected(self):
  self.page.locator('[name=position]').nth(1).evaluate('el => el.readOnly=true')
  self.assertIsNone(validated_input_handle(resolve_input(self.base,self.e),self.e))
