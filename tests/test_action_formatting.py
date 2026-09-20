"""Exercise the production selector without launching the browser/model runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import Mock
import traceback

source = Path(__file__).resolve().parents[1] / 'webchallenger' / 'agent.py'
tree = ast.parse(source.read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'choose_action')
namespace = {'logger': Mock(), 'traceback': traceback, 'get_sim_url': lambda u: u}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

class ActionFormattingTests(unittest.TestCase):
    def agent(self, answers):
        a = NS(cfg=NS(joint_action_selection=True, split_page_sections=True,
                       summarize_state=True, filter_page_info=False),
               prompts=NS(choose_action='Choose one.'), intent='test',
               page_obs=NS(name='Page', url='https://example.com', page_summary='Page summary',
                           browser_dialog=None, list_info=''),
               browser_tabs_obs=[], clipboard='', prev_thought='', prev_action='',
               manual=False, slow_mode=False, episode_history=lambda: '',
               format_list_to_str=lambda rows, numbered: '\n'.join(f'{i+1}) {x}' for i,x in enumerate(rows)),
               joint_action_context=Mock(return_value='ACTIONS: grouped'),
               format_llm_prompt=lambda system, user: (system, user),
               extract_llm_answer=lambda text, keyword: text)
        a.model_manager = NS(llm_call=Mock(side_effect=answers))
        a.choose_action = MethodType(namespace['choose_action'], a)
        return a

    def test_dropdown_elements_and_none_keep_flat_format(self):
        a = self.agent(['2'])
        element = object()
        self.assertIsNone(a.choose_action([element, None], ['Item', 'None'], dropdown=True))
        a.joint_action_context.assert_not_called()
        system, user = a.model_manager.llm_call.call_args.args[0]
        self.assertIn('DROPDOWN ACTIONS:\n1) Item\n2) None', user)
        self.assertNotIn('Action options are grouped', system)

    def test_plain_element_selection_also_stays_flat(self):
        a = self.agent(['1'])
        element = object()
        self.assertIs(a.choose_action([element], ['Item']), element)
        a.joint_action_context.assert_not_called()

    def test_executor_pairs_use_grouped_context(self):
        a = self.agent(['1'])
        option = (object(), None)
        self.assertIs(a.choose_action([option], ['Item'], grouped=True), option)
        a.joint_action_context.assert_called_once()

    def test_retry_truncates_executable_options_together_with_labels(self):
        a = self.agent(['999', '201'])
        options = [(object(), None) for _ in range(201)]
        self.assertIsNone(a.choose_action(options, ['Item'] * 201, grouped=True))
        self.assertEqual(len(a.joint_action_context.call_args.args[0]), 200)
        self.assertEqual(len(a.joint_action_context.call_args.args[1]), 200)

    def test_zero_cannot_select_last_option(self):
        a = self.agent(['0'])
        self.assertIsNone(a.choose_action([object()], ['Item']))

if __name__ == '__main__':
    unittest.main()
