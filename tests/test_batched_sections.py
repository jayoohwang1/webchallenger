"""Offline fixtures exercise real batching and agent dispatch, without browser/model setup."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from webchallenger.batched_sections import BatchedSectionsMixin, parse_batch

SOURCE = Path(__file__).resolve().parents[1] / 'webchallenger/agent.py'

def agent_method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'PageSection': NS, 'logger': Mock()}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace[name]

class Harness(BatchedSectionsMixin):
    analyze_page = agent_method('analyze_page')
    top_page_sections = agent_method('top_page_sections')
    summarize_page = agent_method('summarize_page')

    def __init__(self, visual=False):
        self.cfg = NS(llm_only=not visual, split_page_sections=True, filter_page_info=True,
                      batch_section_summaries=True, batch_detail_extraction=True, caption_img=True)
        self.manual = False
        self.intent = 'Find the correct route'
        self.page_obs = NS(name='Map', url='https://example.com', page_summary='', alerts=[], list_section=None)
        self.screen_change = ''; self.clipboard = ''
        self.model_manager = NS(llm_call=Mock(), multi_img_vlm_call=Mock())
        self.episode_history = lambda: 'Opened map'
        self.format_llm_prompt = lambda prompt, **kw: [{'role': 'user', 'content': prompt}]
        self.batch_section_content = lambda section: section.evidence
        self.section_screenshot = lambda section, **kw: (section.crop, None)
        self.section_info = Mock(side_effect=AssertionError('Per-section detail helper invoked'))
        self.extract_details = Mock(side_effect=AssertionError('Per-section LLM invoked'))


def section(text='Route', crop=None):
    return NS(type='div', class_name='same-class', evidence=text, bbox=True, crop=crop,
              summary='old', task_details='old', task_summary='old', elements=[],
              list_type=None, not_relevant=False, table_filters=[], n_dropdowns=0)

class BatchTests(unittest.TestCase):
    def test_text_summary_is_one_call_and_maps_out_of_order(self):
        a = Harness(); sections = [section('A'), section('B')]
        a.model_manager.llm_call.return_value = json.dumps({'sections': [
            {'id': 's2', 'summary': 'B summary'}, {'id': 's1', 'summary': 'A summary'}]})
        a.summarize_sections_batched(sections)
        self.assertEqual([s.summary for s in sections], ['A summary', 'B summary'])
        a.model_manager.llm_call.assert_called_once(); a.model_manager.multi_img_vlm_call.assert_not_called()

    def test_mixed_images_have_explicit_mapping(self):
        a = Harness(visual=True); crop = object()
        a.model_manager.multi_img_vlm_call.return_value = json.dumps({'sections': [
            {'id': 's2', 'summary': 'A'}, {'id': 's1', 'summary': 'B'}]})
        sections = [section(), section(crop=crop)]
        a.summarize_sections_batched(sections)
        self.assertEqual([s.summary for s in sections], ['A', 'B'])
        kwargs = a.model_manager.multi_img_vlm_call.call_args.kwargs
        self.assertEqual(kwargs['images'], [crop])
        records = json.loads(kwargs['vlm_prompt'].split('SECTIONS:\n')[1])
        self.assertNotIn('image_number', records[0]); self.assertEqual(records[1]['image_number'], 1)
        self.assertEqual([r['id'] for r in records], ['s2', 's1'])
        a.model_manager.llm_call.assert_not_called()

    def test_unknown_duplicate_and_missing_ids_do_not_crash(self):
        a = Harness(); sections = [section('A'), section('B')]
        a.model_manager.llm_call.return_value = json.dumps({'sections': [
            {'id': 's1', 'summary': 'valid'}, {'id': 's3', 'summary': 'unknown'},
            {'id': 's1', 'summary': 'duplicate'}]})
        with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
            a.summarize_sections_batched(sections)
        self.assertEqual([s.summary for s in sections], ['valid', ''])
        a.model_manager.llm_call.assert_called_once()

    def test_details_skip_unknown_id_and_clear_missing_result(self):
        a = Harness(); sections = [section('A'), section('B')]
        a.model_manager.llm_call.return_value = json.dumps({'sections': [
            {'id': 's1', 'summary': 'valid', 'details': 'A detail'},
            {'id': 's3', 'summary': 'unknown', 'details': 'wrong detail'}],
            'page_summary': 'Observation'})
        with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
            result = a.extract_details_batched([(s, s.evidence) for s in sections])
        self.assertEqual(result, 'Observation')
        self.assertEqual([s.task_details for s in sections], ['A detail', ''])
        self.assertEqual([s.task_summary for s in sections], ['valid', ''])

    def test_bad_fields_still_fail_atomically(self):
        a = Harness(); s = section()
        a.model_manager.llm_call.return_value = json.dumps({'sections': [
            {'id': 's1', 'summary': None}]})
        with self.assertRaises(ValueError): a.summarize_sections_batched([s])
        self.assertEqual(s.summary, 'old')

    def test_observed_missing_outer_brace_and_fenced_prose(self):
        valid = json.dumps({'sections': [{'id': 's1', 'summary': 'Exact "quote"'}]})
        for response in (valid[:-1], '```json\n' + valid + '\n```\nExtra prose'):
            with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
                rows, _ = parse_batch(response, ['s1'], ['summary'])
            self.assertEqual(rows['s1']['summary'], 'Exact "quote"')

    def test_incomplete_values_are_not_repaired(self):
        for response in ('{"sections":[{"id":"s1","summary":"truncated',
                         '{"sections":[{"id":"s1","summary":"bad\\-escape"}]}'):
            with self.assertRaises(ValueError):
                parse_batch(response, ['s1'], ['summary'])

    def test_one_correction_retry_reuses_evidence_and_recovers(self):
        a = Harness(visual=True); crop = object(); s = section(crop=crop)
        a.section_screenshot = Mock(return_value=(crop, None))
        a.model_manager.multi_img_vlm_call.side_effect = [
            'not JSON', json.dumps({'sections': [{'id': 's1', 'summary': 'Recovered'}]})]
        with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
            a.summarize_sections_batched([s])
        self.assertEqual(s.summary, 'Recovered')
        a.section_screenshot.assert_called_once()
        calls = a.model_manager.multi_img_vlm_call.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs['images'], calls[1].kwargs['images'])
        self.assertIn('FORMAT CORRECTION:', calls[1].kwargs['vlm_prompt'])

    def test_correction_includes_specific_error_and_response_excerpt(self):
        cases = [
            '{"sections":[{"id":"s1","summary":"bad\\-escape"}]}',
            '{"sections":[{"id":"s1","summary": - unquoted text}]}',
            '{"sections":[{"id":"s1","summary":null}]}',
        ]
        for response in cases:
            with self.subTest(response=response):
                a = Harness(visual=True); crop = object(); s = section(crop=crop)
                a.section_screenshot = Mock(return_value=(crop, None))
                a.model_manager.multi_img_vlm_call.side_effect = [
                    response, json.dumps({'sections': [{'id': 's1', 'summary': 'Recovered'}]})]
                try:
                    parse_batch(response, ['s1'], ['summary'])
                except ValueError as exc:
                    expected = str(exc)
                with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
                    a.summarize_sections_batched([s])
                prompt = a.model_manager.multi_img_vlm_call.call_args.kwargs['vlm_prompt']
                self.assertIn('Parser/validation error: ' + expected, prompt)
                if 'null' not in response:
                    self.assertIn(json.dumps(response), prompt)
                    self.assertIn('Error offset within the excerpt', prompt)
                self.assertEqual(s.summary, 'Recovered')

    def test_correction_retry_is_bounded_and_atomic(self):
        a = Harness(); s = section()
        a.model_manager.llm_call.return_value = 'invalid JSON'
        with self.assertLogs('webchallenger.batched_sections', level='WARNING'):
            with self.assertRaises(ValueError): a.summarize_sections_batched([s])
        self.assertEqual(a.model_manager.llm_call.call_count, 2)
        self.assertEqual(s.summary, 'old')

    def test_three_stage_flow_and_combined_observation(self):
        a = Harness(); sections = [section('A'), section('B')]
        a.page_obs.html_sections = sections
        a.model_manager.llm_call.side_effect = [
            json.dumps({'sections':[{'id':'s1','summary':'A'}, {'id':'s2','summary':'B'}]}),
            'select s1',
            json.dumps({'sections':[{'id':'s1','summary':'Relevant','details':'- Name: B'}], 'page_summary':'Found B'})]
        a.choose_top_sections = Mock(side_effect=lambda ss, **kw: (a.model_manager.llm_call('choose'), [ss[1]])[1])
        a.summarize_sections_batched(sections)
        selected, _ = a.top_page_sections()
        a.analyze_page(selected)
        self.assertEqual(a.model_manager.llm_call.call_count, 3)
        self.assertEqual(sections[1].task_details, '- Name: B')
        self.assertEqual(a.page_obs.page_summary, 'Found B')
        self.assertEqual(a.page_obs.task_info, '- Name: B')

    def test_detail_flag_works_without_summary_flag(self):
        a = Harness(); a.cfg.batch_section_summaries = False
        a.model_manager.llm_call.return_value = json.dumps({
            'sections':[{'id':'s1','summary':'Useful','details':'Name: B'}], 'page_summary':'B'})
        a.analyze_page([section()])
        a.model_manager.llm_call.assert_called_once()
        self.assertEqual(a.page_obs.page_summary, 'B')

    def test_unfiltered_analysis_does_not_batch_details(self):
        a = Harness(); a.cfg.filter_page_info = False
        a.section_info = Mock(return_value='page')
        a.browser_tabs_obs = []; a.prompts = NS(summarize_obs='summarize')
        a.page_sections_context = lambda *args, **kw: 'page'
        a.extract_llm_answer = lambda *args, **kw: 'observed'
        a.format_llm_prompt = lambda *args, **kw: []
        a.analyze_page.__func__.__globals__['get_sim_url'] = lambda x: x
        a.model_manager.llm_call.return_value = 'observed'
        a.extract_details_batched = Mock(side_effect=AssertionError('Unexpected batching'))
        a.analyze_page([section()])
        a.extract_details_batched.assert_not_called()
        a.model_manager.llm_call.assert_called_once()

    def test_empty_batch_makes_no_calls(self):
        a = Harness(); a.summarize_sections_batched([])
        self.assertEqual(a.extract_details_batched([]), '')
        a.model_manager.llm_call.assert_not_called()

    def test_fenced_json_and_page_summary_validation(self):
        text = '```json\n{"sections":[{"id":"s1","details":""}],"page_summary":""}\n```'
        rows, summary = parse_batch(text, ['s1'], ['details'], True)
        self.assertEqual(rows['s1']['details'], '')
        with self.assertRaises(ValueError): parse_batch('{"sections":[]}', [], [], True)

    def test_precomputed_page_summary_makes_no_duplicate_calls(self):
        a = Harness(); s = section(); s.summary = 'Search and directions'
        page = NS(name='Map', html_sections=[s], list_section=s, short_summary='')
        a.summarize_sections_batched = Mock(side_effect=AssertionError('Duplicate summary'))
        a.summarize_page(NS(), page, all_sections=False, sections_summarized=True)
        self.assertEqual(page.short_summary, 'Search and directions')
        a.model_manager.llm_call.assert_not_called()
        a.model_manager.multi_img_vlm_call.assert_not_called()

    def test_observation_optimization_requires_both_flags_and_task(self):
        a = Harness(); self.assertTrue(a.batched_observation_enabled())
        a.cfg.batch_detail_extraction = False
        self.assertFalse(a.batched_observation_enabled())
        a.cfg.batch_detail_extraction = True; a.intent = ''
        self.assertFalse(a.batched_observation_enabled())

    def test_defaults_off(self):
        tree = ast.parse(SOURCE.read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CFG')
        values = {n.target.id: ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.AnnAssign)}
        self.assertIs(values['batch_section_summaries'], False)
        self.assertIs(values['batch_detail_extraction'], False)

if __name__ == '__main__': unittest.main()
