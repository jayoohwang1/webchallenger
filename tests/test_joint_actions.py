from types import SimpleNamespace as NS
import unittest
from webchallenger.batched_sections import BatchedSectionsMixin
from webchallenger.memory.agent_memory import PageMem

class JointActionsTests(unittest.TestCase):
    def setUp(self):
        self.a=BatchedSectionsMixin()
        self.e1,self.e2,self.hidden,self.loose=object(),object(),object(),object()
        self.s1=NS(name='Search',elements=[self.e1,self.e2],task_summary='Search controls',task_details='Query is CMU')
        self.s2=NS(name='Footer',elements=[self.hidden],task_summary='',task_details='')
        self.a.page_obs=PageMem(load_mem=False)
        self.a.page_obs.html_sections=[self.s1,self.s2]
        self.a.page_obs.task_sections=[self.s1]
        self.a.page_obs.outer_elements=[self.e1,self.e2,self.hidden,self.loose,self.loose]
    def test_selected_and_unsectioned_elements_without_duplicates(self):
        self.assertEqual(self.a.joint_action_candidates([self.s1]),[(self.e1,self.s1),(self.e2,self.s1),(self.loose,None)])
    def test_global_indices_preserved_after_executor_filtering(self):
        actions=[(self.e2,self.s1),(self.loose,None),('Type a URL',None),('End task',None)]
        text=self.a.joint_action_context(actions,['Search','Help','Type a URL','Mark task as complete.'])
        for expected in ['Summary: Search controls','Details:\nQuery is CMU','1) Search','2) Help','3) Type a URL','4) Mark task as complete.','OTHER PAGE ELEMENTS','BROWSER AND NAVIGATION ACTIONS']:
            self.assertIn(expected,text)
        self.assertEqual(text.count('1) Search'),1)

    def test_duplicate_section_reference_does_not_repeat_action_numbers(self):
        self.a.page_obs.task_sections=[self.s1,self.s1]
        text=self.a.joint_action_context([(self.e1,self.s1)],['Search'])
        self.assertEqual(text.count('1) Search'),1)

if __name__=='__main__':unittest.main()
