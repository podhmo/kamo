import unittest
from evilunit import test_target


@test_target("kamo:Template")
class ITTests(unittest.TestCase):
    def test_it(self):
        from kamo._sample import template
        t0 = self._makeOne(template, optimize=False, nocache=True)
        t1 = self._makeOne(template, optimize=True, nocache=True)
        # import logging
        # logging.basicConfig(level=logging.DEBUG)
        context0 = dict(x=10, xs=["foo", "bar", "boo"], hello="hello ", boo="(o_0)")
        context1 = dict(x=10, xs=["foo", "bar", "boo"], hello="hello ", boo="(o_0)")
        self.assertEqual(t0.render(**context0), t1.render(**context1))
