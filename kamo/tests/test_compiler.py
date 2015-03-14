# -*- coding:utf-8 -*-
import unittest
from evilunit import test_target


@test_target("kamo:Compiler")
class Tests(unittest.TestCase):
    def _callFUT(self, code):
        from kamo import (
            Lexer,
            Parser
        )
        lexer = Lexer()
        parser = Parser()
        compiler = self._getTarget()()
        return compiler(parser(lexer(code)))

    def test_it0(self):
        code = """
${datetime.now()}
"""
        result = str(self._callFUT(code))
        self.assertIn("c['datetime']", result)

    def test_it1(self):
        code = """
<% from datetime import datetime %>
${datetime.now()}
"""
        result = str(self._callFUT(code))
        self.assertNotIn("c['datetime']", result)

    def test_it2(self):
        code = """
<%datetime = object()%>
from datetime import datetime
${datetime.now()}
"""
        result = str(self._callFUT(code))
        self.assertNotIn("c['datetime']", result)

    def test_it3(self):
        code = """
%for datetime in xs:
    ${datetime.now()}
%endfor
"""
        result = str(self._callFUT(code))
        self.assertNotIn("c['datetime']", result)

    def test_it4(self):
        code = """
<%
def f(datetime):
    pass
%>
${datetime.now()}
"""
        result = str(self._callFUT(code))
        self.assertIn("c['datetime']", result)

    def test_it5(self):
        code = """
${foo.datetime.now()}
"""
        result = str(self._callFUT(code))
        self.assertNotIn("c['datetime']", result)
