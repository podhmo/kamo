import unittest
from evilunit import test_function


@test_function("kamo.expr:collect_variable_name")
class Tests(unittest.TestCase):
    def _callFUT(self, code):
        import ast
        return self._getTarget()(ast.parse(code))

    def test_toplevel0(self):
        code = """\
a = x + f(y) * 1
"""
        result = self._callFUT(code)
        self.assertEqual(list(result), ["a"])

    def test_toplevel1(self):
        code = """\
def g(x):
    h = y
    def k(z):
        pass

"""
        result = self._callFUT(code)
        self.assertEqual(list(result), ["g"])

    def test_toplevel2(self):
        code = """\
class A(object):
     class B(object):
         pass

     def f(self):
          pass

     x = 10
class B:
     pass
"""
        result = self._callFUT(code)
        self.assertEqual(list(sorted(result)), ["A", "B"])

    def test_toplevel3(self):
        code = """\
import foo0
from foo0 import foo1

class B:
    import boo0
    from boo0 import boo1

def f(x):
    import yoo0
    from yoo0 import yoo1
"""
        result = self._callFUT(code)
        self.assertEqual(list(sorted(result)), ["B", "f", "foo0", "foo1"])

    def test_expr0(self):
        code = "a, b, c"
        result = self._callFUT(code)
        self.assertEqual(list(sorted(result)), ["a", "b", "c"])

    def test_expr1(self):
        code = "(a, (b, )), 10"
        result = self._callFUT(code)
        self.assertEqual(list(sorted(result)), ["a", "b"])
