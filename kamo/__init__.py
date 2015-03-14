# -*- coding:utf-8 -*-
import logging
logger = logging.getLogger(__name__)
import ast
import re
from prestring.python import PythonModule, NEWLINE
from functools import partial
from collections import namedtuple
from io import StringIO
from kamo.expr import (
    WithContextExprVistor,
    collect_variable_name
)
marker = object()
"""
{module} :: {statement}+
{statement} :: {doctag} | {comment} | {pythoncode} | {if} | {for} | {deftag} | {text}
{doctag} :: '<%doc>' {text} '<%/doc>'
{comment} :: '##' {text}
{pythoncode} :: '<%' {text} '%>'
{if} :: '%if' {expr} ':' {text} ['%elif' {text} ':' {text}]* ['%else' {text} ':' {text}]? '%endif'
{for} :: '%for' {expr} 'in' {expr} ':' {text} %endfor
{deftag} :: '<%def' {defname} '>' {text} '</%def>'  # not support
{expr} :: {text - {newline}} | '(' {text} ')'
{newline} :: '\n'
{text} :: [{expr} '\n']+
"""


class Intern(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Intern {!r}>'.format(self.name)


class Line(object):
    def __init__(self, body):
        self.body = body

begin_doc = Intern("<%doc>")
end_doc = Intern("</%doc>")
comment = Intern("##")
begin_code = Intern("<%")
end_code = Intern("%>")
begin_if = Intern("%if")
begin_else = Intern("%else")
begin_elif = Intern("%elif")
end_if = Intern("%endif")
begin_for = Intern("%for")
end_for = Intern("%endfor")


class Scanner(re.Scanner):
    def __init__(self, *args, **kwargs):
        super(Scanner, self).__init__(*args, **kwargs)
        self.body = []

    def append(self, x):
        self.body.append(x)

    def extend(self, x):
        self.body.extend(x)

    def __call__(self, body):
        for line in body.split("\n"):
            self.scan(line)
        return self.body


Lexer = partial(Scanner, [
    ('\s*<%doc>(.+)(?=</%doc>)', lambda s, x: s.extend([begin_doc, s.match.group(1)])),
    ("\s*<%(.+)(?=%>)", lambda s, x: s.extend([begin_code, s.match.group(1)])),
    ('\s*<%doc>', lambda s, x: s.append(begin_doc)),
    ('\s*</%doc>', lambda s, x: s.append(end_doc)),
    ('\s*## (.*)', lambda s, x: s.extend((comment, s.match.group(1)))),
    ("\s*<%", lambda s, x: s.append(begin_code)),
    ("\s*%>", lambda s, x: s.append(end_doc)),
    ("\s*%if", lambda s, x: s.append(begin_if)),
    ("\s*%elif", lambda s, x: s.append(begin_elif)),
    ("\s*%else", lambda s, x: s.append(begin_else)),
    ("\s*%endif", lambda s, x: s.append(end_if)),
    ("\s*%for", lambda s, x: s.append(begin_for)),
    ("\s*%endfor", lambda s, x: s.append(end_for)),
    (".+", lambda s, x: s.append(x))
])


Module = namedtuple("Module", "body")
Doc = namedtuple("Doc", "body multiline")
Code = namedtuple("Code", "body")
Text = namedtuple("Text", "body")
Expr = namedtuple("Expr", "body ast decorators declared")
If = namedtuple("If", "keyword expr body")  # xxx: include if, elif, else
For = namedtuple("For", "keyword expr src body")


class Parser(object):
    def __init__(self):
        self.stack = [[]]
        self.frame = self.stack[-1]
        self.depth = 0
        self.i = 0

    @property
    def body(self):
        return self.stack[0]

    def push_frame(self):
        # [[x, y, <pos>]] -> [[x, y, [<pos>]]]
        frame = []
        self.frame.append(frame)
        self.depth += 1
        self.frame = frame

    def pop_frame(self):
        frame = self.stack
        for i in range(self.depth):
            frame = frame[-1]
        self.depth -= 1
        self.frame = frame

    def parse_expr(self, expr, decorators=None, is_declared=True):  # hmm.
        ast_node = ast.parse(expr).body[0]
        if is_declared:
            declared = collect_variable_name(ast_node)
        else:
            declared = set()
        return Expr(expr,
                    ast_node,
                    decorators=decorators or [],
                    declared=declared)

    def __call__(self, tokens):
        self.i = 0
        n = len(tokens)
        while n > self.i:
            self.parse_statement(tokens)
        return self.body

    def parse_statement(self, tokens):
        t = tokens[self.i]
        if t is begin_doc:
            self.parse_doc(tokens)
        elif t is comment:
            self.parse_comment(tokens)
        elif t is begin_code:
            self.parse_code(tokens)
        elif t is begin_if:
            self.parse_if(tokens)
        elif t is begin_elif:
            self.parse_elif(tokens)
        elif t is begin_else:
            self.parse_else(tokens)
        elif t is end_if:
            self.parse_end_if(tokens)
        elif t is begin_for:
            self.parse_for(tokens)
        elif t is end_for:
            self.parse_end_for(tokens)
        else:
            self.parse_text(tokens)

    def parse_doc(self, tokens):
        self.i += 1  # skip
        body = []
        while tokens[self.i] is not end_doc:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip
        self.frame.append(Doc(body, multiline=True))

    def parse_comment(self, tokens):
        self.i += 1  # skip
        self.frame.append(Doc([tokens[self.i]], multiline=False))
        self.i += 1

    def parse_code(self, tokens):
        self.i += 1  # skip
        body = []
        while tokens[self.i] is not end_doc:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip
        code = self.parse_expr("\n".join(body), is_declared=True)
        self.frame.append(Code(code))

    def parse_if(self, tokens):
        self.i += 1  # skip
        self.frame.append(("if", self.parse_expr(tokens[self.i].strip(": "))))  # hmm.
        self.i += 1
        self.push_frame()
        self.parse_statement(tokens)

    def _create_if_block(self, tokens):
        # create if-block, elif-block, else-block
        self.pop_frame()
        body = self.frame.pop()
        keyword, cond = self.frame.pop()
        self.frame.append(If(keyword, cond, body))

    def parse_elif(self, tokens):
        self._create_if_block(tokens)
        self.i += 1  # skip
        self.frame.append(("elif", self.parse_expr(tokens[self.i].strip(": "))))  # hmm.
        self.i += 1
        self.push_frame()
        self.parse_statement(tokens)

    def parse_else(self, tokens):
        self._create_if_block(tokens)
        self.i += 1  # skip
        self.frame.append(("else", self.parse_expr(tokens[self.i].strip(": "))))  # hmm.
        self.i += 1
        self.push_frame()
        self.parse_statement(tokens)

    def parse_end_if(self, tokens):
        self._create_if_block(tokens)
        self.i += 1

    def parse_for(self, tokens):
        self.i += 1  # skip
        # for expr in expr:
        expr, src = [e.strip(" ") for e in tokens[self.i].rsplit(" in ", 1)]
        expr = self.parse_expr(expr.strip(" "), is_declared=True)
        src = self.parse_expr(src.rstrip(": "))
        self.frame.append(("for", expr, src))
        self.i += 1
        self.push_frame()
        self.parse_statement(tokens)

    def parse_end_for(self, tokens):
        # create for-block
        self.pop_frame()
        body = self.frame.pop()
        keyword, expr, src = self.frame.pop()
        self.frame.append(For(keyword, expr, src, body))
        self.i += 1  # skip

    emit_var_rx = re.compile("\${[^}]+}")  # 雑

    def parse_text(self, tokens):
        body = []
        for token, is_emit_var in split_with(self.emit_var_rx, tokens[self.i]):
            if is_emit_var:
                token = token[2:-1]  # ${foo} -> foo
                token_with_filter = [e.strip(" ") for e in token.split("|")]  # foo|bar|boo -> [foo, bar, boo]
                token = token_with_filter[0]
                token = self.parse_expr(token, token_with_filter[1:])
            body.append((token, is_emit_var))
        self.frame.append(Text(body))
        self.i += 1


def split_with(rx, sentence):
    r = []

    while sentence:
        m = rx.search(sentence)
        if not m:
            r.append((sentence, False))
            return r
        if not m.start() == 0:
            r.append((sentence[:m.start()], False))
        r.append((m.group(0), True))
        sentence = sentence[m.end():]
    return r


class _DeclaredStore(object):
    def __init__(self):
        self.stack = [set()]

    def __contains__(self, k):
        return any(k in frame for frame in self.stack)

    def push_frame(self, s):
        self.stack.append(s)

    def pop_frame(self, s):
        self.stack.pop(s)


class Compiler(object):
    def __init__(self, m=None, default="''", getter="c[{!r}]", default_decorators=["str"]):
        self.depth = 0
        self.m = PythonModule()
        self.variables = None
        self.default = default
        self.getter = getter
        self.declaredstore = _DeclaredStore()
        self.default_decorators = default_decorators

    def __call__(self, tokens, name="render", args="io, **c"):
        """
        from: ${x}
        create:
          def render(io, **context):
              context["x"]
        """
        with self.m.def_(name, args):
            self.variables = self.m.submodule()
            self.variables.stmt("write = io.write")
            # self.variables.stmt("get = c.get")
            # self.variables.stmt("M = object()")
            for t in tokens:
                self.visit(t)
        return self.m

    def visit(self, t):
        method = getattr(self, "visit_{}".format(t.__class__.__name__.lower()))
        method(t)

    def visit_text(self, node):
        for token, is_visit_var in node.body:
            if is_visit_var:
                self.m.stmt("write({})".format(self.calc_expr(token, emit=True)))
            else:
                self.m.stmt("write({!r})".format(token))
        self.m.stmt("write('\\n')")
        self.m.append(NEWLINE)

    def visit_doc(self, doc):
        if doc.multiline:
            self.m.stmt("########################################")
        for line in doc.body:
            self.m.stmt("# {}".format(line))
        if doc.multiline:
            self.m.stmt("########################################")
            self.m.sep()

    def visit_code(self, code):
        for line in code.body.body.split("\n"):  # xxx:
            self.m.stmt(line)
        self.declaredstore.push_frame(code.body.declared)
        self.m.sep()

    def calc_expr(self, expr, emit=False):
        io = StringIO()
        v = WithContextExprVistor(io, self.declaredstore, getter=self.getter)
        v.visit(expr.ast)
        result = io.getvalue()
        if emit:
            if expr.decorators:
                for f in expr.decorators:
                    result = "{}({})".format(f, result)
            for f in self.default_decorators:
                result = "{}({})".format(f, result)
        return result

    def visit_if(self, node):
        self.m.stmt("{} {}:".format(node.keyword, self.calc_expr(node.expr)))
        with self.m.scope():
            self._visit_children(node.body)

    def visit_for(self, node):
        self.m.stmt("{} {} in {}:".format(node.keyword, node.expr.body, self.calc_expr(node.src)))
        self.declaredstore.push_frame(node.expr.declared)
        with self.m.scope():
            self._visit_children(node.body)

    def _visit_children(self, node):
        if isinstance(node, list):
            for c in node:
                self._visit_children(c)
        else:
            self.visit(node)


class CacheManager(object):
    def __init__(self):
        self.cache = {}

    def __getitem__(self, s):
        return self.cache[hash(s)]

    def __setitem__(self, s, v):
        self.cache[hash(s)] = v

    def __contains__(self, s):
        return hash(s) in self.cache


_default_cache_manager = CacheManager()


class Template(object):
    def __init__(self, s, cache_manager=_default_cache_manager):
        self.s = s
        self.cache_manager = cache_manager

    def render(self, **kwargs):
        io = StringIO()
        self.get_render_function()(io, **kwargs)
        return io.getvalue()

    def get_render_function(self):
        if self.s in self.cache_manager:
            logger.debug("cached: hash=%s", hash(self.s))
            return self.cache_manager[self.s]
        else:
            lexer = Lexer()
            parser = Parser()
            compiler = Compiler()
            env = {}
            code = str(compiler(parser(lexer(self.s)), name="render"))
            logger.debug("compiled code:\n%s", code)
            exec(code, env)
            fn = self.cache_manager[self.s] = env["render"]
            return fn

if __name__ == "__main__":
    from kamo._sample import template
    print("========================================")
    print("input")
    print("========================================")
    print(template)
    print("========================================")
    print("compiled")
    print("========================================")
    lexer = Lexer()
    lexer(template)
    parser = Parser()
    parser(lexer.body)
    compiler = Compiler()
    compiler(parser.body)
    for i, line in enumerate(str(compiler.m).split("\n")):
        print("{:3< }: {}".format(i, line))
    env = {}
    exec(str(compiler.m), env)
    import sys
    print("========================================")
    print("output")
    print("========================================")
    env["render"](sys.stdout, x=10, xs=["foo", "bar", "boo"], hello="hello ", boo="(o_0)")
