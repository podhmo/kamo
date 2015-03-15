# -*- coding:utf-8 -*-
import logging
logger = logging.getLogger(__name__)
import ast
import re
import os.path
import tempfile
import shutil
import hashlib
import stat
from prestring.python import PythonModule
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
{deftag} :: '<%def' {defname} '>' {text} '</%def>'
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
begin_def = Intern("<%def>")
end_def = Intern("</%def>")
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
    ("\s*<%(!?)\s*(.+)\s*(?=%>)", lambda s, x: s.extend([begin_code, s.match.group(1), s.match.group(2)])),
    ('\s*<%doc>', lambda s, x: s.append(begin_doc)),
    ('\s*</%doc>', lambda s, x: s.append(end_doc)),
    ('\s*<%def\s*name="([^>]+)"\s*>', lambda s, x: s.extend([begin_def, s.match.group(1)])),
    ('\s*</%def>', lambda s, x: s.append(end_def)),
    ('\s*## (.*)', lambda s, x: s.extend((comment, s.match.group(1)))),
    ("\s*<%(!?)", lambda s, x: s.extend([begin_code, s.match.group(1)])),
    ("\s*%>", lambda s, x: s.append(end_doc)),
    ("\s*%\s*if", lambda s, x: s.append(begin_if)),
    ("\s*%\s*elif", lambda s, x: s.append(begin_elif)),
    ("\s*%\s*else", lambda s, x: s.append(begin_else)),
    ("\s*%\s*endif", lambda s, x: s.append(end_if)),
    ("\s*%\s*for", lambda s, x: s.append(begin_for)),
    ("\s*%\s*endfor", lambda s, x: s.append(end_for)),
    (".+", lambda s, x: s.append(x))
])


Doc = namedtuple("Doc", "body multiline")
Code = namedtuple("Code", "body ast declared is_module_level")
Def = namedtuple("Def", "body name args declared")
Text = namedtuple("Text", "body")
Expr = namedtuple("Expr", "body ast decorators declared")
If = namedtuple("If", "keyword expr body")  # xxx: include if, elif, else
For = namedtuple("For", "keyword expr src body")
Optimized = namedtuple("Optimized", "tokens")


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

    def parse_expr(self, expr, decorators=None, is_declared=False):  # hmm.
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
        elif t is begin_def:
            self.parse_def(tokens)
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
        is_module_level = bool(tokens[self.i])
        self.i += 1  # skip
        body = []
        while tokens[self.i] is not end_doc:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip
        body = "\n".join(body)
        ast_node = ast.parse(body)
        declared = collect_variable_name(ast_node)
        self.frame.append(Code(body,
                               ast_node,
                               declared=declared,
                               is_module_level=is_module_level))

    def parse_def(self, tokens):
        self.i += 1  # skip
        body = []
        arguments = tokens[self.i]
        name = arguments.split("(", 1)[0]
        args = [e.strip() for e in arguments[len(name) + 1:-1].split(",")]
        self.i += 1

        while tokens[self.i] is not end_def:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip

        parsedbody = []
        for token, is_emitting_var in split_with(self.emit_var_rx, "\n".join(body)):
            if is_emitting_var:
                token = token[2:-1]  # ${foo} -> foo
                token_with_filter = [e.strip(" ") for e in token.split("|")]  # foo|bar|boo -> [foo, bar, boo]
                token = token_with_filter[0]
                token = self.parse_expr(token, token_with_filter[1:])
            parsedbody.append((token, is_emitting_var))
        self.frame.append(Def([Text(parsedbody)], name, args, declared=set([name])))

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
        self.frame.append(("else", None))  # hmm.
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
        for token, is_emitting_var in split_with(self.emit_var_rx, tokens[self.i]):
            if is_emitting_var:
                token = token[2:-1]  # ${foo} -> foo
                token_with_filter = [e.strip(" ") for e in token.split("|")]  # foo|bar|boo -> [foo, bar, boo]
                token = token_with_filter[0]
                token = self.parse_expr(token, token_with_filter[1:])
            body.append((token, is_emitting_var))
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

    def pop_frame(self):
        self.stack.pop()


class Optimizer(object):
    def optimize(self, tokens, text, result):
        last_is_text = False
        for t in tokens:
            if isinstance(t, Text):
                emitting_status = False
                for pair in t.body:
                    if pair[1] == emitting_status:  # emitting_status
                        text.body.append(pair)
                    else:
                        emitting_status = not emitting_status
                        self.compact(text)
                        result.append(text)
                        text = Text([pair])
                if text.body[-1][1] is False:
                    text.body.append(("\n", False))
                else:
                    self.compact(text)
                    result.append(text)
                    text = Text([("\n", False)])
                last_is_text = True
            else:
                if last_is_text:
                    self.compact(text)
                    result.append(text)
                    text = Text([("", False)])
                    last_is_text = False

                if isinstance(t, If):
                    body = []
                    self.optimize(t.body, Text([("", False)]), body)
                    result.append(If(t.keyword, t.expr, body))
                elif isinstance(t, For):
                    body = []
                    self.optimize(t.body, Text([("", False)]), body)
                    result.append(For(t.keyword, t.expr, t.src, body))
                elif isinstance(t, Def):
                    body = []
                    self.optimize(t.body, Text([("", False)]), body)
                    result.append(Def(body, t.name, t.args, t.declared))
                else:
                    result.append(t)

        if last_is_text:
            self.compact(text)
            result.append(text)

    def compact(self, text):
        if text.body[0][1] is False:  # text
            body = "".join(pair[0] for pair in text.body)
            text.body.clear()
            text.body.append((body, False))
        if text.body[0][0] == "":
            text.body.pop(0)

    def __call__(self, tokens):
        r = []
        self.optimize(tokens, Text([("", False)]), r)
        self.body = Optimized(r)
        return self.body


class Compiler(object):
    def __init__(self, m=None, default="''", getter="c[{!r}]", default_decorators=["str"]):
        self.depth = 0
        self.m = PythonModule()
        self.variables = None
        self.toplevel = None
        self.default = default
        self.getter = getter
        self.declaredstore = _DeclaredStore()
        self.default_decorators = default_decorators
        self.optimized = False

    def __call__(self, tokens, name="render", args="io, **c"):
        """
        from: ${x}
        create:
          def render(io, **context):
              context["x"]
        """
        if isinstance(tokens, Optimized):
            tokens = tokens.tokens
            self.optimized = True

        self.toplevel = self.m.submodule()
        with self.m.def_(name, args):
            self.variables = self.m.submodule()
            self.variables.stmt("write = io.write")
            # self.variables.stmt("get = c.get")
            # self.variables.stmt("M = object()")
            for t in tokens:
                self.visit(t)
        self.optimized = False
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
        if not self.optimized:
            self.m.stmt("write('\\n')")

    def visit_doc(self, doc):
        if doc.multiline:
            self.m.stmt("########################################")
        for line in doc.body:
            self.m.stmt("# {}".format(line))
        if doc.multiline:
            self.m.stmt("########################################")
            self.m.sep()

    def visit_code(self, code):
        m = self.toplevel if code.is_module_level else self.m
        for line in code.body.split("\n"):  # xxx:
            m.stmt(line)
        self.declaredstore.stack[-1].update(code.declared)
        m.sep()

    def visit_def(self, node):
        self.declaredstore.stack[-1].update(node.declared)
        with self.m.def_(node.name, *node.args):
            try:
                self.declaredstore.push_frame(set(node.args))
                for text in node.body:
                    self.visit_text(text)
                self.m.return_("''")
            finally:
                self.declaredstore.pop_frame()

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
        if node.expr is None:  # else
            self.m.stmt("{}:".format(node.keyword))
        else:
            self.m.stmt("{} {}:".format(node.keyword, self.calc_expr(node.expr)))
        with self.m.scope():
            self._visit_children(node.body)

    def visit_for(self, node):
        self.m.stmt("{} {} in {}:".format(node.keyword, node.expr.body, self.calc_expr(node.src)))
        self.declaredstore.push_frame(node.expr.declared)
        try:
            with self.m.scope():
                self._visit_children(node.body)
        finally:
            self.declaredstore.pop_frame()

    def _visit_children(self, node):
        if isinstance(node, list):
            for c in node:
                self._visit_children(c)
        else:
            self.visit(node)


class TemplateNotFound(Exception):
    pass


class TemplateManager(object):
    def __init__(self, directories=["."], optimize=True, tmpdir=tempfile.gettempdir()):
        self.directories = directories
        self.template_cache = {}
        self.module_cache = {}  # xxx?
        self.optimize = optimize
        self.tmpdir = tmpdir

    def lookup(self, filename):
        if filename in self.template_cache:
            return self.template_cache[filename]
        for d in self.directories:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                return self.load_template(filename, path)
        raise TemplateNotFound(filename)

    def load_template(self, filename, path):
        template = Template(None,
                            module_id=filename,
                            path=path,
                            tmpdir=self.tmpdir,
                            manager=self,
                            optimize=self.optimize)
        self.template_cache[filename] = template
        return template

    def create_template(self, s):
        return Template(s,
                        module_id=None,
                        path=None,
                        tmpdir=self.tmpdir,
                        manager=self,
                        optimize=self.optimize)
        self.template_cache[template.module_id] = template
        return template

    def load_module(self, module_id, path):
        if module_id in self.module_cache:
            logger.info("cached: module_id=%s", module_id)
            return self.module_cache[module_id]
        try:
            module_path = os.path.join(self.tmpdir, module_id)
            if path is not None and os.path.exists(path) and os.path.exists(module_path):
                file_mtime = os.stat(path)[stat.ST_MTIME]
                if file_mtime >= os.stat(module_path)[stat.ST_MTIME]:
                    logger.info("cache is obsoluted: module_id=%s (mtime=%s)", module_id, file_mtime)
                    return None
            module = load_module(module_id, module_path)
            self.module_cache[module_id] = module
            return module
        except FileNotFoundError:
            return None


default_manager = TemplateManager([])


class Template(object):
    def __init__(self, source=None, module_id=None, path=None, tmpdir=None,
                 manager=default_manager, optimize=True, nocache=False):
        # from file path is not None, from string source is not None
        self._source = source
        self.module_id = module_id or hashlib.md5(source.encode("utf-8")).hexdigest()
        self.tmpdir = tmpdir
        self.path = path
        self.manager = manager
        self.optimize = optimize
        self.nocache = nocache

    @property
    def source(self):
        if self._source is not None:
            return self._source
        with open(self.path) as rf:
            self._source = rf.read()
            return self._source

    def render(self, **kwargs):
        io = StringIO()
        self.get_render_function()(io, **kwargs)
        return io.getvalue()

    @property
    def code(self):
        import inspect
        return inspect.getsourcefile(self.get_render_module())

    def get_render_module(self):
        module = None
        if not self.nocache:
            module = self.manager.load_module(self.module_id, self.path)
        if module is not None:
            return module
        module = self.compile()
        return module

    def get_render_function(self):
        return self.get_render_module().render

    def _compile(self, source):
        lexer = Lexer()
        parser = Parser()
        compiler = Compiler()
        if self.optimize:
            optimizer = Optimizer()
            return compiler(optimizer(parser(lexer(source))), name="render")
        else:
            return compiler(parser(lexer(source)), name="render")

    def compile(self):
        code = str(self._compile(self.source))
        return _compile(self.module_id, code, tmpdir=None)


def load_module(module_id, path):
    from importlib import machinery
    return machinery.SourceFileLoader(module_id, path).load_module()


def _compile(module_id, code, tmpdir=None):
    tmpdir = tmpdir or tempfile.gettempdir()
    logger.debug("compiled code:\n%s", code)
    fd, path = tempfile.mkstemp()
    os.write(fd, code.encode("utf-8"))
    dst = os.path.join(tmpdir, module_id)
    logger.info("generated module file: %s", dst)
    shutil.move(path, dst)
    return load_module(module_id, dst)


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
    optimizer = Optimizer()
    optimizer(parser.body)
    compiler = Compiler()
    compiler(optimizer.body)
    for i, line in enumerate(str(compiler.m).split("\n")):
        print("{:3< }: {}".format(i, line))
    env = {}
    exec(str(compiler.m), env)
    import sys
    print("========================================")
    print("output")
    print("========================================")
    env["render"](sys.stdout, x=10, xs=["foo", "bar", "boo"], hello="hello ", boo="(o_0)")
