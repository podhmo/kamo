# -*- coding:utf-8 -*-
import logging
logger = logging.getLogger(__name__)
import ast
import re
from functools import partial
from collections import namedtuple
# newline, multiline ?

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
comment = Intern("##")
begin_code = Intern("<%")
end_code = Intern("%>")
begin_if = Intern("%if")
begin_else = Intern("%else")
begin_elif = Intern("%elif")
end_if = Intern("%endif")
begin_for = Intern("%for")
end_for = Intern("%endfor")
begin_paren = Intern("(")
end_paren = Intern(")")


class Lexer(re.Scanner):
    def __init__(self, *args, **kwargs):
        super(Lexer, self).__init__(*args, **kwargs)
        self.body = []

    def append(self, x):
        self.body.append(x)

    def __call__(self, body):
        for line in body.split("\n"):
            self.scan(line)


Scanner = partial(Lexer, [
    ('\s*<%doc>', lambda s, x: s.append(begin_doc)),
    ('\s*</%doc>', lambda s, x: s.append(end_doc)),
    ('\s*## .*', lambda s, x: s.append(comment)),
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
Doc = namedtuple("Doc", "body")
Code = namedtuple("Code", "body")
Text = namedtuple("Text", "body")
If = namedtuple("If", "keyword predicate body")  # xxx: include if, elif, else
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

    def parse_expr(self, expr):  # hmm.
        return ast.parse(expr).body[0]

    def __call__(self, tokens):
        self.i = 0
        n = len(tokens)
        while n > self.i:
            self.parse_statement(tokens)

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
            self.frame.append(Text(tokens[self.i]))
            self.i += 1

    def parse_doc(self, tokens):
        self.i += 1  # skip
        body = []
        while tokens[self.i] is not end_doc:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip
        self.frame.append(Doc(body))

    def parse_comment(self, tokens):
        self.frame.append(Doc([tokens[self.i]]))
        self.i += 1

    def parse_code(self, tokens):
        self.i += 1  # skip
        body = []
        while tokens[self.i] is not end_doc:
            body.append(tokens[self.i])
            self.i += 1
        self.i += 1  # skip
        self.frame.append(Code(body))

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
        expr = self.parse_expr(expr.strip(" "))
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


if __name__ == "__main__":
    from minimako._sample import template
    scanner = Scanner()
    scanner(template)
    parser = Parser()
    parser(scanner.body)
    print(parser.body)
    #print(scanner.body)
    
