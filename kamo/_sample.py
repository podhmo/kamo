# -*- coding:utf-8 -*-
template = """\
<%doc>kamo</%doc>

## this is comment

<%!
def double(x):
    return x + x

def rstrip(x):
    return x.rstrip()

%>

<%def name="tag(s, e)">
<${e}>${s}</${e}>
</%def>

${tag("foo", "a")}

%if (x % 2) == 0:
${hello|double|double|rstrip}
  % if x == 10:
    ## this is comment
<%doc> comment </%doc>
    <%c["xs"].append("hai")%>
    ${boo}
  % endif
%endif

%for i,x in enumerate(xs):
* ${i} ${x}
%endfor

a b c 
d
e
f
g
h
i
j
"""
