# -*- coding:utf-8 -*-
template = """\
<%doc>kamo</%doc>

## this is comment

<%
def double(x):
    return x + x
def rstrip(x):
    return x.rstrip()
%>

%if (x % 2) == 0:
${hello|double|double|rstrip}
  %if x == 10:
    ${boo}
  %endif
%endif

%for i,x in enumerate(xs):
* ${i} ${x}
%endfor
"""
