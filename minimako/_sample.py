# -*- coding:utf-8 -*-
template = """\
<%doc>minimako</%doc>

## this is comment

<%
def double(x):
    return x + x
%>

%if (x % 2) == 0:
${hello|double|double}
  %if x == 10:
    ${boo}
  %endif
%endif

%for i,x in enumerate(xs):
* ${i} ${x}
%endfor
"""
