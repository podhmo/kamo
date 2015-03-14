# -*- coding:utf-8 -*-
template = """\
<%doc>
minimako
</%doc>

## this is comment

<%
def double(x):
    return x + x
%>

%if x == "10":
  ${hello|double|double}
  %if x == "20":
    ${boo}
  %endif
%endif

%for i,x in enumerate(xs):
* ${i} ${x}
%endfor
"""
