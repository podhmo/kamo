# -*- coding:utf-8 -*-
from kamo import Template
from datetime import date

template = Template("""
<%doc>
cal.kamo
</%doc>
<%
import calendar
from datetime import date
month = ["","睦月","如月","弥生","卯月","皐月","水無月","文月","葉月","長月","神無月","霜月","師走"]

def paren(i):
    return "({})".format(i)
%>
# ${today.year}年
========================================

%for i in range(1, 13):
${i}月${month[i]|paren}
----------------------------------------

  %for d in range(1, calendar.monthrange(today.year, i)[1]):
    %if date(today.year, i,  d) <= today:
- ${d} ☓
    %else:
- ${d}
    %endif
  %endfor
%endfor
""")
import logging
logging.basicConfig(level=logging.DEBUG)
today = date.today()
print(template.render(today=today))
