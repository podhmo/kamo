# -*- coding:utf-8 -*-
import logging
from kamo import Template
logging.basicConfig(level=logging.DEBUG)


template = Template("""
<%
def decorate(s):
    return "** {} **".format(s)
%>
<%from datetime import datetime%>

${greeting|decorate}
${name}: this is my first sample! (now: ${datetime.now()})
""")


print(template.render(name="foo", greeting="chears"))
print("----------------------------------------")
print(template.render(name="boo", greeting="chears"))
