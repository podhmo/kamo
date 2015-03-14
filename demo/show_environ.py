# -*- coding:utf-8 -*-
from kamo import TemplateManager
import os.path
m = TemplateManager(directories=[os.path.dirname(__file__)])
template = m.lookup("template.kamo")
print(template.render())
