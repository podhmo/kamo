# -*- coding:utf-8 -*-
from kamo import TemplateManager
import os.path
import logging
logging.basicConfig(level=logging.DEBUG)


m = TemplateManager(directories=[os.path.dirname(__file__)])
template = m.lookup("template.kamo")
print(template.render())
