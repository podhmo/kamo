kamo
========================================

almost subset of mako.

.. code-block:: python

  # -*- coding:utf-8 -*-
  import logging
  from kamo import Template
  # logging.basicConfig(level=logging.DEBUG)


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

generated function is such as below.

.. code-block:: python

  def render(io, **c):
      write = io.write
      def decorate(s):
          return "** {} **".format(s)

      from datetime import datetime

      write(str(decorate(c['greeting'])))
      write('\n')

      write(str(c['name']))
      write(': this is my first sample! (now: ')
      write(str(datetime.now()))
      write(')')
      write('\n')


