minimako
========================================

almost subset of mako.

::

  <%doc>
  minimako
  </%doc>

  ## this is comment

  <%
  def double(x):
      return x + x
  %>

  %if x == "10":
    ${hello}
  %else:
    ${hello|double}
  %endif

  %for i,x in enumerate(xs):
  * ${i} ${x}
  %endfor

  <%def name="element(tag='a')">
  <${tag}>${caller.body}</${tag}>
  </%def>

  <%self:element tag="p">
  hello
  </%self:element>


