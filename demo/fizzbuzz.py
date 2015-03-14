from kamo import Template
template = Template("""
%for x in range(1, N):
  %if x % 15 == 0:
"fizzbuzz"
  %elif x % 3 == 0:
"fizz"
  %elif x % 5 == 0:
"buzz"
  %else:
${x}
  %endif
%endfor
""")

print(template.render(N=100))
