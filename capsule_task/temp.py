import os

ls = os.listdir("./test_0718_32")

result = dict()
for path in ls:
   name = path.split(".")
   
   result[name[0].split('_')[0]] = name[0].split('_')[0]
   list = []
   
   
   print(,",", name[1][-6:])