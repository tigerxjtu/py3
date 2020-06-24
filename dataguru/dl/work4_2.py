
list = ['Apple', 'orange', 'Peach', 'banana']

def my_sort(s):
    return s.lower()

result = sorted(list,key=my_sort)
print(result)
