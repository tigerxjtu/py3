# Q1. 生产一个list,包含20以内的奇数，访问新生成序列的最后5个数据。
def Q1():
    mylist = list(range(1, 20, 2))
    for item in mylist[-5:]:
        print(item)


# Q2. 列表a = [1,2,3,4,2,5,6,2], 计算每个数字出现的次数，并将结果存储在字典中, 找到第二个2出现的索引位置。
def reducer(acc, item):
    index, value = item
    # if value in acc:
    #     acc[value]['count'] += 1
    #     acc[value]['index'].append(index + 1)
    # else:
    #     acc[value] = {"count": 1, "index": [index + 1]}
    # # print(acc)
    acc_value = acc.get(value, dict(count=0, index=[]))
    acc_value['count'] += 1
    acc_value['index'].append(index + 1)
    acc[value] = acc_value
    return acc


from functools import reduce


def Q2():
    a = [1, 2, 3, 4, 2, 5, 6, 2]
    result = reduce(reducer, enumerate(a), {})
    print(result)
    for key in result:
        print("%d: %d" % (key, result[key]["count"]))
    print("第二个2出现的索引位置是", result[2]['index'][1])


# Q3.  Products = [['iphone',6888],['MacPro',14800],['小米6',2499],['Coffee',31],['Book',60],['Nike',699]],将其转换为字典形式，结果如下:
#
# {'iphone': 6888,'MacPro': 14800, '小米6': 2499,'Coffee':31, 'Book': 60,
#
# 'Nike': 699}

def Q3():
    Products = [['iphone', 6888], ['MacPro', 14800], ['小米6', 2499], ['Coffee', 31], ['Book', 60], ['Nike', 699]]
    result = {product[0]: product[1] for product in Products}
    print(result)


if __name__ == '__main__':
    print('================第一题===================')
    Q1()
    print('================第二题===================')
    Q2()
    print('================第三题===================')
    Q3()
