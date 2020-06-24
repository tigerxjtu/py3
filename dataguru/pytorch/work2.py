
# 回数是指从左向右读和从右向左读都是一样的数，例如12321，909。请利用filter()筛选出回数

def is_huishu(num):
    v=str(num)
    return v==v[::-1]

nums=[12321,12422,909,3,898978989,8989768989,8989779898]
results=filter(is_huishu,nums)
for r in results:
    print(r)