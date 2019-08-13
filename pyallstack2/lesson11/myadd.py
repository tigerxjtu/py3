
def myadd(a,b):
    try:
        int_a = int(a)
        int_b = int(b)
    except Exception as e:
        print(e)
        return None

    int_sum = int_a+int_b
    return int_sum

if __name__ == '__main__':
    print(myadd(1,3))
    print(myadd(-1,3))
    print(myadd(1.5,3))
    print(myadd("8",3))
    print(myadd("abc",3))