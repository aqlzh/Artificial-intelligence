import random


def get_rectangle():
    a = random.random()
    b = random.random()

    fat = int(a >= b)

    return a, b, fat


if __name__ == '__main__':
    print(get_rectangle())
