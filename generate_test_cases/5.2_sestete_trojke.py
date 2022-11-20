import random

vhod = [(3, 5, 8), (2, 6, 9), (1, 1, 2), (10, 5, 15)]


def solve(cases):
    return all([c == a + b for a, b, c in cases])


n = 8
length = 5
test_cases = [
    [
        (a := random.randint(1, 9), b := random.randint(1, 9),
         a + b if random.choice([True] * 10 + [False]) else random.randint(1, 9))
        for _ in range(5)
    ]
    for _ in range(n)
]
print("Test cases:")
print("-----------------------")
print("podatki =", test_cases)
print("-----------------------")
print("solutions =", [solve(test_case) for test_case in test_cases])
print("-----------------------")
