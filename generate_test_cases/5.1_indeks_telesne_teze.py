import random


def solve(test_cases):
    return [f"{ime} {round(teza / (visina / 100) ** 2, 2)}" for ime, teza, visina in test_cases]


imena = ["Franc", "Janez", "Marko", "Andrej", "Ivan", "Marija", "Ana", "Maja", "Irena", "Mojca"]
n = 8
length = 5
test_cases = [
    [
        (random.choice(imena), random.randint(50, 90), random.randint(160, 190))
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
