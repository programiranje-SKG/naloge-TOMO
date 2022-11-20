

# import random
#
# def vsa_liha(sez):
#     for st in sez:
#         if st % 2 == 0:
#             return False
#     return True
#
# test_cases = [[random.randint(10,20) for _ in range(random.randint(4,7))] for _ in range(6)]
# # solutions = [round((sum(s) - min(s) - max(s)) / (len(s) - 2), 2) for s in test_cases]
# solutions = [max(s) * len(s) - sum(s) for s in test_cases]
#
# print(test_cases)
# print(solutions)

#
# test_cases = [[12, 66, 89, 6, 80, 44],
#               [47, 63, 5, 41, 11, 27, 67, 91],
#               [15, 17, 67, 77, 62, 8, 13],
#               [89, 78, 37, 60, 85, 54, 17, 38, 6, 90, 23],
#               [71, 35, 25, 7, 83, 99, 37],
#               [1, 83, 85, 95, 27, 19, 103, 97, 79, 69, 13, 1]]
#
# solutions = [False, True, False, False, True, True]

podatki = [["Ana", 55, 165], ["Berta", 60, 153]]
for ime, teza, visina in podatki:
    print(ime, round(teza / (visina / 100)**2, 2))


imena = ["Franc", "Janez", "Marko", "Andrej", "Ivan", "Marija", "Ana", "Maja", "Irena", "Mojca"]

