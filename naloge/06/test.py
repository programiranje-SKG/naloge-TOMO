

def postevanka_7(n):
    sez = []
    for i in range(1, n + 1):
        if i % 7 == 0 or vsebuje_7(i):
            sez.append("BUM")
        else:
            sez.append(i)
    return sez

def vsebuje_7(n):
    while n > 0:
        if n % 10 == 7:
            return True
        n //= 10
    return False

def najdi_emso(zacetek: str):
    for i in range(10):
        trenutni = zacetek + str(i)
        if preveri_emso(trenutni):
            return trenutni


def preveri_emso(emso):
    vsota = 0
    for i in range(len(emso[:-1])):
        vsota += (7 - i % 6) * int(emso[i])
    return (vsota + int(emso[-1])) % 11 == 0

if __name__ == '__main__':
    # print(postevanka_7(100))
    # print(najdi_emso("211200750526"))
    print(preveri_emso("1203999500233"))