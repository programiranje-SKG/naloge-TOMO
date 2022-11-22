import random

korpus = ["adijo", "aerobika", "afna", "agencija", "akcija", "aktiven", "album", "Američan, Američanka", "Amerika",
          "ameriški", "ampak", "angleščina", "angleško", "Argentina", "Argentinec, Argentinka", "arhitekt, arhitektka",
          "Avstralija", "Avstrija", "Avstrijec, Avstrijka", "avto", "avtobus", "avtobusna postaja", "avtomat",
          "avtomehanik", "Babica", "balkon", "banana", "banka", "bankomat", "bar", "bel", "bencinska črpalka",
          "beseda", "besedilo", "biolog, biologinja", "biologija", "biti", "blagajna", "blizu", "blok", "bluza",
          "bolan", "boleti", "bolj", "bolniška", "bolnišnica", "borovničev", "božič", "brati", "brez", "burek",
          "Cel", "cena", "cenik", "center", "cesta", "cigareta", "copati", "Čaj", "čakati", "čao", "čas", "časopis",
          "častiti", "čestitati", "čestitka", "češki", "četrtek", "čevlji", "čez", "čistiti", "človek", "čokolada",
          "čokoladni", "črn", "čudovit", "Da", "daleč", "dan", "danes", "darilni bon", "darilo", "dati",
          "datum rojstva", "debel", "dedek", "delati", "delno", "delo", "delovni zvezek", "denar", "deset",
          "desno", "devet", "dež", "deževati", "direktor, direktorica", "dnevna soba", "dober", "dober dan",
          "dober tek", "dober večer", "dobiti", "dobro jutro", "dokument", "dolar", "dolg", "dolgčas",
          "dolgočasen", "dom", "doma", "domača naloga", "domov", "dopoldne", "dopust", "dosegljiv", "dovolj", "drag"]

n = 8
def solve(a, b):
    return sum([1 for i, j in zip(a, b) if i == j])


test_cases = [
    [random.choice(korpus).upper(), random.choice(korpus).upper()]
    for _ in range(n)
]
print("Test cases:")
print("-----------------------")
print("podatki =", test_cases)
print("-----------------------")
print("solutions =", [solve(*test_case) for test_case in test_cases])
print("-----------------------")
