def testna_nalog(arg):
    """
    ta naloga zgolj vrne vhodni argument
    """
    pass


def se_napadata(top1, top2):
    """
    Prejme koordinati dveh topov in vrne True, če se med seboj napadata (torej: če sta v isti vrstici ali stolpcu)
    in False, če se ne. Če gre za eno in isto trdnjavo, pa vedno vrne False.

    Primer klicev:
        >>> se_napadata("a5", "a8")
        True
        >>> se_napadata("c5", "f5")
        True
        >>> se_napadata("a5", "c3")
        False
        >>> se_napadata("c3", "c3")
        False
    """
    pass


def napadeni(top, topovi):
    """
    Funkcija prejme koordinato enega topa in seznam koordinat vseh topov na šahovnici. Vrniti mora seznam koordinat
    vseh topov, ki jih napada podani top. Napadeni topovi naj bodo našteti v enakem vrstnem redu, v katerem se pojavijo
    v seznamu topovi. S tem se ti ni treba posebej ukvarjati: če ne boš kaj brez potrebe mešal(a), se bo zgodilo
    samo od sebe.

    Primer klicev:
        >>> napadeni("c3", ["c1", "c3", "d6", "c6", "e5", "a3"])
        ["c1", "c6", "a3"]
        >>> napadeni("c1", ["c1", "c3", "c4", "c5"])
        ["c3", "c4", "c5"]
    """
    pass


def napadenost(top, topovi):
    """
    Funkcija prejme koordinato topa in koordinate vseh topov ter vrne število topov, ki napadajo podani top.
    Napadenost topa f5 na sliki je 3, saj ga napadajo trije topovi (vključno s topom a5, ker lahko
    le-ta preskoči top c5).

    Primer klicev:
        >>> napadenost("c3", ["c1", "c3", "d6", "c6", "e5", "a3"])
        3
        >>> napadenost("c1", ["c1", "c3", "c4", "c5"])
        3
    """
    pass


def varen(top, topovi):
    """
    Funkcija vrne True, če podani top ni napaden in False, če je. Na sliki je le en top, za katerega bi ta funkcija
    vrnila True: top d3.

    Primer klicev:
        >>> varen("c3", ["c1", "c3", "d6", "c6", "e5", "a3"])
        False
        >>> varen("d3", ["a5", "c5", "c6", "c8", "d3", "f5", "f7"])
        True
    """
    pass


def varni(topovi):
    """
    Vrne seznam varnih topov.

    Primer klicev:
        >>> varni(["c1", "c3", "d6", "c6", "e5", "a3"])
        ["e5"]
        >>> varni(["a5", "c5", "c6", "c8", "d3", "f5", "f7"])
        ["d3"]
        >>> varni(["a5", "c5", "c6", "c8", "f5", "f7"])
        []
    """
    pass


def najbolj_napaden(topovi):
    """
    Funkcija prejme seznam topov in vrne koordinati topa, ki ga napada največ drugih topov. Če je enako napadenih
    več, vrne tistega, ki je prej na seznamu. Če ni napaden nihče, vrne None.

    Za topove na sliki ta funkcija vrne "c5", saj ga napadajo štirje drugi topovi.

    Primer klicev:

         >>> najbolj_napaden(["a5", "c5", "c6", "c8", "d3", "f5", "f7"])
         "c5"
    """
    pass


def vse_varno(topovi):
    """
    Funckcija vrne True, če noben top ne napada nobenega drugega. Na sliki ni vse varno. Nikakor.

    Primer klicev:

        >>> vse_varno(["a5", "c5", "c6", "c8", "d3", "f5", "f7"])
        False

    """
    pass


def direkten_napad(top1, top2, topovi):
    """
    (Težja naloga)

    Funkcija prejme koordinati dveh topov in seznam vseh topov. Funkcija vrne True, če se top1 in top2 napadata -
    seveda tako, da med njima ni nobenega drugega topa.

    Funkcija, se_napadata(top1, top2), ki smo jo napisali prej, bi za topa a5 in f5 vrnila True, četudi je med njima
    top na c5. Funkcija, ki jo pišemo zdaj, pa mora v takem primeru vrniti False.

    Primer klica:

        >>> direkten_napad("a5", "f5", ["a5", "c5", "c6", "c8", "d3", "f5", "f7"])
        False

    """
    pass


if __name__ == "__main__":
    # tukaj lahko testiramo funkcije
    pass
