def se_napadata(top1, top2):
    return top1 != top2 and (top1[0] == top2[0] or top1[1] == top2[1])


def napadeni(top, topovi):
    nap = []
    for top2 in topovi:
        if se_napadata(top, top2):
            nap.append(top2)
    return nap


def napadenost(top, topovi):
    return len(napadeni(top, topovi))


def varen(top, topovi):
    return napadenost(top, topovi) == 0


def varni(topovi):
    var = []
    for top in topovi:
        if varen(top, topovi):
            var.append(top)
    return var


def najbolj_napaden(topovi):
    naj_top = None
    naj_nap = 0
    for top in topovi:
        nap = napadenost(top, topovi)
        if nap > naj_nap:
            naj_nap = nap
            naj_top = top
    return naj_top


def vse_varno(topovi):
    for top in topovi:
        if not varen(top, topovi):
            return False
    return True


def direkten_napad(top1, top2, topovi):
    nap = napadeni(top1, topovi)
    if top2 not in nap:
        return False

    if top1[0] == top2[0]:
        linija = 1
    else:
        linija = 0
    for top_vm in nap:
        if top1 != top_vm != top2 and \
                top1[1 - linija] == top_vm[1 - linija]  and \
                (top1[linija] > top_vm[linija]) != (top2[linija] > top_vm[linija]):
            return False
    return True


if __name__ == "__main__":
    pass
