# creates filenames
def filename(pref, suf, firstf, lastf):
    numb = range(firstf, lastf + 1)
    fname = []
    for i in range(len(numb)):
        if numb[i] < 10000:
            fname.append(pref + str(numb[i]).zfill(4) + suf)
        else:
            fname.append(pref + str(numb[i]) + suf)
    return fname
