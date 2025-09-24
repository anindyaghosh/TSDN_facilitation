def naming_convention(i):
    # zfill pads string with zeros from leading edge until len(string) = 6
    return 'IMG' + str(i).zfill(6)