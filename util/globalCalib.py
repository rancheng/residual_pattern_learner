import util.settings as settings
import numpy as np

wG = []
hG = []
fxG = []
fyG = []
cxG = []
cyG = []
fxiG = []
fyiG = []
cxiG = []
cyiG = []
KG = []
KiG = []

wM3G = 0
hM3G = 0

def setGlobalCalib(w, h, K):
    K = np.array(K)
    wlvl = w
    hlvl = h
    while wlvl%2 == 0 and hlvl%2 ==0 and wlvl*hlvl > 5000 and settings.pyrLevelsUsed < settings.PYR_LEVELS:
        wlvl /= 2
        hlvl /= 2
        settings.pyrLevelsUsed += 1
    wM3G = w - 3
    hM3G = h - 3
    wG.append(w)
    hG.append(h)
    KG.append(K)
    fxG.append(K[0,0])
    fyG.append(K[1,1])
    cxG.append(K[0,2])
    cyG.append(K[1,2])
    KiG.append(np.linalg.inv(K))
    fxiG.append(KiG[0][0,0])
    fyiG.append(KiG[0][1,1])
    cxiG.append(KiG[0][0,2])
    cyiG.append(KiG[0][1,2])

    for level in range(1, settings.pyrLevelsUsed):
        wG.append(w >> level)
        hG.append(h >> level)
        fxG.append(fxG[-1] * 0.5)
        fyG.append(fyG[-1] * 0.5)
        cxG.append((cxG[0] + 0.5) / (int(1 << level)) - 0.5)
        cyG.append((cyG[0] + 0.5) / (int(1 << level)) - 0.5)

        Klvl = np.array([[fxG[-1], 0, cxG[-1]],
                         [0, fyG[-1], cyG[-1]],
                         [0,       0,       1]
                         ])

        KG.append(Klvl)
        KiG.append(np.linalg.inv(KG[-1]))
        fxiG.append(KiG[-1][0,0])
        fyiG.append(KiG[-1][1,1])
        cxiG.append(KiG[-1][0,2])
        cyiG.append(KiG[-1][1,2])
