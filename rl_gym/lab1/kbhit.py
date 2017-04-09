import os
import msvcrt

class KBHit:


    def getch(self):
        return msvcrt.getch().decode('utf-8')

    def getarrow(self):
        d = msvcrt.getch()
        if d == b'\xe0':
            c = msvcrt.getch()
            vals = [75, 80, 77, 72]
            return vals.index(ord(c))
        else:
            return ord(d)

        return vals.index(ord(c.decode('utf-8')))

    def kbhit(self):
        return msvcrt.kbhit()
