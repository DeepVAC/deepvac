import sys
is_ddp = False
if '--rank' in sys.argv:
    is_ddp = True