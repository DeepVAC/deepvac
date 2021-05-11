from datetime import datetime

def getPrintTime():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')