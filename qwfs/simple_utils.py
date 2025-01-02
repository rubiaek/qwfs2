import datetime

def tnow():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
