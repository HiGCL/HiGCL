import datetime

def log(msg):
	time = datetime.datetime.now()
	print("%s: %s" % (time, msg))


