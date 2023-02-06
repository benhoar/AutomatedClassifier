import logging

def source_of_logs():
	logging.debug("debug info")
	logging.info("just some info")
	logging.error("uh oh :(")

def main():
   level = logging.DEBUG
   fmt = '[%(levelname)s] %(asctime)s - %(message)s'
   logging.basicConfig(level=level, format=fmt)
   source_of_logs()

if __name__ == "__main__":
   main()