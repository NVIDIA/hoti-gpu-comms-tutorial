.PHONY: all clean lsa gin run-lsa_SOLVED run-gin_SOLVED

all: lsa gin

lsa:
	$(MAKE) -C 01-lsa-device-api all

gin:
	$(MAKE) -C 02-gin-put-device-api all

run-lsa_SOLVED:
	$(MAKE) -C 01-lsa-device-api run_SOLVED

run-gin_SOLVED:
	$(MAKE) -C 02-gin-put-device-api run_SOLVED

clean:
	$(MAKE) -C 01-lsa-device-api clean
	$(MAKE) -C 02-gin-put-device-api clean
