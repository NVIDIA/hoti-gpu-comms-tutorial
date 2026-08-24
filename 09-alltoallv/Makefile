SUBDIRS := \
	01-nvshmem-alltoallv \
	02-nccl-lsa-alltoallv \
	03-nccl-gin-alltoallv \
	04-nccl-lsa-gin-alltoallv

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for directory in $(SUBDIRS); do $(MAKE) -C $$directory clean; done
