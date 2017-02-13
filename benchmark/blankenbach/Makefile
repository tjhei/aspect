.PHONY: all

nproc=1

files:=$(wildcard case*.prm)
files:=$(basename $(files))
files:=$(addsuffix .stat, $(files))

all: $(files)
	./get_statistics *.stat

%.stat: %.prm
	@echo "** $*.prm -> output-$*/statistics -> $@"
	mpirun -n $(nproc) ./aspect $*.prm >/dev/null
	@cp output-$*/statistics $@
	#@rm -rf output-$*
	@echo created $@.
