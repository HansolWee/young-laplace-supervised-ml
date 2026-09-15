FC = gfortran
FFLAGS = -O3 -ffree-line-length-none
LDFLAGS =
LDLIBS =

COMMON = mod_yl.o mod_yl_benchmark.o
NEWTON_OBJECTS = $(COMMON) main_yl.o
EVAL_OBJECTS = $(COMMON) mod_yl_python.o main_yl_evaluate.o

.PHONY: all run debug clean

all: yl yl_evaluate

yl: $(NEWTON_OBJECTS)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

yl_evaluate: $(EVAL_OBJECTS)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

mod_yl.o: mod_yl.f90
	$(FC) $(FFLAGS) -c $< -o $@

mod_yl_benchmark.o: mod_yl_benchmark.f90 mod_yl.o
	$(FC) $(FFLAGS) -c $< -o $@

mod_yl_python.o: mod_yl_python.f90 mod_yl.o mod_yl_benchmark.o
	$(FC) $(FFLAGS) -c $< -o $@

main_yl.o: main_yl.f90 mod_yl.o mod_yl_benchmark.o
	$(FC) $(FFLAGS) -c $< -o $@

main_yl_evaluate.o: main_yl_evaluate.f90 mod_yl.o mod_yl_python.o
	$(FC) $(FFLAGS) -c $< -o $@

run: yl
	./yl

debug:
	$(MAKE) clean
	$(MAKE) FFLAGS="-O0 -g -Wall -Wextra -fcheck=all -fbacktrace -ffree-line-length-none" all

clean:
	$(RM) yl yl_evaluate $(NEWTON_OBJECTS) $(EVAL_OBJECTS)
	$(RM) mod_yl.mod mod_yl_benchmark.mod mod_yl_python.mod
