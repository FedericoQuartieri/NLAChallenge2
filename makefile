FLAGS = -I ${mkEigenInc} # -I ${mkLisInc} -L${mkLisLib} -llis
GCC = g++
MTX_ARGS = a2.mtx w.mtx sol.txt hist.txt

all: rerun
	@touch makefile

execlean:
	@rm -f **.exe

imgclean:
	@rm -f *.png *.mtx *.txt

recompile: allclean main.exe
	@echo "Recompiled main."
allclean: execlean imgclean
	@touch makefile

lis:
	@mpirun -n 4 ./test1 a2.mtx w.mtx x.mtx convergence_history.mtx -tol 1.0e-9 -maxiter 2000 -i bicgstab -p ilu
rerun: recompile
	./main.exe ${MTX_ARGS}

%.exe: %.cpp
	@$(GCC) $(FLAGS) $< -o $@
	
#%.cpp:
#	$(GCC) $(FLAGS) $@ -o $@.exe
