FLAGS = -I ${mkEigenInc} # -I ${mkLisInc} -L${mkLisLib} -llis
GCC = g++
MTX_ARGS = 

all: rerun
	@touch makefile

execlean:
	@rm -f **.exe

imgclean:
	@rm -f *.png *.mtx *.txt


allclean: execlean imgclean
	@touch makefile

recompile: allclean main.exe
	@echo "Recompiled main."


rerun: recompile
	./main.exe ${MTX_ARGS}

%.exe: %.cpp
	@$(GCC) $(FLAGS) $< -o $@
	