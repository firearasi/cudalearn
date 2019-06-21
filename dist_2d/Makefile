NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main.exe

main.exe: kernel.o 
	$(NVCC) $^ -o $@

kernel.o: kernel.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
