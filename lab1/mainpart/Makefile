all:
	mpicc main.c -o ./build/task -lm

test:
	mpicc main.c -o ./build/task -lm

run: 
	mpirun -n 9 ./build/task 100 100

clear:
	rm ./build/task
	rm *.txt