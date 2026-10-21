# FIT3143 Lab 3 - build the WSN simulation
CC     = mpicc
CFLAGS = -fopenmp -O2 -Wall
SRC    = wsn_main.c wsn_common.c task1_node.c task2_base.c

wsn: $(SRC) wsn_common.h
	$(CC) $(CFLAGS) -o wsn $(SRC)

clean:
	rm -f wsn
