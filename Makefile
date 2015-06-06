CC = gcc
CCFLAGS = -Wall -I/usr/local/browndeer/include
LDFLAGS = -L/usr/local/browndeer/lib -lcoprthr_opencl

all: hello

hello:
	$(CC) hello-opencl.c -o hello $(CCFLAGS) $(LDFLAGS)

clean:
	@rm hello

.PHONY: clean
