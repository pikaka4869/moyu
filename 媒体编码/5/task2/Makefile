CC = gcc
CFLAGS = -O3 -march=native -Wall
LIBS = $(shell pkg-config --libs libavcodec libavformat libavutil libswscale) -lpthread
CFLAGS += $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale)
SRC = encoder_libx264.c
OUT = encoder_libx264

all:
	$(CC) $(CFLAGS) -o $(OUT) $(SRC) $(LIBS)

clean:
	rm -f $(OUT)
