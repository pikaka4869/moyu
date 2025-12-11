/* encoder_libx264.c
 Sample low-latency H.264 encoder wrapper using libavcodec (libx264).
 Notes:
  - Requires FFmpeg development libraries (libavcodec, libavformat, libavutil, libswscale).
  - Compile with pkg-config:
      gcc -o encoder_libx264 encoder_libx264.c $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale) -lpthread
  - Reads raw YUV420p "test_cif.yuv" (CIF 352x288) and writes "out_lowlatency.h264".
  - Tuned for low-latency: no B-frames, small GOP, "zerolatency" tune, "veryfast" preset.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <stdatomic.h>

#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>

#define WIDTH 352
#define HEIGHT 288
#define FRAMES 150

typedef struct FrameItem {
    uint8_t *data; // raw YUV420p
    int64_t pts;
} FrameItem;

#define RING_CAP 16
FrameItem* ring[RING_CAP];
_Atomic int ring_head = 0;
_Atomic int ring_tail = 0;

static inline int ring_push(FrameItem* f){
    int t = ring_tail;
    int n = (t + 1) % RING_CAP;
    if (n == ring_head) return 0;
    ring[t] = f;
    ring_tail = n;
    return 1;
}
static inline int ring_pop(FrameItem** out){
    int h = ring_head;
    if (h == ring_tail) return 0;
    *out = ring[h];
    ring_head = (h+1)%RING_CAP;
    return 1;
}

volatile int running = 1;
const char *input_yuv = "test_cif.yuv";
const char *out_h264 = "out_lowlatency.h264";

void* capture_thread(void* arg){
    FILE *f = fopen(input_yuv, "rb");
    if (!f){ perror("fopen input"); running=0; return NULL;}
    int frame_size = WIDTH*HEIGHT + 2*(WIDTH/2)*(HEIGHT/2);
    for(int i=0;i<FRAMES && running;i++){
        uint8_t *buf = (uint8_t*)av_malloc(frame_size);
        size_t r = fread(buf,1,frame_size,f);
        if (r != frame_size){ av_free(buf); break;}
        FrameItem* it = (FrameItem*)malloc(sizeof(FrameItem));
        it->data = buf;
        it->pts = i;
        while(!ring_push(it)) usleep(500);
        usleep(33333); // simulate ~30fps capture
    }
    fclose(f);
    running=0;
    return NULL;
}

int main(int argc, char** argv){
    avcodec_register_all();

    const AVCodec *codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) {
        fprintf(stderr,"libx264 encoder not found. Ensure libx264 is installed and libavcodec built with it.\n");
        return -1;
    }
    AVCodecContext *c = avcodec_alloc_context3(codec);
    if (!c) return -1;
    c->bit_rate = 500000;
    c->width = WIDTH;
    c->height = HEIGHT;
    c->time_base = (AVRational){1,30};
    c->framerate = (AVRational){30,1};
    c->gop_size = 30;
    c->max_b_frames = 0; // low-latency: no B-frames
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    // Low-latency tuning via options
    av_opt_set(c->priv_data, "preset", "veryfast", 0);
    av_opt_set(c->priv_data, "tune", "zerolatency", 0);
    av_opt_set(c->priv_data, "profile", "baseline", 0);
    av_opt_set(c->priv_data, "x264-params", "nal-hrd=cbr:force-cfr=1:keyint=30", 0);

    if (avcodec_open2(c, codec, NULL) < 0){
        fprintf(stderr,"Could not open codec\n");
        return -1;
    }

    FILE *out = fopen(out_h264, "wb");
    if (!out){ perror("fopen out"); return -1; }

    pthread_t cap_thread;
    pthread_create(&cap_thread, NULL, capture_thread, NULL);

    AVFrame *frame = av_frame_alloc();
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    av_image_alloc(frame->data, frame->linesize, c->width, c->height, c->pix_fmt, 32);

    while(running || ring_head != ring_tail){
        FrameItem* it = NULL;
        if (!ring_pop(&it)) { usleep(100); continue; }
        // copy raw YUV into AVFrame planes
        uint8_t *src = it->data;
        int y_size = WIDTH*HEIGHT;
        int uv_size = (WIDTH/2)*(HEIGHT/2);
        // Y
        for(int r=0;r<HEIGHT;r++){
            memcpy(frame->data[0] + r*frame->linesize[0], src + r*WIDTH, WIDTH);
        }
        // U
        uint8_t *src_u = src + y_size;
        for(int r=0;r<HEIGHT/2;r++){
            memcpy(frame->data[1] + r*frame->linesize[1], src_u + r*(WIDTH/2), WIDTH/2);
        }
        // V
        uint8_t *src_v = src + y_size + uv_size;
        for(int r=0;r<HEIGHT/2;r++){
            memcpy(frame->data[2] + r*frame->linesize[2], src_v + r*(WIDTH/2), WIDTH/2);
        }
        frame->pts = it->pts;

        // encode
        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;
        if (avcodec_send_frame(c, frame) == 0) {
            while(avcodec_receive_packet(c, &pkt) == 0){
                fwrite(pkt.data,1,pkt.size,out);
                av_packet_unref(&pkt);
            }
        }
        av_free(it->data);
        free(it);
    }

    // flush encoder
    avcodec_send_frame(c, NULL);
    // Note: flushing loop omitted for brevity (above loop handles most)

    fclose(out);
    av_freep(&frame->data[0]);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    pthread_join(cap_thread, NULL);

    printf("Encoding finished. Output: %s\n", out_h264);
    return 0;
}
