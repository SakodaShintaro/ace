#!/bin/bash

set -eux

TARGET_DIR=$1

ffmpeg -r 10 \
       -i ${TARGET_DIR}/frame_%05d.png \
       -vcodec libx264 \
       -pix_fmt yuv420p \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       -r 10 \
       ${TARGET_DIR}/../movie.mp4
