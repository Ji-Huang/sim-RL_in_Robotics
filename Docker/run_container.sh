docker run  --rm -it \
            --name smlr_py38_new \
            --gpus all \
            -e DISPLAY \
            -v $(pwd):/home/smlr:rw \
            --privileged \
            --net="host" \
            rezaarrazi/smlr_py38_new:latest \
            bash