if [ ! -d "train2014" ]; then
    echo "Downloading COCO image dataset"
    curl http://msvocds.blob.core.windows.net/coco2014/train2014.zip > train2014.zip
    unzip train2014.zip
else
    echo "Already found COCO image dataset"
fi
