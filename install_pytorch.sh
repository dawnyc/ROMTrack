echo "****************** Installing pytorch ******************"
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
apt-get install libturbojpeg
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing onnx and onnxruntime-gpu ******************"
pip install onnx onnxruntime-gpu==1.6.0

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm

echo "****************** Installing yacs/einops/thop ******************"
pip install yacs
pip install einops
pip install thop

echo "****************** Installing vot-toolkit ******************"
pip install vot-toolkit==0.5.3
pip install vot-trax==3.0.3

echo "****************** Installation complete! ******************"
