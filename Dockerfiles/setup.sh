git clone -b rel-0.9.0 https://github.com/Theano/Theano.git
pip3 install -e Theano
git clone -b v0.6.5 https://github.com/Theano/libgpuarray.git
cd /libgpuarray
mkdir build 
cd /libgpuarray/build/ 
cmake .. -DCMAKE_BUILD_TYPE=Release 
make 
make install 
cd  /libgpuarray
python3 setup.py build
python3 setup.py install
python3 setup.py build_ext -L /usr/local/lib -I /usr/local/include
ldconfig
pip2 install opencv-python
pip2 install six
pip2 install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
cd / 
mkdir build 
cd build
# dependencies...
apt-get update
apt-get install -y libyaml-0-2 python-six
git clone git://github.com/lisa-lab/pylearn2
cd pylearn2
python2 setup.py develop --user
