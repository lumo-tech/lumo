- [CUB_200_2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)

num_class: 200
train: 5994
test: 5794

```
# https://www.kaggle.com/datasets/xiaojiu1414/cub-200-2011?resource=download
kaggle datasets download -d xiaojiu1414/cub-200-2011
```

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

num_class: 120
train: 

```
kaggle competitions download -c dog-breed-identification

# or
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt
```

 - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)


num_class: 16185
train: 8144
test: 8041


```
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
```

 - [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

```
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
```

- [FGVC-Aircraft Benchmark](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

```
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz
```



## FC-ImageNet

从 1000 个类别中构建均匀分布的层级数据集，最终选择 200 个类别，这两百个类别可以划分到 20 个粗粒度类别中，每个粗粒度类别包含 20 个类别。

ImageNet 的类别基于 WordNet 构建，因此可以利用 WordNet 提供的层级信息构建子类别。 

步骤：
 - 基于 wordnet 构建层级结构，最终形成的是一个树状结构
 - 对非叶子节点（不在 ImageNet 中的类），统计其包含的叶子节点的数量（即该粗粒度类别包含了多少 ImageNet 中的类）
 - 从根类开始，如果该非叶子节点包含了 K 个以上的叶子节点，那么对该节点进行一次分裂操作
 - 挑选 K，使得分裂结束时，有至少 20 个粗粒度类别包含至少 10 个以上的子类（没办法恰好选出来）
 - 对粗粒度类别按名称排序，选取前 20 个，对每个粗粒度类别下的细粒度类别（位于 ImageNet 中的类别）按类别名排序，选取前 10 个，最终得到平衡的层级Imagenet子数据集