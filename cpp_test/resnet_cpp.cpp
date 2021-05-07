#include <torch/torch.h>

//3*3卷积
torch::Tensor conv3x3(int in_planes,int out_planes,int stride=1):
    //3x3 convolution with padding
    //in_planes@输入通道数，输入深度
    //out_planes@输出通道数，输出深度
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

struct Bottleneck : torch::nn::Module{
    //难点1：类中的属相设置
    //难点1-1：非常规属性怎么设置

    //属性定义
    //这样写？
  Bottleneck()
      : conv1 = register_module("conv1", torch::nn::Conv2d(inplanes, planes, kernel_size=1, bias=False));
        bn1 = register_module("bn1",nn.BatchNorm2d(planes))

        conv2 = register_module("conv2", torch::nn::Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False));
        bn2 = register_module("bn2",nn.BatchNorm2d(planes))

        conv3 = register_module("conv3", torch::nn::Conv2d(planes, planes * 4, kernel_size=1, bias=False));
        bn3 = register_module("bn3",nn.BatchNorm2d(planes * 4))

        relu = register_module('relu',nn.ReLU(inplace=True))

        bool downsample;
        int stride;
 {
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("bn1", bn1);
   register_module("bn2", bn2);
   register_module("bn3", bn3);
 //forward()方法
  torch::Tensor forward(torch::Tensor x) {
        identity = x

        out = conv1(x)
        out = bn1(out)
        out = relu(out)

        out = conv2(out)
        out = bn2(out)
        out = relu(out)

        out = conv3(out)
        out = bn3(out)

        if(! downsample){
          identity = downsample(x)
        }

        out += identity
        out = relu(out)

        return out


 }

    //还是这样写？
    Net() {
    // Construct and register two Linear submodules.
        conv1 = register_module("conv1", torch::nn::Conv2d(inplanes, planes, kernel_size=1, bias=False));
        bn1 = register_module("bn1",nn.BatchNorm2d(planes))

        conv2 = register_module("conv2", torch::nn::Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False));
        bn2 = register_module("bn2",nn.BatchNorm2d(planes))

        conv3 = register_module("conv3", torch::nn::Conv2d(planes, planes * 4, kernel_size=1, bias=False));
        bn2 = register_module("bn3",nn.BatchNorm2d(planes * 4))

        relu = register_module('relu',nn.ReLU(inplace=True))

        bool downsample;
        int stride;

        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }
}