# jittor 安装

记录jittor的安装过程和遇到的问题。

## 安装

环境：

- ubuntu 22.04 LTS x86_64
- cuda 12.1
- cudnn 8.9.6
- gcc/g++ 10.5.0

使用Anaconda管理虚拟环境。

安装依赖:

```bash
sudo apt install libomp-dev
```

安装jittor:

```bash
pip install jittor
```

测试：
```bash
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
```

## 问题

### std::function问题

`test_example`报错：

```bash
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with '...':
    435 |           function(_Functor&& __f)
        |

/usr/include/c++/11/bits/std_function.h:435:145: note:      '_ArgTypes'
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with '...':
    530 |           function(_Functor&& __f)
        |

/usr/include/c++/11/bits/std_function.h:530:145: note:      '_ArgTypes'      
```

这个问题理论上出在jittor_utils尝试编译，但是引用c++11的std::function时出现了问题。原本的gcc/g++版本为11，但是这个版本的std::function有问题。将gcc/g++版本降级到10.5.0，问题解决。

以下是降级gcc/c++的参考方法:

1. 查看gcc10可选版本：

```bash
sudo apt-get update
sudo apt-cache policy gcc-10
```

2. 安装gcc10：

```bash
sudo apt-get install gcc-10=10.5.0-1ubuntu1~22.04
```

3. 修改默认gcc版本：

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50
sudo update-alternatives --config gcc
```

选择gcc10对应的序号并回车。

4. 修改g++版本同理

注意保持gcc和g++版本一致，否则会出现一些难以预料的问题。

### 链接libstdc++.so.6问题

`test_example`报错：

```bash
ImportError: /home/username/anaconda3/envs/DragGAN-jt/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /home/username/.cache/jittor/jt1.3.8/g++10.5.0/py3.7.16/Linux-6.5.0-27x81/12thGenIntelRCxbc/default/cu11.5.119_sm_86/jittor_core.cpython-37m-x86_64-linux-gnu.so)
```

其实操作系统中的`libstdc++.so.6`版本是够的，但是Anaconda中的环境中的`libstdc++.so.6`版本不够。这里的解决方案是从操作系统中拷贝`libstdc++.so.6`到Anaconda环境中，当然也可以建立一个软链接。

1. 查找所有`libstdc++.so.6`：

```bash
find / -name "libstdc++.so.6*"
```

至少可以观察到两个`libstdc++.so.6.0.*`路径，一个是系统的路径，一个是Anaconda的路径。

```bash
/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30
/home/username/anaconda3/envs/DragGAN-jt/lib/libstdc++.so.6.0.29
```

2. 拷贝`libstdc++.so.6.0.30`到Anaconda环境中并建立软链接：

```bash
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/username/anaconda3/envs/DragGAN-jt/lib/
ln -s /home/username/anaconda3/envs/DragGAN-jt/lib/libstdc++.so.6.0.30 /home/username/anaconda3/envs/DragGAN-jt/lib/libstdc++.so.6
```

注意sudo权限。如果操作系统中的`libstdc++.so.6`本身版本就不够，那么需要升级gcc/g++版本。

### test_cudnn_op问题

注意cudnn是必须的，并且和cuda版本对应。

查看cudnn版本：

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
