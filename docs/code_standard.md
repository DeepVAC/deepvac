# Code Standard
DeepVAC项目中的代码规范。使用的是C++举例，Java、Python等其它语言也同样遵守。

## 代码理念
代码理念和Python之禅一致，如下所示：
```python
>>> import this
The Zen of DeepVAC, by Gemfield

1.漂亮胜于一切；
2.可读性很重要；
3.变量名即注释；
4.明了胜于晦涩；
5.简洁胜于复杂；
6.复杂胜于凌乱；
7.扁平胜于嵌套；
8.超过三层嵌套应坐立不安；
9.尽量复用代码；
10.应为没有模块化而寝食难安；
11.多行短代码胜于一行长代码；
12.不容易实现的想法就不是好想法；
13.不容易解释的实现就不是好实现；
14.容易解释的代码才可能是好代码；
15.善用容器和巧妙算法来重构冗长逻辑；
16.命名空间是个绝妙的设计；
17.条件分支尽量重构为表驱动；
18.写if前要三思而后行；
19.善用列表推导式；
20.不要以特殊理由逃避上述规则。
```
当在实践中遇到冲突的理念时，必须：
- 首先确保代码漂亮；
- 其次确保代码简洁；
- 其次确保代码扁平；
- 其次确保代码明了；
- 其次确保代码可读性；
- 其次确保短代码；
- 其次确保代码可解释；
- 其次确保代码巧妙；
- 以上无一确保时，确保离职手续已办妥。

## 代码规范

- file name (文件名)
```bash
syszux_msg.h
syszux.h
syszux_msg.cpp
```
- variable name (变量名)
```c++
int gemfield;
int gemfield_flag;
//global
int g_gemfield_flag;
//const
const int kGemfieldFlag;
```
- class name (类名)
```c++
class Syszux;
class SyszuxMsg;
```
- class data member name (类的数据成员)
```c++
int gemfield_;
int syszux_msg_;
```

- function name (函数名)
```c++
void get();
void getGemfieldConfig();
```
- namespace name (命名空间)
```c++
namespace gemfield;
namespace gemfield_log;
```

- ENUM & MACRO (枚举&宏)
```c++
//enum
enum GEMFIELD;
//macro
define GEMFIELD gemfield;
define GEM_FIELD 7030;
```

- flow control(流控制)
```c++
//if
if(true){
    int gemfield = 719;
}

//for
for(...){
    int gemfield = 719;
}

//switch
switch( level ){
    case gemfield::None:	
        resip::Log::setLevel( resip::Log::None );
        break;
    default:
        resip::Log::setLevel( resip::Log::None);
        break;
}
```

## python的import顺序
按照如下优先级顺序来import模块:
- import标准库
- import三方库
- import cv系列
- import torch系列
- import torchvision系列
- import deepvac系列
- import 本项目
