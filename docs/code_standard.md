# Code Standard
deepvac项目中的代码规范。使用的是C++举例，Java、Python也遵守。

## 代码理念
代码理念和Python之禅一致，如下所示：
```python
>>> import this
The Zen of DeepVAC, by Gemfield

漂亮胜于一切；
明了胜于晦涩；
简洁胜于复杂；
复杂胜于凌乱；
扁平胜于嵌套；
多行短代码胜于一行长代码；
可读性很重要；
不要以特殊理由逃避上述规则；
不要放过任何Error，除非有明确意图；
不容易实现的想法就不是好想法；
不容易解释的实现就不是好实现；
容易解释的代码才可能是好代码；
命名空间是个绝妙的设计；
```
需要特别指出的：
- 变量名即是注释;
- 善用容器和巧妙的算法来重构冗长的逻辑；
- 圈复杂度，超过3层嵌套就要保持警惕了；
- 模块化，注意设计模式；
- if分支尽量重构为table driven，写if前三思；
- Python的话要善用list conprehension;

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