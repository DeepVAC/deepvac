# Code Standard
deepvac项目中的代码规范。使用的是C++举例，Java、Python也遵守。

## 通用代码规范
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

- 善用容器和巧妙的算法来重构冗长的逻辑；
- 圈复杂度，超过3层嵌套就要保持警惕了；
- 模块化，注意设计模式；
- if分支尽量重构为table driven，写if前三思；
```

## Python代码规范
- 善用list conprehension；