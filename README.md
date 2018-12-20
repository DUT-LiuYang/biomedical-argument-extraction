# biomedical-argument-extraction
生物医学事件要素(argument)识别

## 项目介绍
I will give a description of this project after the paper is published.

## 事件要素的定义
一个事件由一个触发词和若干个要素构成。

触发词的类型反映了事件的类型，通常为动词或者动名词。而事件的要素指的是事件的参与者，  
通常为句子中的实体或者另一个事件。

如下图所示，该示例中包含两个事件：  
第一个事件为Development事件（表示为E1），包含一个触发词“formation”和其对应的Theme类型的要素“capillary tubes”；  
第二个事件为Negative Regulation事件（表示为E2），包含一个触发词“inhibited”，一个Cause类型要素“Thalidomide”和一个Theme类型要素E1。  

两个事件的结构化表示为：  
Event E1 (Type: Development, Trigger: formation, Theme: capillary tubes);  
Event E2 (Type: Negative_regulation, Trigger: inhibited, Theme: Event E1, Cause: Thalidomide)。  
其中事件E2为嵌套事件，因为它的Theme要素同样为一个事件。

![avatar](resource/example.png)

一般来说，识别要素的时候，需要将事件的参与者与所在事件的触发词对应起来。  
本项目中，将对应后的触发词和事件参与者称作要素。

## 关于
```
未来数据研究所.LiuYang
```
