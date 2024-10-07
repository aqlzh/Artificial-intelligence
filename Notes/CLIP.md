## 一、CLIP 串烧





过去一年中，大家如何将CLIP模型和思想应用到其他领域中去。

![img](https://i0.hdslb.com/bfs/note/9088935963e04eddac34a9dda1f580c156b7d2f4.jpg@690w_!web-note.webp)



![img](https://i0.hdslb.com/bfs/note/83fb66ea6fb6cbfc8ae6f361644e5712878cade7.jpg@690w_!web-note.webp)



![img](https://i0.hdslb.com/bfs/note/a79f7e1aa1e092646bc1b6e7a380ba05caf014c3.jpg@690w_!web-note.webp)



CLIP： 给定图像文本对，分别通过对应编码器，对角线上元素是正样本，其他位置是负样本。结果是zero-shot能力非常强。推理的时候，对于一个图片，把所有可能候选都作为prompt丢进模型，计算相似度，最大相似度对应的文本标签就是类别。

### 1、Lseg





![img](https://i0.hdslb.com/bfs/note/2e59f44debf0cb71d1060c8655d59ebc3683f715.jpg@690w_!web-note.webp)

图片分类->像素级别分类。

![img](https://i0.hdslb.com/bfs/note/2c85013193c2abf8491b4262e7c358755c84f1a5.jpg@690w_!web-note.webp)

zero-shot 分割 零样本

![img](https://i0.hdslb.com/bfs/note/7c6fce0725f7e8ffc3cac847380f378d1685d8fe.jpg@690w_!web-note.webp)



dpt : vision transformer + decoder

文本编码器：用的CLIP的文本编码器，而且自始至终是锁住的。

文章意义是：将文本加入到传统图像任务中

![img](https://i0.hdslb.com/bfs/note/5cf6d66f38d96c8772714511a7cb3552ec7b5a3d.jpg@690w_!web-note.webp)



![img](https://i0.hdslb.com/bfs/note/1b6c279009f2287f1bf5609557d2734650851552.jpg@690w_!web-note.webp)

Lseg zero-shot还是比1-shot这种要差很多，有十几个点的提升空间。

- **LSeg可以直接应用于新的类别,而无需为这些类别重新训练模型,这是zero-shot模型的典型特征（为什么说LSeg是zero-shot？ perplexity回答）**
- One-shot，少样本，每个新类别至少有一个分类。

 

![img](https://i0.hdslb.com/bfs/note/f50a3345c999ec1116c555ed920701ae2c5282fd.jpg@690w_!web-note.webp)



failure cases: 

- CLIP本质上是选择相似性
- zero-shot做分割能提升的空间很大



### 2、group ViT

![img](https://i0.hdslb.com/bfs/note/87028fb13c45875766a7f72290b1ca1b433d7925.jpg@690w_!web-note.webp)

- LSeg虽然用了CLIP的预训练参数，但是不是无监督学习框架，也不是对比学习，没有对文本进行学习，还是依赖手工标注的segmentation mask。

- 如何摆脱掉手工标注，如何用文本来做监督信号，从而达到无监督训练？ GroupViT在这个方向上做出了贡献。

![img](https://i0.hdslb.com/bfs/note/8734010b5951b2d096d511abab54f96247ed5615.jpg@690w_!web-note.webp)

视觉grouping： **聚类中心点，从点开始发散，得到group，是一种自下而上的方式。**

groupViT贡献：已有的ViT框架中，加入grouping block，同时加入可学习的group tokens。

图像编码器输入：patch embedding + group tokens

group tokens : 64 * 384, 64是希望刚开始有比较多的聚类空间，384是为了和patch embedding同维。

经过6层transformer layer学习之后，加了一个grouping block，认为此时group token已经学的挺好的了，尝试来cluster一下，合并成为更大的group，学到更有语义的信息。此时加入grouping block，将patch embedding分配到group token上。此时整个ViT的输入长度从196 + 64 -> 64

![img](https://i0.hdslb.com/bfs/note/ed9e25d5b1d0591955bb07f4a0b8a9e7f51914cc.png@690w_!web-note.webp)

聚类分配的过程是不可导的，这里使用**gumbel softmax**将整个模型变成可导的，从而整个模型就可以做端到端的训练了。

到此为止完成了第一阶段的grouping。

由于一般segmentation中类别也不会很多，所以这里作者加了新的8个grouping tokens(8*384)，希望将64个再次映射到8个，本文作者在第9层transformer layer之后再做了一次groupign block。图像分成了8大块，每个块对应了一个特征。



challenge: 文本有一个特征，但是图像为序列长度为8的特征序列，如何将8打开特征融合成1块，变成整个图像的image level的特征，作者采用了average pooling。



总的来说，模型还是比较简单，所以scale性能可能比较好。



模型怎么做zero-shot推理？

![img](https://i0.hdslb.com/bfs/note/9245646a8579816da5d9422d976b01ca62ee37fa.png@690w_!web-note.webp)



局限性： 模型最后只有8个group embedding，图片分割最多只能检测8类。

问题： group token如何映射到原图片上？

![img](https://i0.hdslb.com/bfs/note/b89e2c07c71617bad5804a986550398bcf47d28a.jpg@690w_!web-note.webp)

上图对应groupin的效果



数值上的表现

![img](https://i0.hdslb.com/bfs/note/4cf87a3286871be4592fa78baba19707741bb858.png@690w_!web-note.webp)

无监督分割还是很难

两个limitations:

- ==现在groupViT更偏向图像编码器，没有很好使用dense特性==。
- 分割中的背景类，groupViT不光选择最大相似度，还设置了相似度的阈值。如果和所有类的相似度都达不到阈值，就认为是背景类。对于类别很多的情况，很容易出现前景物体置信度和背景物体置信度差不太多，如何设置阈值就很重要。分割做的很好，但是分类做的比较差。（这里将预测分割的部分和gt对上，所以只要分割做的好，分类就一定正确，发现这样性能提升了20多个点）。 这个问题的本质是因为CLIP的训练方式，只能学习到物体语义非常明确的东西，学不到非常模糊的（比如背景）的类。解决方案：可学习阈值？修改zero-shot推理方式？训练中增加约束将背景类融入其中。



### 3、ViLD



**知识蒸馏**是一种模型压缩技术，通常用于将知识从一个性能强大的预训练模型（**教师模型**）迁移到另一个较小且高效的模型（**学生模型**）中。在 ViLD 中，教师模型是像 CLIP 这样的视觉语言预训练模型，学生模型是一个目标检测器。知识蒸馏的目的是让学生模型获得与教师模型相似的表现，特别是在处理开放词汇检测任务时。



![img](https://i0.hdslb.com/bfs/note/bc1745f6d78ec200f1513054edd129f415cd7bc3.jpg@690w_!web-note.webp)

VilD

==看标题就知道是把clip当成一个teacher，从而去蒸馏模型，从而达到zero shot做目标检测==。



引言：

![img](https://i0.hdslb.com/bfs/note/c7f159478f0da00abf42e96749d48259b835ec06.png@690w_!web-note.webp)

是否可以在现有数据集上，不去做额外标注（黄鸭子），检测出新类别。

模型：

![img](https://i0.hdslb.com/bfs/note/dd76cc73b16449ceacd48099a5ef2360f5792451.jpg@690w_!web-note.webp)



a是有监督的baseline,bcd都是viLD方法。

baseline就是两阶段mask rcnn。

![img](https://i0.hdslb.com/bfs/note/dccec2278d7daeaeee5e696fef5234cd4eca873b.jpg@690w_!web-note.webp)

这里的文本类别还是base类，做有监督训练。

**ViLD-text只是把图像特征和文本特征连接到了一起。其他类别都塞给了背景类，专门有一个背景类embedding，在模型中学习。**

数学公式：

![img](https://i0.hdslb.com/bfs/note/fa392d85bfc6e388da5cf939687dd00eb55fc46d.jpg@690w_!web-note.webp)

图像I，phi(I)抽取图像特征，r是提前知道的proposal，经过额外的计算R，就得到region embedding e_r。

接下来定义了一个background embedding, e_{bg}，然后还有从文本中的t1,t2,..,t_{CB}, 分别点乘计算相似度，类似logits，然后和gt计算ce loss。

![img](https://i0.hdslb.com/bfs/note/34c84b4d674491e0496e9f9563eb6b13fdbff768.jpg@690w_!web-note.webp)

ViLD-image，想法： 希望region embedding尽可能和CLIP一致，最简单的方法是知识蒸馏。**粉色背景是teacher网络，左边是student网络。**使用L1 loss做蒸馏。因为现在的监督信号不再是人工标注，而是CLIP的编码，这里不再受基础类的限制，抽的proposal既可以是基础类的proposal，也可以是新类的proposal，都可以训练了，增强了做==open-vocabulary==的能力。 弊端：这里没有用上全部N个proposal，但是这里只用了M个，提前把每个图片，利用提前训练好的RPN(region proposal network)，预抽取M个proposal，这些proposal全部做crop & resize，过CLIP模型，把M个clip image embedding抽好，所以训练时候，只需要硬盘上load过来就行了。

![img](https://i0.hdslb.com/bfs/note/5552b29a9e4df78dfc69aa8b3cc8d05e9c3fb868.jpg@690w_!web-note.webp)

<font color=red>ViLD = VilD text + VilD image</font>

![img](https://i0.hdslb.com/bfs/note/209f5ad6cb489ebd7675f042403804c58dd88410.jpg@690w_!web-note.webp)

模型总览图



主要的表格

LVIS数据集：非常**长尾的目标检测数据集**，共1203个类，使用的还是COCO的图片，有一些类只标注了一次两次，所以这里有三个类，r(rare), c(common), f(frequent)。这样算AP的时候，就可以针对这几个分别算AP。

![img](https://i0.hdslb.com/bfs/note/911f84497dd2d637f68cd07142739c1490921f0e.jpg@690w_!web-note.webp)

common和frequent认为是基础类，rare是novel类。

但是这里LVIS数据集的特性导致有监督的模型也未必能够在rare上做的很好。





![img](https://i0.hdslb.com/bfs/note/19b549a59fecf4c74422545794913f625baf5b13.jpg@690w_!web-note.webp)

zero-shot方法，==直接拓展到其他数据集上，虽然比有监督上差，但在小规模数据集上，已经比较接近了，所以它 open-vocabulary上已经做的挺好了==。



### 4、GLIP



![img](https://i0.hdslb.com/bfs/note/ea598ae15026e842707fdf0d398d43f41dd7376f.jpg@690w_!web-note.webp)

研究动机：怎么利用更多数据（没有精心标注数据），==将图像文本对用上==。

vision grouding任务，给一句话，将这句话中的物体和当前图片中的物体找出来。

将detection和phrasing grounding合起来。



**Unified Formulation**

![img](https://i0.hdslb.com/bfs/note/e0ec70b9ea2048712a203eadf42e16496bd52c57.jpg@690w_!web-note.webp)

detection分类loss怎么算？vision grouding分类loss怎么算？如何合并到同一个框架下面。



01:03:1420240712





![img](https://i0.hdslb.com/bfs/note/e9082ce9b82848b2dd43cad540900809257667ea.png@690w_!web-note.webp)



detection的cls loss计算方式： bounding box，接一个分类头，然后得到分类logits，然后用nms把bounding box筛选一下，和gt计算cross entropy loss.

![img](https://i0.hdslb.com/bfs/note/1a76d2505aaf95bfe1b25040a5ab30bb94fde215.png@690w_!web-note.webp)



vision grounding cls loss计算方式：

匹配分数，图像中一样的处理，句子变成prompt，过一个文本的编码器，然后和图像embedding计算相似度（画图和VilDtext一致）



两种方式差不多，判断什么时候算positive match，什么时候算negative。



![img](https://i0.hdslb.com/bfs/note/fabd8625b3ea3a3eb40ecf48cccfd62ea9cfc682.png@690w_!web-note.webp)

Caption是self-training，用的伪标签peuodo label





![img](https://i0.hdslb.com/bfs/note/7d47a0b4875957f7fa710c7f472987c9951c8243.jpg@690w_!web-note.webp)

模型的总体框架

![img](https://i0.hdslb.com/bfs/note/2ecf0dd31576db3b47f130b95cd90bdf5fff7177.png@690w_!web-note.webp)

效果





![img](https://i0.hdslb.com/bfs/note/fbd2abe0c5a98957aa84644e462f4e5a090834cf.jpg@690w_!web-note.webp)

数值结果，zero-shot可以达到50AP, finetune一下，也可以达到60。

GLIP不论是zero-shot还是finetune，性能都非常强。

### 5、GLIP V2



![img](https://i0.hdslb.com/bfs/note/fde41d81dd001f7a5c487f6f5ce9f3e19097aa9e.jpg@690w_!web-note.webp)

GLIPv2: 分割检测、VQA、Visual grouding、Visual captioning都放进来了



![img](https://i0.hdslb.com/bfs/note/0550f45215f65ca53c20910163058088727736b9.jpg@690w_!web-note.webp)

思想和框架和GLIP差不多，但是加入了更多任务。

OFA， Unified-IO，提供统一框架囊括更多任务，争取把预训练任务学的又大又好。





### 6、CLIPasso



（CLIPasso: Semantically-Aware Object Skectching）  语义感知的对象偏移

**将CLIP做teach, 用它蒸馏自己的模型**

![img](https://i0.hdslb.com/bfs/note/f6ae4415e057030cd079937ae7ff142582a95fe0.png@690w_!web-note.webp)



- ​	semantic loss: <原始,生成>特征尽可能的接近
- ​	几何形状上的限制，==geomatric loss==: 
- perceptual loss把模型前面几层的输出特征算<原始，生成i>的相似性，而不是最后的2048维的特征（<font color=red>因为前面的特征含有长宽的概念，对几何位置更加的敏感</font>）。保证几何形状，物体朝向 位置的一致性
- **基于saliency的初始化方式：**用一个训练好的VIT，把最后一层的多头自注意力加权平均得到一个saliency map，对saliency map显著的地方进行采点。（**在显著的地方采点其实就相当于自己已经知道了这个地方有物体或已经沿着这个物体的边界画贝兹曲线了）效果更稳定**

![img](https://i0.hdslb.com/bfs/note/f76c00ac85d431a3361e51d6e03acf4d3e28202d.png@690w_!web-note.webp)



- 一张V100 6min 2000 iters
- 后处理：一张input，三张简笔画，取两个loss最低的那张

优点：

- zero-shot: 不受限于数据集里含有的类型
- 能达到任意程度的抽象，只需要控制笔画数

![img](https://i0.hdslb.com/bfs/note/33c9ff5605b146ebf52507ec56e619df0372edac.png@690w_!web-note.webp)

局限性：

- <font color=blue>有背景的时候，效果不好</font>（自注意力图等不好）-> automatic mask的方式如U2Net，将物体扣出里（但是是two step了，不是end to end）
- 简笔画都是同时生成的，不像人画的时候具有序列性（做成auto-regressive，根据前一个笔画去定位下一笔在哪 ） 及其没有时序
- 必须提前制定笔画数，手动+同等抽象度不同图像需要的笔画数不一样多，（将笔画数也进行优化）

CLIP+视频



### 7、CLIP4clip



>**Empirical study** 是一种基于**观察**和**实验**数据的研究方法，==通过直接收集和分析实际数据，来验证理论或提出新发现==。这种研究方法与纯理论研究相对，它依赖于现实世界的证据，并通过系统的观测和实验设计来探索、验证或反驳假设。





CLIP4clip: An empirical study of CLIP for end to  end video clip retrieval

![img](https://i0.hdslb.com/bfs/note/b3f37256c61561cc84a635fce7cd9a52e613fd53.png@690w_!web-note.webp)



视频是有时序的。一系列的帧，10个image token(cls token)如何做相似度计算:

1.<font color=red>parametr-free 直接取平均（目前最广泛接受的）。没有考虑时序，区分不了做下和站起来</font>

2.加入时序，LSTM或transformer+位置编码

late fusion:==已经抽取好图像和文本的特征了，只是在最后看怎么融合==

![img](https://i0.hdslb.com/bfs/note/f35237114f62f04b0e31adc43307768ee5e09b85.png@690w_!web-note.webp)

3.early fusion：最开始就融合

文本和位置编码, patch喂入一个transformer

![img](https://i0.hdslb.com/bfs/note/5e23f48b32c660b0dc311c44f7d9bd39d1e90dfe.png@690w_!web-note.webp)

直接拿CLIP做视频文本的retrieval,效果直接秒杀之前的那些方法

少量数据集：直接mean效果最好（CLIP在4million上训练的，微调反而不好）

![img](https://i0.hdslb.com/bfs/note/29437bad9ab5584a443ab0c558d4d120fdd2fd8a.png@690w_!web-note.webp)

 

![img](https://i0.hdslb.com/bfs/note/eb6997663872f5e5285385dfa29ab0f164f2e6cd.png@690w_!web-note.webp)

So, 大家都是直接mean

insights:

![img](https://i0.hdslb.com/bfs/note/0b01627a7e11646d172b4bbae6bbad03fb312d98.png@690w_!web-note.webp)

 Gradient search,多试几组学习率。



### 8、ActionCLIP



ActionCLIP: 动作识别

动机：

- 动作识别中标签的定义，标记是非常困难的。
- ==遇到新类，更细粒度的类==

![img](https://i0.hdslb.com/bfs/note/8c23be82e9c0eeae5acaf529db782d4700c51a60.png@690w_!web-note.webp)

==因为这里的文本就是标好的labels，非对角线点也可能是正样本。->交叉熵换成KL散度(两个分布的相似度)==

三阶段：pre-train, prompt, finetune

![img](https://i0.hdslb.com/bfs/note/be61ae79f0eab8ec3e3a03b9927f2d8f4cf4d278.png@690w_!web-note.webp)



![img](https://i0.hdslb.com/bfs/note/64729aef9c1435af7bae2373390b6a06129ea058.png@690w_!web-note.webp)



==**shift: 在特征图上做各种各样的移动，达到更强的建模能力。没有增加额外的参数和存储**。==

19年tsm将shift用到了时序

shift window，swin transformer里有用到

![img](https://i0.hdslb.com/bfs/note/f86b1c615e5118bfc0a0686d3d072202fd96cdbc.png@690w_!web-note.webp)

multimodal framework: 把one hot的标签变成language guided的目标函数

都是RGB+分类，使用CLIP预训练好的效果更好

![img](https://i0.hdslb.com/bfs/note/8083145d8f62590077ce13a633e06c859f3c4402.png@690w_!web-note.webp)

因为识别的数据集很大，funetune足够了

![img](https://i0.hdslb.com/bfs/note/c2520edcf2d62ce6f33b89a283a63bae2818fd97.png@690w_!web-note.webp)

zero/Few-shot的能力：

![img](https://i0.hdslb.com/bfs/note/85ebba1d600ac7f5f35e94d5bbfc4adc5361cef3.png@690w_!web-note.webp)

视频还有很多难点



55:21利于下游任务



拿CLIP作为visual encoder for diverse 下游vision-language tasks的初始化参数, 再finetune



56:06clip+语音



### 9、AudioCLIP

![img](https://i0.hdslb.com/bfs/note/e0cc25b945d5c73b1ef55309480c616158f6e8ea.png@690w_!web-note.webp)

文本，视频（帧），语音成triplet

**三个相似度矩阵，loss**

zero-shot语音分类



57:303D

### 10、PointCLIP



数据集很小

只要是RGB图像，CLIP都能处理的很好

![img](https://i0.hdslb.com/bfs/note/d48af73212a28714f4ad3a0b129059eef1a4dfe3.png@690w_!web-note.webp)

prompt: 明确告诉是点云





59:21

### 11、DepthCLIP



把深度估计看成了一个分类问题而不是回归



![img](https://i0.hdslb.com/bfs/note/dae5655cd2afdbe9ffd16a5629e97e3ed746e8c5.png@690w_!web-note.webp)

类别和[0.5,1,1.5..]对应



总结：

1.仅用CLIP提取更好的特征，点乘

2.clip做teacher，蒸馏

3.不用预训练的CLIP，仅用多模态对比学习的思想



### 12、拓展总结





回顾CLIP，用对比学习的方式学习一个视觉-语言的多模态模型。


1.对比学习预训练，文本和图片分别经过编码器得到特征。对角线上为n个正样本对，其他位置为n2-1负样本对。图片特征与文本特征建立了联系，此时模型从图片学到的特征可能不仅仅是图片本身的特征，还有文本的语义信息。openAI自建大规模的数据集WIT（webimage text）

2.zero-shot推理，prompt template。单词变成句子（预训练时是句子，避免distribution gap），再经过预训练好的文本编码器，得到文本特征。

3.测试图片经过预训练好的图片编码器，得到图片的特征。将图片特征与文本特征进行cos相似度计算，进行匹配。

与图片对应的文本可以看做高级标签，文本与图像建立了联系，文本引导模型从图片中提取文本的语义信息。 



#### ①Lseg



![image-20241005221929785](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005221929785.png)

第一行图中，能够完美的将狗和树分开，为了验证模型的容错能力，加一个汽车vehicle的标签，模型中也并没有出现汽车的轮廓。另一方面，==模型也能区分子类父类，标签中不再给出dog而是给出pet，dog的轮廓同样可以被分割开来==。

第三行图中，椅子、墙壁甚至地板和天花板这种极为相似的目标也被完美的分割开来。

![image-20241005222045802](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222045802.png)


如上图，与CLIP结构非常像，模型总揽图中**图像和文本**分别经过图像编码器（Image Encoder）和文本编码器（Text Encoder）==得到密集dense的图像文本特征==。此处密集的图像特征需进一步放大（up scaling）得到新的特征的图与原图大小一致，这一步也是为分割任务的实现。然后模型的输出与ground true的监督信号做一个交叉熵损失就可以训练起来了。Image Encoder的结构就是ViT+decoder，其中decoder的作用就是把一个bottleneck feature慢慢upscale上去。

![image-20241005222223938](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222223938.png)

这里的Loss不像CLIP使用对比学的loss，而是跟那些Ground True mask做的cross entropy loss，并非一个无监督训练。这篇论文的意义在于将文本的分支加入到传统的有监督分割的pipeline模型中。通过矩阵相乘将文本和图像结合起来了。训练时可以学到language aware（语言文本意识）的视觉特征。从而在最后推理的时候能使用文本的prompt任意的得到分割的效果。

![image-20241005222301886](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222301886.png)


本文中文本编码器的参数完全使用的CLIP的文本编码器的参数，因为分割任务的数据集都比较小（10-20万），==为保证文本编码器的泛化性，就直接使用并锁住CLIP中文本编码器的参数。图像编码器使用Vit / DEit的预训练权重，使用CLIP的预训练权重效果不太好==。


​                       ![image-20241005222310597](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222310597.png)                      

Spatial Regularization Blocks这个模块是简单的conv卷积或者DWconv，这一层进一步学习文本图像融合后的特征，理解文本与图像如何交互。后边的消融实验证明，两层Spatial Regularization Blocks效果最好，但是四层Spatial Regularization Blocks突然就崩了。其实Spatial Regularization Blocks这个模块对整个性能没有多大影响，可以先不去考虑。

![image-20241005222355195](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222355195.png)

PASCAL数据集上的结果，LSeg在zero-shot 上效果要好不少，但是对于1-shot来说还是差了15个点左右。如果使用大模型（ViT-L）也还是差了6个点左右。

![image-20241005222415291](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241005222415291.png)

<font color=red size=5>本质上再算图像特征和文本特征之间的相似性，并不是真的再做一个分类，就会把dog识别成toy玩具狗</font>。  

 

#### ② Group Vit



[Group ViT（Semantic Segmentation Emerges from Text Supervision）CV - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv18810713/?spm_id_from=333.999.0.0)



#### ③ ViLD



[ViLD（Open-Vocabulary Object Detection via Vision and Language Ko - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv18815536/?spm_id_from=333.976.0.0)





#### ④ GLIP V1/2

[GLIP_V1/V2（Ground Language-Image Pre-train）CVPR2022 - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv18815566/?spm_id_from=333.976.0.0)



#### ⑤CLIP Passo

[CLIP Passo：Semantically-Aware Object Sketching图像生成抽象的简笔画 - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv18854569/?spm_id_from=333.976.0.0)



#### ⑥CLIP4 clip



[CLIP4clip：An Empirical Study of CLIP for End to End Video Clip R - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv18854602/?spm_id_from=333.976.0.0)



