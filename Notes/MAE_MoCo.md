## 一、MAE  阅读



### 1、与之前论文联系



**和之前精读论文的关系？**



Transformer

- 一个纯基于注意力机制的编码器和解码器
- 表现比 RNN 架构好，在机器翻译任务



BERT

- 使用 一个 Transformer 编码器
- 拓展到更一般的 <font color=red>NLP</font> 的任务
- 使用了 完型填空 的==自监督==的训练机制
- 不需要使用标号
- 去预测一个句子里面 不见 masked 的词 
- 从而获取对文本特征抽取的能力
- BERT 极大的扩展了 Transformer 的应用
- 在一个大规模的、没有标号的数据上 训练出非常好的模型出来



ViT 

- 将 Transformer 用到 <font color=red>CV</font> 上面
- 把整个图片分割成很多 16 * 16 的小方块
- 每一个方块 patch 做成一个词 token，然后放进 Transformer 进行训练
- 证明：训练数据足够大 （1,000万 或者 一个亿的训练样本）的时候，Transformer 的架构 精度 优于 CNN 架构





![img](https://i0.hdslb.com/bfs/note/11dcb4bc7dddbb255c9c10bb96b4c0598e90f489.png@620w_!web-note.webp)



MAE：

- BERT 的一个 CV 的版本
- 基于 ViT ，BERT化
- 把整个训练 拓展到没有标号的数据上面
- 通过完型填空来获取图片的一个理解
- 不是第一个将 BERT 拓展到 CV 上
- MAE 很有可能 未来影响最大
- ==BERT 加速了 Transformer 架构 在 NLP 的应用==
- ==MAE 加速 Transformer 在 CV 上的应用==



### 2、标题 + 作者



02:46



Masked Autoencoders **带掩码的自编码器** 是可扩展的视觉学习器 scalable vision learners



**scalable**：可扩展的，模型比较大

**efficient**：算法特别快



**vision learners**：≠ classifier，一个 backbone 的模型



**masked**：来源于 BERT

- BERT 是 masked language model，带掩码的语言模型；完形填空
- 每次挖掉一些东西，然后去预测挖掉的东西



**Auto-encoder**：

- auto “自”，ML模型 auto 自模型；i.e., 自回归模型
- 标号 y 和 样本 x 来自 同一个东西
- 语言模型中，每一次用前面那些词，去预测接下来的一个词；在另外一个样本里面，我们这个预测的词也是标号，也是成为另外一个样本的 x 本身
- 样本 x 和 标号 y 来自于同样的句子里面的词 --> auto 
- NLP 里，语言模型是一大类，有没有 auto 大家都理解
- Transformer、BERT 用的是 encoder，加不加 auto 大家都觉得没问题
- CV 里 auto 自任务很少
- 图片（像素）的标号很少来自于图片本身，图片的标号更多是文本
- encoder-decoder 在图片里用得很多
- ==加 auto 在 encoder之前，MAE 的图片标号是图片本身，区分于其它工作==



**标题模板很不错！**

强有力的句式：文章的结论总结成一句话，XX 是 好 XX

结论放在 title 里，句式相对客观

i.e., GPT 系列 3篇文章

GPT1: Improving Language Understanding by Generative Pre-Training (Generative Pre-Train Model 就是GPT模型的名字由来）

GPT2: Language Models are Unsupervised Multitask Learners

GPT3: Language Models are Few-Shot Learners



**Example**：

我的手写笔是世界上 最好用来做笔记的笔 ❌

手写笔是一个做笔记很好的工具 ✔

- 虽然手写笔是我第一个发明的
- 从读者角度讲述，手写笔为什么它是一个好东西？



**作者**



06:03



FAIR

一作：Kaiming 恺明 ResNet 一作 + Project lead （从公司拿资源做项目、管理者；不是一般最后的大老板）

*：equal technial contribution

最后两位作者：CV 物体检测的大佬



### 3、摘要



07:10



**第一句话扩展 title MAE**

masked autoencoders are scalable vision learners --> Masked AutoEncoders are scalable self-supervised learners for computer vision. 



**第二句话 MAE 简单**

随机盖住图片里的一些块(patch, image 的一个块)，==再重构 缺失的像素==。

- 思想来自于 BERT 带掩码的语言模型
- 这里不是 masked 词，而是图片的一个块 image，重构这个缺失块里面的所有像素



**第三句话 2个 core designs**

**asymmetric encoder-decoder architecture**

- 虽然 MAE 说自己是 masked autoencoder
- 任何一个模型都有一个编码器解码器
- BERT 预测相对简单， 解码器：最后的一个全连接输出层
- MAE 预测一个块里面的所有像素
- MAE 的编码器 只关注 可见的 patches，节省计算时间
- 如果 一个 patch 被丢掉了，编码器不会对它进行编码



- MAE a lightweight decoder 重构原始的像素



**遮住大量的块 (i.e., 75%) 重构像素是一个非显然 nontrivial and meaningful 有意义的 self-supervisory task 自监督任务**

- ==如果只遮住几块的话，插值法就可以得到缺失的像素==，trivial 
- 遮住一大半的块，迫使模型学习更好的表征



**第四句话：asymmetric encoder-decoder + 75% masked --> train large models efficiently and effectively**

更有效的训练模型

- 大：比较有挑战的任务，不是求显然的解 nontrivial
- 块：不看遮住的块，训练量是 1 / 4，加速 3 倍 or 以上



**第五句话：结果好**

最简单的 a vanilla ViT-Huge 模型 在 ImageNet-1K 87.8% 准确率

- ViT 的自监督学习，效果不那么好，所以没怎么展开
- ViT 的作者认为还是需要 有标号的模型、用更大的训练集，效果更好
- MAE 使用小数据集  ImageNet-1K 100w 图片，self-supervise 效果很好



**第六句话：迁移学习效果**

MAE 主要用来做 迁移学习，在别的任务上表现好

shows promising scaling behavior





### 4、关键图



10:49



CV 最重要的一张图就是放在第一页的右上角



图1：MAE 的模型架构

**预训练流程**：input --> patches --> masked --> unmasked patches in encoder --> unmasked + masked 按位置排列 进 decoder --> decoder 重构 masked patches 的像素

- **patches + masked**：一张红色鸟图片进来，切成 patches，masked 块 (3/4) 是 灰色的。
- **unmasked patches，encoder**：没有 masked (1 / 4) 的块 进入 encoder (ViT)，得到每一块的特征（蓝色）。
- encoder 的输出 和 masked tokens 按照在图片中的原始位置排列成一长条向量 （包含位置信息）。
- 长条向量 进入 decoder，解码器尝试重构缺失的像素信息，还原原始图片





![img](https://i0.hdslb.com/bfs/note/7294f04595235a39b9b66a62ec67fb40510578e3.png@620w_!web-note.webp)



encoder 比 decoder 高：==计算量主要来自于 encoder，对图片的像素进行编码==



优化 encoder by 编码器只用处理 unmasked patches，i.e., 一张图里 1/4 的像素，--> 计算量降低

- Transformer 模型计算量特别大，几倍加速也很重要。





**Q：什么情况不需要解码器？**

用 MAE 做一个 CV 的任务，只需要用编码器。一张图片进来，不需要做掩码，直接切成 patches 格子块，然后得到所有 patches 的特征表示，当成是这张图片的特征表达，用来做 CV 的任务。



**图2：ImageNet 测试集图片**



三列分别是：80% masked tokens, MAE 重构的效果，ImageNet 真实图片



![img](https://i0.hdslb.com/bfs/note/506bd283de3d0601db9b2016dd9899f7193aa3a4.png@620w_!web-note.webp)



虽然细节有一点模糊，钟的指针、车的形状、狗、灯都还原的很好。

- 图片尺寸只有那么高，分辨率有限



Note: MAE 不一定对 所有图片的构造都那么好，图 2 是展示比较好的样例



**图3：COCO**

不同的数据集，效果也惊人。



![img](https://i0.hdslb.com/bfs/note/0a890bb24a5ebf5d7db4362aac5f18253927c360.png@620w_!web-note.webp)



图4 同一张图片、masked patches 的不同比例 的还原效果

95%效果惊人，蘑菇🍄、斑马🦓、车🚗、西红柿 都还原出来了。

![img](https://i0.hdslb.com/bfs/note/b1370ee2952a30ece18687fc49e5d7b9e41b2cc5.png@620w_!web-note.webp)





### 5、结论



15:19



3段



**Simple algorithms that scale well are the core of deep learning**. ==简单 + 可以拓展很好的算法是 DL 的核心==



**simple**：作者的简单是在 ViT 基础上，MAE 提出来的东西相对简单



**scale well**：能跑 大数据集



作者对 simple 和 scale well 是不是有什么误解？哈哈哈哈

- MAE 基于 ViT，ViT 基于 Transformer，整个 Transformer 模块里面有那么多东西，要那么多层堆起来 --> 比较复杂
- 很有钱的时候，scale well 无限加数据（无需标号）



In NLP, self-supervised learning methods enable benefits from exponentially scaling methods. NLP 自监督学习 火 🔥



CV 里 有标号的预训练数据是主流。



MAE 在 ImageNet 数据集上，通过自编码器学习到 可以媲美 有标号的 结果。



**第二段：图片和语言的差别**

- a word in a sentence：一个词是语义单元，包含较多语义信息
- a patch in an image：一定的语义信息，但不是一个语义的 segment
- 一个 patch 并不含有一个特定的物体
- 可能是多个物体的一小块 or 一个物体重叠的一块
- 即使图片和语言的 masked 的单元包含语义信息不同，MAE or Transformer 可以学到一个隐藏的比较好的语义表达



**第三段：broader impacts**

如果工作出圈，对社会的影响？

- 只用了图片本身信息学习
- 图片本身有 bias 的话，倾向于某一些图片 or  有一些不好的图片，可能会有负面的社会影响
- MAE 可以用来生成不存在的内容
- MAE 是生成模型，生成原始的像素
- 和 GAN 类似，有误导大家的可能
- 如果要使用这个工作，请一定要考虑潜在的影响



So far, MAE 在干什么？ 效果怎么样？



### 6、导言



18:33



**导言第一段：问题所在**

深度学习飞速发展，但 CV 仍然以来百万级别、甚至更多有标注的数据



**导言第二段：大量有标注数据是必须的吗？其它领域怎么解决的？**

NLP 的自监督学习很好

- ==GPT、BERT 可以使用 无标注 的数据及逆行训练，得到千亿级别可学习的参数模型==
- GPT 系列，一个标准的语言模型
- BERT 一个带掩码的自编码模型

CV 里已有的 maksed autoencoder 带掩码的自编码器

- denoising autoencoder，一张图片里加入很多噪音，通过去噪来学习对这张图片的理解
- 最近也有很多工作将 BERT 应用于 CV



但，作者认为 BERT 在 CV 领域的应用落后于 NLP

**What makes masked autoencoding different between vision and language？**

什么使得 带掩码的自编码器模型在 CV 和 NLP 处理上的不一样呢？



**1）==CV 使用 CNN，卷积窗口不好将 mask 放进去==**

**archtectural gap has been addressed by ViT**

- CNN 在一张图片上，使用一个卷积窗口、不断地平滑，来汇聚一些像素上面的信息 + 模式识别
- Transformer 的一个 mask 对应的是一个特定的词，会一直保留，和别的词区分开来
- **卷积上做掩码？**
- 图片的一块盖住 by 像素替换成一个特定的值，
- 卷积窗口扫过来、扫过去时，无法区分边界，无法保持 mask 的特殊性，无法拎出来 mask；最后从掩码信息很难还原出来
- **卷积不好加入位置编码？** 不那么充分
- Transformer 需要位置编码：attention 机制没有位置信息
- 卷积自带位置信息，不断平移时，不需要加入位置信息



**2）语言和图片的信息密度不同**

NLP 的一个词是一个语义的实体，一个词在字典里有很长的解释；一句话去掉几个词，任务很难，i.e., 完形填空 --> BERT 的 mask 比例不能过高



CV 的图片去掉一个块，通过对邻居的像素值进行插值还原。**怎么让任务不那么 trivial 呢？**

- **随机去掉很高比例的块**，极大降低图片的冗余性
- 这一块的周边块都被去掉了，这一块和很远的块的关系不那么冗余
- nontrivial 任务，使 模型去看 一张图片的 holistic 全局信息，而不仅关注局部





3）The autoencoder‘s decoder

CV 还原图片的原始像素：低层次的表示

NLP 还原句子里的词：语义层次更高，i.e., BERT 的一个全连接层还原词



图片分类、目标检测的 decoder：一个全连接层

语义分割（像素级别的输出）：一个全连接层不够，很有可能使用一个转置的卷积神经网络、来做一个比较大解码器。



**MAE 的想法：**

随机遮	住的像素信息，让它使用一个非对称的编码器和解码器的机制。



<font color=red>**非对称**：编码器和解码器看到的东西不一样</font>

- 编码器只看到可见块
- 解码器拿到编码器的输出之后，重构 masked patches
- 非对称的原因：
- 大量 masked 块
- 编码器只看可见块，极大降低计算开销、减少内存消耗



**导言最后一段：实验结果**

MAE预训练，只使用 ImageNet-1K 100w 无标号数据，ViT-Large/-Huge 达到 ViT 需要 100倍于 ImageNet-1K 的数据 的效果。



迁移学习效果也很好，预训练的模型在 目标检测、实例分割、语义分割 的效果都很好。



和 NLP 类似的效果：

在大量的没有标号的数据上，通过自监督学习训练出来模型，迁移学习效果不错



2页，图片 + 使用了 问题 - 回答问题 - 引出想法 的写法



**更本质的问题？**

把 BERT 从 NLP 用到 CV 有什么问题？

MAE 算法为什么要设计成这样？

- ViT 解决 图片中的 mask
- 大量随机 masked 块，降低图片冗余度
- 非对称的自编码器-解码器



**写作建议：**

<font color=blue>**讲清楚，你为什么要这么做？你对这个问题的动机？**</font>

- 没有动机 就是技术报告了，i.e. AlexNet





### 7、相关工作



26:58



**1）带掩码的语言模型**：BERT, GPT

**2）自编码器在 CV 的应用**

- MAE 也是一种形式的 带去噪的自编码
- masked patch 在这一个图片块里面加了很多噪声
- 和 传统的 DAE(Denoising autoencoder) 是很不一样的
- MAE 基于 ViT、transformer 架构

**3）带掩码的自编码器在 CV 的应用**

- iGPT，GPT 在 image 的应用
- ViT 最后一段，怎么用 BERT 方式训练模型
- BEiT，BERT 在 image 上的应用
- 给每一个 patch 离散的标号，更像 BERT 

MAE 直接重构 原始的像素信息



**4）self-supervised learning**

最近火 🔥 的 contrastive learning，==使用数据增强==

MAE 不需要数据增强（实验部分有讲）



相关工作总结：4个角度的对比，MAE 和它们都不同，但是**没有特别讲 MAE 和每一个不一样的地方在哪里**。

esp, iGPT 和 BEiT



**写作建议：**

相关工作介绍，写清楚和特别相关的工作的区别，不要让大家去猜





### 8、MAE模型



28:49



MAE 是一个简单的 自编码器（无监督）

- 看到了部分的观察的数据
- 用 观察到的部分数据 **重构** 完整的原始信号



所有自编码器的工作

- 将观察到的信号 映射到一个潜在 latent 表示里面
- 潜在表示：语义空间的一个表示
- 解码器 用 潜表示 latent 重构原始信号



**MAE 的自编码器 和 经典自编码器的不同？**

- asymmetric design 非对称结构
- 编码器只看 可见块
- 忽略不可见 masked 块，节省计算开销



**掩码 mask 如何工作？**



29:39





<font color=red>和 ViT 的一样图片 patches 化， i.e., 一张图片 九宫格分割，3 * 3，每一格 代表一个 patch，作为一个词 token</font>



![img](https://i0.hdslb.com/bfs/note/29d408076e5b05e55be18a35f38e3cb7ca360347.png@620w_!web-note.webp)



**random sampling?**

随机均匀采样块保留, 剩余块用 mask 掩码盖住。



**MAE 的关键技术？**

只采样少量的块，其余的块全覆盖，去掉图片 patches 之间的冗余度。--> 增加任务的复杂度



**MAE 的编码器**

- a ViT， 没有任何改动
- but applied only on visible, unmasked patches, 只作用于 可见块



**MAE 编码器如何作用于可见块呢？**



30:34



和 ViT 一样：

- 每一个 patch 块拿出来，做一个线性投影
- \+ 位置信息 --> token 



和 ViT 不一样：

- masked 块不进入 MAE 编码器
- i.e., 随机采样概率 1 / 4， 25% 样本块进入 ViT，计算量减少



**MAE 的解码器**

要重构 masked 块的像素信息，需要看到 可见块（编码器对可见块的潜表示） 和 masked 块 （没有进入编码器）



a shared, learned vector 通过**一个** 共享的可以学到的向量来表示 each mask token 



==**每一个被盖住的块都表示成同样一个向量，**此向量值可学习==



解码器是 another series of Transformer blocks 另外一个 transformer

- 需要位置信息，不然无法区分对应哪一个 掩码masked tokens 



**可见块的位置信息 question？**

位置信息 要不要也对那些编码器过来的 潜在表示 也加上



==因为可见块的潜表示其实本来已经加过一次了，那么这个地方要不要再加上一次==？



**解码器什么时候用？**

pre-training；别的下游任务，解码器不需要，只需要编码器对图片编码得到潜表示 --》灵活，想用什么随便用



**解码器的架构 大 吗？**

相对小，计算开销不到编码器的 1 / 10





**怎么样重构出原始的像素？**



32:23



解码器的最后一层： a linear projection 

- 一个 patch 是 16 * 16 像素的话，线性层会投影到长为 256 的维度
- 再 reshape(16, 16), 还原原始像素信息
- ==损失函数： MSE，像素值相减，再平方和==
- 只作用于非可见块的损失，和 BERT 一样
- 可见块的图片编码器已经看到了，看到答案就不算正确率了





对预测的像素做一次 normalization，使像素均值为 0 方差为 1，数值更稳定。



**pre-training 的 normalization 可，但预测的时候怎么办呢？**

- pre-training 有标号的样本均值方差可算



**Simple implementation** 



33:31



- 对每一个输入 patch 生成 a token：一个一个 patch 的线性投影 + 位置信息
- 随机采样：==randomly shuffle== 随机打断序列，把最后一块拿掉。
- 从 头部均匀的、没有重置的 样本 采样
- 25% 意味着 随机 shuffle， 只保留前 25% 
- after encoding 解码时：append 跟以前长度一样的这些掩码的一些词源 mask tokens （一个可以学习的向量 + 位置信息），重新 unshuffle 还原到原来的顺序
- MSE 算误差时，跟原始图的 patches 对应



The decoder is applied to this full list (with positional embeddings added). 

**编码器处理可见块的潜表示需不需要再加位置信息？**



**shuffle 和 unshuffle 有什么好处？**

没有稀疏的操作，实现快，不影响 ViT 块的实现





总结：解码器怎么做？怎么还原像素信息？实现中随机采样怎么做？





### 9、实验



35:22





第四章：ImageNet 的实验



在 ImageNet-1K 100万张图片 数据集上

- 先做自监督的预训练（不用标号，只拿图片）
- 然后再在同样的数据集上做有标号的监督训练



**做法：**

- end to end 的微调，允许改整个模型 所有的可学习的参数；
- linear probing 允许改最后一层的线性输出层



结果：在验证集上报告 top-1 的精度，用 中心剪裁的 224*224 的图片



Baseline: ViT-Large / 16, 16 * 16 

ViT-Large 比 ResNet50 要大很多，很容易 overfitting 





**比较的 3 种情况：**

**scratch, original: 76.5,** ViT 所有的内容在 ImageNet-1K上训练, 结果不稳定 200 epoches



**scratch, our impl.: 82.5** 加入 strong regularization A.2

- ViT 文章说 ViT 需要很大的数据才行
- 小一点的数据 + 合适的正则化 ✔



**baseline MAE: 84.9** 先使用 MAE 做预训练，再在 ImageNet 上微调 50 epoches

- 数据集没变化，预训练和微调都是 ImageNet 

MAE 纯从图片上学到不错的信息



**主要结果 表1**



37:36



**ablation study**

**a: 解码器的深度**，多少个 Transformer 块; end to end fine-tuning 贵一点，但效果好

- 全都 ft，深度和效果关系不大 84.x
- 只调 lin, 深度深一点好



**b: 解码器的宽度**，每一个 token 表示成一个多长的向量

- 512 比较好



**c: 编码器要不要加入被盖住的 masked 块：**

- **不加**很好，精度高、计算量更少
- 非对称的架构 精度好、性能好



![img](https://i0.hdslb.com/bfs/note/7ecd309199ce7297bf0796d2cc3915ad4a3cbdd8.png@620w_!web-note.webp)



**d: 重构的目标**

- 每个像素的MSE
- **每个像素的MSE + normalization 均值为0 方差为 1 效果好**
- PCA 做一次降维
- dVAE: BEiT 的做法，通过 ViT 把每一个块映射到一个离散的 token，像 BERT 一样的去做预测





**e 怎么样做数据增强**

- 什么都不做
- 固定大小的裁剪
- **随机大小的裁剪**
- 裁剪 + 颜色变化

MAE 对数据增强不敏感



**f 怎么采样 被盖住的块**

- <font color=red>随机采样 最简单最好</font>
- 按一块块的采样 50 %
- 按一块块的采样 75 %
- 网格采样



**表1 主结果内容的展开图**



40:15





使用不同的掩码率的时候的效果：



10%的块被遮住： 83.2%

\> 40%的块被遮住：精度大幅提升

只调最后一层：更敏感



![img](https://i0.hdslb.com/bfs/note/0331e5654b0b4ca48e84d65cd98310fc1fcb0bbf.png@620w_!web-note.webp)





**表2 训练时间**

- ViT-Large + 解码器只使用一层 Transformer 的块：84.8% 精度不错，耗时最少
- 带掩码的块 + 大的解码器，加速 3.7倍
- ViT huge 加速也比较多



![img](https://i0.hdslb.com/bfs/note/d4c3e7e55fb932deda8df1fe19f8c31422fccc60.png@620w_!web-note.webp)



**绝对时间**

128个 TPU v3的 core， tensorflow 实现

训练时间是10个小时 和 大概是一天多，可以忍受





**图 6 表示的是不同的掩码采样策略的区别**

- 随机采样效果好
- 尽量的按照一块一块的来切
- 按照格点来切



![img](https://i0.hdslb.com/bfs/note/10cc8644d42c4897afa66ad8def2f342ba52b7a6.png@620w_!web-note.webp)





**图 7 预训练的轮数和微调的精度的对比**



![img](https://i0.hdslb.com/bfs/note/ca4bc214ce8cb2bec134ac1e09a9d33d8ac91f6d.png@620w_!web-note.webp)



 ImageNet-1K 上训练个 1,000 个数据轮，精度有提升，在一直训练一直学习，过拟合也没那么多严重，因为1,000轮是非常非常多的

- 一般在 ImageNet 上训练， 200轮 enough





**4.1 讲的是不同的超参数下的一些结果**

- 自己跟自己比



**4.2 就是比的是之前结果**

- 主要的结果在表3和图8里面





![img](https://i0.hdslb.com/bfs/note/5d30295cdab2f14896ff98d8981e3098123cf54b.png@620w_!web-note.webp)



表3：==跟前面结果比 MAE 效果是最好的==

图8：跟 ViT 里面的结果比

- 最上面的虚线：<font color=red>ViT 在 JFT 3亿标号的图片数据集合的效果</font>
- 排第二的线：只使用 ImageNet-1K 也就是1/300数据的效果
- 两根线很接近，不能说这是一个很公平的比较
- JFT数据集包括的类数远远大于 ImageNet
- 它们很多是一些 顾我自己 care 的一些目标，但是 ImageNet很多都是一些猫猫狗狗的图片
- 测试集也是 ImageNet，JFK 它多了很多很多 可能跟你那些标号不那么一样的图片



把验证集换到一个 不那么跟 ImageNet 相像的数据集上，可能这个差距会大一点



**主要比较的是算法，而不是数据集**





**4.3 调编码器所有层的参数和最后一层的参数效果差距大**

**到底调多少层：**

- 少，快，精度差
- 多，慢，精度好

![img](https://i0.hdslb.com/bfs/note/2907e839937588ab073242c7acfff98ca94c6106.png@620w_!web-note.webp)



调 4 - 5 层比较好

- 底层不调：底层学到的东西稍微是比较低层次一点，你可以换一个任务也不需要变太多
- 上面那些层，跟你的任务相关，要调



**5迁移学习实验结果**



45:08





COCO 的目标检测和分割

- MAE 做它的主干网络 效果最好
- 跟之前的 ViT 和 BEiT 比



重构像素的比较

- 像 BEiT 那样重构 or 用 dVAE 学出来的标号比 差别不大
- 重构原始的像素 简单、好





![img](https://i0.hdslb.com/bfs/note/8eeef4086553d00b340bfeabfdd6b893174356aa.png@620w_!web-note.webp)





### 10、评论



45:45



MAE 算法不难

- 利用 ViT 来做跟 BERT 一样的自监督学习
- ViT 文章已经做了这个事情了



**MAE 在 ViT 的基础提升点**

- 需要盖住更多的块，降低剩余块之间的冗余度，任务变得复杂一些
- 使用一个 Tranformer 架构的解码器，直接还原原始的像素信息，使得整个流程更加简单一点
- 加上在 ViT 工作之后的各种技术，训练更鲁棒



以上三点 MAE 使得在 ImageNet-1K 这个数据集上使用自监督训练的效果超过了之前的工作



**写作：简单，故事性好**



导言：为什么 MAE 这么做

非常详细的实验数据：整个 MAE 里面 所有可以调的超参数 它到底是什么意思？





**简单的想法、非常好的结果、详细的实验 ==》**

**一个非常高质量的工作**







**从一篇比较新的文章开始， 怎么样得到你自己的一些研究思路呢？**



vit 在小数据集上要多加正则化约束







我自己对其的总结





这篇论文的标题是《Masked Autoencoders Are Scalable Vision Learners》，由 Facebook AI Research (FAIR) 的团队撰写。论文主要探讨了在计算机视觉领域中，使用掩蔽自编码器（Masked Autoencoders, MAE）作为可扩展的自监督学习者的方法。以下是对论文内容的概要和主要章节内容的描述：



摘要

- 论文展示了==掩蔽自编码器==（MAE）是一种用于计算机视觉的可扩展**自监督学习者。**
- MAE 的方法很简单：随机掩蔽输入图像的块，并重建缺失的像素。
- 论文基于两个核心设计：开发了不对称的编码器-解码器架构，并发现掩蔽输入图像的高比例（例如75%）可以产生有意义且非平凡的自监督任务。
- 这些设计使得训练大型模型变得高效和有效，加速了训练过程，并提高了准确性。
- 论文的方法允许学习高容量模型，这些模型泛化能力强，例如，一个普通的 ViT-Huge 模型在仅使用 ImageNet-1K 数据的方法中取得了最佳准确性（87.8%）。



第1章 引言 (Introduction)

- 论文讨论了深度学习模型的增长趋势，以及在硬件快速发展的推动下，模型对数据的大量需求。
- 作者指出，在自然语言处理（NLP）中，自监督预训练方法（如 BERT）已成功应对这一挑战，而在计算机视觉（CV）中，自监督学习的方法却落后于 NLP。
- 论文提出了一种基于 Transformer 的掩蔽自编码器方法，通过掩蔽图像块并重建这些块来学习视觉表示。



第2章 相关工作 (Related Work)

- 论文回顾了在 NLP 中使用掩蔽语言建模和自回归语言建模的自监督预训练方法。
- 作者讨论了自动编码器在表示学习中的应用，并将其与计算机视觉中的去噪自编码器（Denoising Autoencoders, DAE）联系起来。
- 论文还探讨了自监督学习方法在计算机视觉中的研究进展，特别是在图像对比学习方面的工作。



第3章 方法 (Approach)

- 论文详细介绍了 MAE 的设计，包括编码器和解码器的不对称架构。
- 编码器仅对可见的未掩蔽图像块进行操作，而解码器则从潜在表示和掩蔽标记重建完整的信号。
- 论文还讨论了掩蔽策略、重建目标和简单实现等技术细节。



第4章 ImageNet 实验 (ImageNet Experiments)

- 论文描述了在 ImageNet 数据集上进行自监督预训练的实验设置，并评估了所提出方法的表示学习性能。
- 作者通过与监督预训练和自监督预训练的对比，展示了 MAE 在不同模型大小和掩蔽比例下的性能。
- 论文还探讨了训练计划、解码器设计和掩蔽策略对性能的影响。



第5章 迁移学习实验 (Transfer Learning Experiments)

- 论文评估了使用 MAE 预训练的模型在下游任务中的迁移学习能力，包括目标检测、实例分割和语义分割。
- 作者展示了 MAE 在这些任务中相比于监督预训练模型的性能优势。



第6章 讨论和结论 (Discussion and Conclusion)

- 论文总结了 MAE 的主要发现，并讨论了自监督学习在计算机视觉领域的潜力。
- 作者指出，尽管图像和语言在本质上是不同的信号，但 MAE 能够通过丰富的隐藏表示学习到复杂的视觉概念。



附录 (Appendix)

- 论文提供了实验的额外细节，包括训练设置、超参数选择、模型变体的比较等。

整体而言，这篇论文为自监督学习在计算机视觉领域的应用提供了有价值的见解，并展示了掩蔽自编码器作为一种有效方法的潜力。









## 二、MoCo 阅读





MoCo: CVPR 2020 最佳论文，视觉 + 对比学习的里程碑式的工作



对比学习：

- 简单、好用；
- 19年以来，视觉领域乃至整个 ML 领域最火的方向之一；
- 盘活了从17年很卷的CV领域，MoCo是其中的优秀工作之一



**MoCo：无监督的表征学习工作 在 CV上表现怎么样？**

- 分类：逼近了有监督的 baseline
- 检测、分割、人体关键点检测：大幅超越有监督预训练的模型 (ImageNet 上预训练的模型)
- CV 领域的定心丸，无监督学习真的可以，有可能真的不需要大规模、有标号的数据集做预训练。
- 侧面正面了 **Yann LeCun** 的 NeurIPS 2016 的演讲图



![img](https://i0.hdslb.com/bfs/note/c7a1dfae467d9e5e343fc62889daf30895f33af5.png@620w_!web-note.webp)



蛋糕🎂：ML

樱桃🍒：RL

蛋糕的糖霜 icing：有监督学习

**蛋糕本质：无监督学习**



现在无监督的进展：

- NLP 大模型都是自监督的预训练方式
- CV 的大模型用自监督预训练，也快了



Note: 多听大佬的 talk 报告有好处，找下一个研究方向



**什么是对比学习？+ 前人工作**



01:39



MoCo 动量对比学习：假设读者已经了解对比学习



对比学习：通过对比去学习模型，只需要知道图 1 和 图 2 相似，图 1、图 2 和 图 3 不相似；而不需要真的知道 图 1 和 图 2 代表的是人，图 3 代表的是狗。



![img](https://i0.hdslb.com/bfs/note/976897b4dd45be0ec39bb95b2c56f1d02670c32e.png@620w_!web-note.webp)





3 张图进入一个网络 M 得到特征 f1、f2、f3，在一个学习好的特征空间 embedding space 中，f1、f2 的特征尽量近，和 f3 的特征尽量远离。



![img](https://i0.hdslb.com/bfs/note/d7f2fb3a445b105dd1bee24e89c24ff56f19e18f.png@620w_!web-note.webp)



对比学习学到的很好的特征：类似物体在这个特征空间 相邻，不类似的物体在特征空间 远离



类似 meta-learning 的基于度量的学习？





**Q: 图 1 和 图 2 相似，和图 3 都不相似，难道不是有监督学习吗？Why 对比学习在 CV 领域被认为是无监督训练呢？**

==CV 领域 设计巧妙的代理任务 pre-text task，人为设立一些规则 —— 定义哪些图片相似、哪些图片不相似，为自监督学习提供监督信号，从而自监督训练==





**Example 代理任务 instance discrimination 个体判别**



04:30





一个无标注的数据集，n 张图片，x1, x2, ..., xn



**随机选取一张图片，做 transformation** 



以 x1 图片为例，x1 随机裁剪 + 数据增广 得到 xi1, xi2 （看起来和 x 1 有区别的 2 张照片，x1 的正样本），数据集中的其它图片 x_j, j ≠ i 是 x1 的负样本



i.e., ImageNet-1K 此时不是 1000 个类别，而是 100w 个类别。每个图片都是它自己的正样本，其余都是负样本。





![img](https://i0.hdslb.com/bfs/note/6f316ff053611f410064dc29bb20bf7e04759ca2.png@620w_!web-note.webp)



基于 图片和图片本身的变换是正样本，和其它图片是负样本 的代理任务， + 模型 得到特征，+ 对比学习的目标函数 i.e., NCE loss (正文有提)



![img](https://i0.hdslb.com/bfs/note/ca91999f2c17ed82d14dff4129bf45855c4aaa0e.png@620w_!web-note.webp)



==对比学习的框架：灵活性--定义正负样本的规则==

- 同一个视频里的任意两帧 是 正样本，和其它视频的所有帧是负样本
- NLP, simCSE 把同样的句子扔给模型，但是做 2 次 forward，通过不同的 dropout 得到一个句子的 2 个特征；和其它所有句子的特征都是负样本。 
- CMC 论文：一个物体的不同视角 view（正面、背面；RGB 图像、深度图像）作为不同形式的正样本。
- 多模态领域：Open AI 的 CLIP 模型





### 1、题目和作者



07:33



动量对比学习的方法做无监督视觉特征学习



Momentum Contrast: 动量对比学习

- 动量：(指数)加权移动平均值$ y_t = m * y_{(t - 1)} + (1 - m) * x_t$
- m: 动量的超参数
- y_(t - 1): 上一个时刻的输出
- x_t: 当前时刻的输入
- m 趋近于 1，y_t 改变缓慢，当前时刻的输入 x_t 没什么影响
- m 趋近于 0, y_t 更多依赖于当前时刻的输入。
- MoCo 利用动量的特性，缓慢的更新一个编码器，==从而让中间学习到的字典中的特征尽可能保持一致==。



作者来自 FAIR, 大佬 * 5



### 2、摘要



09:12



本文提出 MoCo (动量对比) 做无监督的表征学习。



**MoCo 从什么角度做对比学习呢？**

**dictionary look-up**, 字典查询任务, a dynamic dictionary with a queue and a moving-averaged encoder 动态字典

- **一个队列**：<font color=red>队列中的样本**无需梯度回传**，可以放很多负样本，让字典变得很大</font>
- **一个移动平均的编码器**：让字典的特征尽可能的保持一致
- 一个大的、一致的字典，有利于 无监督的对比学习 训练。



**本文的亮点是什么？**

结果 nice, MoCo （第一个）在 （分类、检测、分割）主流的 CV 任务证明 无监督学习 也不比 有监督学习 差

- ImageNet 分类：linear protocol 做测试，MoCo 和 之前最好的无监督学习方式差不多
- ==linear protocol （类似 MAE 的 linear probing）测试：freeze backbone 主干网络的参数不调整，只微调最后的分类全连接层（分类头）。把 backbone 当成特征提取器，可以证明 backbone 学习图片特征的好坏。==
- 容易迁移到下游任务。**满足了大家对 大规模、无监督训练 的想象：**
- 7 个 下游任务表现好 counterpart: 模型使用的一样的 i.e., Res50，只是训练方式不一样，i.e., 有监督的带标签数据训练，无监督的不带标签的数据训练。（控制-训练方式-变量法）
- 在大规模的数据集上进行无监督训练，模型学到一个很好的特征，而且学到的特征可以迁移，在小、少标注数据的下游任务取得好的结果。



**MoCo 有什么意义呢？**

==填补了 CV 领域的无监督学习和有监督学习的 gap==



### 3、引言



12:04



**第一段：无监督学习为什么在 CV 不成功？原始信号不一样** 

<font color=red>**NLP 的离散单词更具语义性，CV的连续、高维信号不好构建字典**</font>

引入无监督学习的成功：

- 无监督学习在 NLP 很成功, i.e., GPT, BERT
- 无监督学习在 CV 大幅落后于 主流的 有监督学习

无监督在 CV 不成功的原因是什么？

- 原始信号空间的不同
- NLP 原始信号是==离散的，词、词根、词缀==，容易构建 tokenized dictionaries 做无监督学习
- tokenized: 把一个词对应成某一个特征
- Why tokenized dictionaries 有助于无监督学习？
- **把字典的 key 认为是一个类别，有类似标签的信息帮助学习**
- NLP 无监督学习很容易建模，建好的模型也好优化
- CV **原始信号是连续的、高维的，不像单词具有浓缩好的、简洁的语义信息，不适合构建一个字**典
- 如果没有字典，无监督学习很难建模



**第二段：别人怎么用对比学习的方法在 CV 的无监督学习里？dynamic dictionaries**

近期结合 <font color=blue>对比学习和 CV 的无监督学习效果不错</font>，出发点motivation 不一样，但可以被归纳为 **“动态字典法”**



![img](https://i0.hdslb.com/bfs/note/4803a4afa50f2b1e833ed75914cb824b22e31c27.png@630w_!web-note.webp)



x_1^1: anchor

x_1^2: positive

x2, x3, ......, xn: negative  

编码器 E_11 和 E_12 可以一样，可以不一样 



**Q：负样本使用哪个编码器？**

E_12：因为 positive 和 negative 都是相对 anchor f_11 来说的。==正负样本使用同样的编码器==



![img](https://i0.hdslb.com/bfs/note/f8c8012f7ce51fbf339b746f70ee07909b74c501.png@630w_!web-note.webp)



**Q: 对比学习怎么做？**

f11 和 f12 相近，f11 和 f2, f3, ......, fn 远离



**Q: Why 对比学习可以归纳成 在做一个动态的字典 呢？**



15:25



==f11 当成 query 在 f12, f2, f3, ......, fn 组成的字典的 key 特征条目 k1, k2, ...... 里面查找，dictionary look-up 靠近 f12, 远离 f2, f3, ......==

- be similar to its matching key and dissimilar to others
- learning is formulated as minimizing a contrastive loss 最小化对比学习的目标函数



![img](https://i0.hdslb.com/bfs/note/85b651568e280324a74b1e21f79d3714eec2f1f7.png@630w_!web-note.webp)



**第三段：从动态字典的角度看对比学习，什么样的字典才适合呢？ 大 + 一致性**



17:09





- large 
- 从连续高维空间做更多的采样。字典 key 越多，表示的视觉信息越丰富，匹配时更容易找到具有区分性的本质特征。
- 如果 字典小、key 少，==模型可能学到 shortcut 捷径，不能泛化==
- consistent 
- 字典里的 key (k0, k1, k2, ......, kN) 应该由相同的 or 相似的编码器生成
- **如果字典的 key 是由不同的编码器得到的**，query q 做字典查询时，很有可能 找到和 query 使用同一个 or 相似编码器生成的 key，而不是语义相似的 key。另一种形式的 shortcut solution



![img](https://i0.hdslb.com/bfs/note/8c6f711014b2b167d0fbcd98c74d04cb63856f71.png@640w_!web-note.webp)



已有的 CV 对比学习 动态字典方法在 large or consistent 上有不足。



引言结构：介绍研究动机、相关研究工作的不足、想要达到的目标 ---> 本文的工作





**第四段：本文的 MoCo**



19:07





**为什么要提出 MoCo? 给CV 无监督对比学习 构建一个 大 (by queue)+ 一致 (momentum encoder) 的字典**



图1 MoCo 框架图 queue, momentum encoder





![img](https://i0.hdslb.com/bfs/note/23e8f83e2164097c3695e45d55c969d43e87bcab.png@640w_!web-note.webp)



- queue 数据结构: 剥离 字典的大小 和 显卡内存的限制，让字典的大小 和 模型每次做前向传播的 batch size 的大小 分开
- 字典很大（成千上万），意味着要输入很多很多的图片，显卡内存吃不消
- current mini-batch enqueued and the oldest mini-batch dequeued 当前 mini-batch 入队，最早进入队列的 mini-batch 出队
- 队列的大小 == 字典的大小，但是每次做 iteration 更新，并不需要更新字典中所有 key 元素的值。普通 GPU 训练



- momentum encoder: 
- Q：使用 queue，只有当前 mini-batch 的特征是由当前的编码器得到的；==之前的 key 是由不同时刻的编码器抽取的特征，如何保持 consistent 呢==？
- momentum encoder 由 当前时刻的 encoder 初始化而来	
- $theta_k = m * theta_(k-1) + (1-m) * theta_q$
- 动量参数 m 较大时，theta_k 的更新缓慢，不过多的依赖于 theta_q 当前时刻的编码器，即不随着当前时刻的编码器快速改变，尽可能保证 字典里的 key 都是由相似的编码器生成的特征，保证特征的 consistent





<font color=red>关键是保持特征的一致性</font>

基于 large + consistent dynamic dictionary，MoCo 可以很好的无监督学习视觉特征。



**第五段：MoCo 的代理任务 pretext task？ instance discrimination**



22:04



MoCo 建立模型的一种方式，很灵活，可以和很多代理任务使用



**instance discrimination**: （个体判别）query 和 key 匹配 如果它们来自于同一张图片的不同视角, i.e., 不同的裁剪



MoCo 用 instance discrimination 无监督训练 在 ImageNet 上可以和之前最好的结果打个平手 or 更好的表现 competitive results





**第六段：MoCo 的效果怎么样？ 卖结果**



23:05



**无监督学习的目的**：==在一个很大的无标注的数据集上训练，模型学到的特征可以很好的迁移到下游任务==。



MoCo 做到了。7个检测 or 分割的任务表现很不错。超越 ImageNet 有监督训练的结果，甚至有时大幅度超越 in some cases by nontrivial margins.





**无监督学习的期待：更多数据、更大的模型，性能会提升，不饱和。**



MoCo 在 10亿 Instagram 数据集（更糙 relatively curated 真实*******、一张图片有多个物体; ImageNet 数据集的图片大多只有一个图片、在图片中间） 上性能还有提升





中型 ImageNet or 大型 Instagram 数据集，MoCo 把 无监督学习和有监督学习的 坑🕳 填平。



应用展望：之前 ImageNet 预训练好的模型，可以尝试替换为 MoCo 预训练好的模型。



### 4、结论



25:33



==结论：MoCo在一系列的任务和数据集上效果很好 positive result==



- 1000 倍数据集数量的增加， MoCo 性能的提升不高
- 大规模数据集可能没有完全被利用
- 尝试开发其它的代理任务 pretext task
- <font color=red>除了 instance discrimination 代理任务，类似 NLP 的代理任务 masked auto-encoding</font>
- MAE, 大佬 2 年前就有了想法，做了实验；做研究急不来
- 像 NLP 的 BERT 使用 masked language model 完形填空做自监督预训练



==点题总结：MoCo 和 其它对比学习的 代理任务的解和==

- MoCo 设计的初衷：去构造一个大的字典，从而让正负样本能够更有效地去对比，提供一个稳定的自监督信号，最后去训练这个模型





### 5、相关工作



27:30





unsupervised / self-supervised learning: 

- 自监督学习是无监督学习的一种。
- 前人研究不怎么区分，MoCo使用 无监督学习 unsupervised learning (定义更广泛一些)



两个可以做的点：pretext tasks and loss functions

- **代理任务**：不是大家实际感兴趣的任务 (检测、分类、分割实际应用任务)，而是为了 学习一个好的数据特征表示
- **损失函数**：和代理任务可以分开研究。 <font color=red size=5>MoCo 的创新点在损失函数，又大又一致的字典 影响 info NCE 目标函数的计算</font>





28:30





==损失目标函数：衡量 模型的预测输出 和 固定的目标之间的 difference==。

- L1 or L2 losses 
- i.e., Auto-encoder（生成式网络的做法）, 输入一张原图 or 一张被干扰的图，经过编码器、解码器 重构输入的图，衡量是原图 和 重构图 之间的差异。



**判别式网络**：eight positions 2015

一张图片 打成 有序号的 9 宫格，给 中间的 第 5 格 和 剩下随机挑一格，**预测**随机挑的这一格是中间 第5 格 的**方位**（8个方位可选）。



pretext tasks：分类任务，因为每一个方格都自带序号，输出分到 8 个方位的哪一类。





**损失函数：判别式、生成式、对比学习、对抗学习**



- 对比学习的损失：**目标不固定，训练过程中不断改变。目标有编码器抽出来的特征（MoCo 的字典）而决定**
- 判别式：预测 8 个位置中的哪一个方位
- 生成式：重建整张图
- 对比学习的目标：==测量 样本对 在特征空间的相似性==。
- 相似样本离得近，不相似样本离得远
- 最近无监督表现好的文章都用了 contrastive learning (Sec. 3.1 讨论)





- 对抗学习的损失：衡量两个概率分布之间的差异，i.e., GAN
- unsupervised data generation 做无监督的数据生成
- 对抗性的方法做特征学习
- 如果可以生成很好、很真实的图片，模型应该学到数据的底层分布
- GAN 和 NCE 的关系 noise-contrastive estimation Ref. [24]





代理任务的生成：

- denoising auto-encoders 重建整张图
- context auto-encoders 重建某个 patch
- cross-channel auto-encoders (colorization) 给图片上色当自监督信号
- pseudo-labels 图片生成伪标签
- exemplar image 给同一张图片做不同的数据增广，它们都属于同一个类。
- patch ordering 九宫格方法：打乱了以后预测 patch 的顺序, or 随机选一个 patch 预测方位 eight positions 
- 利用视频的顺序做 tracking 
- 做聚类的方法 clustering features



对比学习和代理任务的关系：

- **不同的代理任务 可以和 某种形式的对比学习的目标函数 配对使用**
- MoCo 论文里 instance discrimination 个体判别方法  ++++ examplar based 代理任务很相关
- CPC contrastive predictive coding 用上下文信息预测未来 ++++ context auto-encoding 上下文自编码
- CMC contrastive multiview coding 利用一个物体的不同视角做对比 ++++ colorization 图片上色（同一个图片的 2 个视角：黑白 和 彩色）



**相关工作总结：**



32:38



简洁明了

从 代理任务 和 目标函数 （2 个和有监督学习不同的点）写相关工作



有监督学习的过程



无监督学习 or 自监督学习 缺少 ground truth，没有标签怎么办？

- 代理任务来帮忙，自己造标签。
- 代理任务生成自监督的信号，充当 ground truth 的标签信息



**有输出 y 和 标签信息 ground truth，还需要什么呢？**

==目标函数 L，衡量 输出 Y 和 标签 ground truth 的差异，让模型学到更好==



MoCo 从 目标函数 L 和 代理任务 pretext tasks 生成 ground truth 写相关工作





### 6、MoCo方法



33:44



**3.1 Contrastive learning as dictionary look-up** 



对比学习和最近的发展，都可以看成是一个训练一个 encoder 来做 字典查询 的任务





### 7、实验



01:06:10











### 8、总结



01:23:20



感谢 MoCo 论文和高效实现，普通 GPU 跑对比学习的实验，做激动人心的研究。

MoCo 激励学者研究 “MoCo 学出来的特征 和 有监督学习学出来的特征有什么区别？还能从什么方向提高对比学习？”



**期待对比学习的论文串烧**

第一阶段：Contrastive Predictive Coding (CPC), CMC Contrastive Multiview Coding, 

第二阶段：MoCo v1, simCLR v1, MoCo v2, simCLR v2 

第三阶段：不需要负样本的 BYOL, bootstrap your own latent, SimSiam

第四阶段： 用了 vision transformer 的 MoCo v3, 





