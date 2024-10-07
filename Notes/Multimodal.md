**- 多模态串讲**



> 在多模态领域中视觉特征要远远大于文本特征



根据vilt论文里的figure2, 我们可以得出这样一个结论，我们需要好的visual embed，**图像编码器比文本编码器要大**，因为图像更为复杂，同时**modality interaction也要很好**，text embedding已经很成熟，一般用BERT，所以这个很轻量化了已经

因此我们总结出理想化的情况应该是接近下图(c)的情况

![img](https://i0.hdslb.com/bfs/note/2a7d4c1be6e1388c2974c204f764e548050ae156.png@690w_!web-note.webp)



我们可以考虑一些常用的loss:

Image text contrastive loss。 （ITC）

Image text matching loss。 （ITM）

Masked language modelling loss  （MLM）

Word patch alignment (这个在vilt中用到，但是计算很慢，pass)

所以上面前三个loss是比较好用的



### 1、ALBEF

因此我们就可以引出**ALBEF**

**- ALBEF (align before fuse)**

出发点 - 在multimodal interaction之前我们要align好text and image token，以便于multimodal encoder学习。**AL**ign image and text representation **BE**fore **F**using (ALBEF) using a contrastive loss, 这是**贡献1**

先对齐再融合：利用动量提炼进行视觉和语言表征学习





**贡献2 -** Momentum distillation, self-training method which learns from pseudo-targets produced by a momentum model

不同的损失函数其实是在为同一个图像文本对，生成不同的视角，变相地做data augmentation，达到semantic preserving的目的

**主体方法：**

![img](https://i0.hdslb.com/bfs/note/4fb0ac7cc7c96b0135ee55f6dbf66ea313445e5d.png@690w_!web-note.webp)

目标函数：

1. ITC loss, image text contrastive loss. 图像和文本分别通过encoder tokenise, <font color=red>CLS token是一个全局特征（图中绿色方块旁边的黄色方块）</font>， down sample (786x1 => 256x1)然后 normalisation，然后进行正负样本的学习 （预先存了很多个负样本），  这一步就是align  (和MoCo完全一样)
2. ITM loss, image text matching loss. 在multimodal encoder的输出之后加一个==二分类头==，这里很特别的是，每个batch里我拿一个图片和batch里除了配对文本之外的所有的文本做cosine similarity (借助之前ITC的那个模块)，挑一个相似度最高的作为负样本 (hard negative) 来训练，加大难度
3. MLM, masked language modeling. ==类似BERT的完形填空==，mask住一个词语，去预测mask的词语，但是融合了图像的信息

一个小细节，计算ITC和ITM loss的时候，输入的都是<font color=blue>原始的image and text embedding </font>（下图橙色的T'表示masked text embedding），算MLM loss的时候，用的是原始的image embedding，但是是masked后的text embedding，因此每一次训练iteration其实做了2次forward，一次用了原始的image and text embedding，另一次用了原始的image和masked的text embedding，因为你要算多个loss函数

![img](https://i0.hdslb.com/bfs/note/3528a305340ecc475102931cb54d9e4c2053f898.png@690w_!web-note.webp)



![img](https://i0.hdslb.com/bfs/note/cd89b73525d9cb66efc7a5b1ba126ad347f16d49.png@690w_!web-note.webp)

**Momentum distillation**

动机 - ==从网上爬下来的图像文本对通常weakly-correlated==，即文本并没有很好地描述图像，从而产生了noise

![img](https://i0.hdslb.com/bfs/note/1b4c12b0fbfdb5aeb5f258439a1005d980f44b23.png@690w_!web-note.webp)

如果可以找到额外的监督信号，那就好了，这里额外的就是momentum model产生的一些pesudo-target，实际上是一个softmax score，当ground-truth有noise的时候，pesudo-target就很有用了

最近很火的一种就是==self-training==

![img](https://i0.hdslb.com/bfs/note/740c6fc58d797d0bd79ec1f8508c3d5e9904a2a7.png@690w_!web-note.webp)

因为之前是one-hot label (==对应的image和text就是1，不对应的地方是0==)，现在momentum model输出的是一个softmax score，也就是不是one-hot，而是一些momentum model输出的对于可能正确的one-hot label的概率，因此不再用cross-entropy而是用KL-divergence来计算loss

![img](https://i0.hdslb.com/bfs/note/2afd73cc6aedc34c57f1d004efc4723c33db492c.png@690w_!web-note.webp)

所以实际上有5个loss，2个ITC，2个MLM，1个ITM，ITM这个loss ground truth很清晰，所以不需要momentum的版本





**自训练（self-training）** 的技术在模型训练中的应用，尤其是在==视觉模型中使用动量模型（momentum model）生成伪标签（pseudo targets）的方式==。

- 早期Google的Noisy Student在ImageNet上提高了模型性能，**近期的DINO也属于自训练的一种。**
- 文章作者采用了自训练的方式，构建了动量模型，并通过**指数移动平均（EMA）** 来生成伪标签。伪标签不再是传统的one-hot标签，而是通过softmax得到的得分。
- 训练目标不仅是模型的预测结果接近真实标签，还要求其预测结果尽量与动量模型生成的伪标签相匹配，以提高训练效果。
- 举例来说，原有的ITC损失基于one-hot标签，==现在则加入伪标签损失，计算**KL散度** 而非交叉熵，并为原始损失和伪标签损失设置权重，最终形成**动量版本的ITC损失**==。
- 类似的，自训练还应用到了MLM损失上。
- 最终，ALBEF模型的训练损失包括5个部分：两个ITC损失、两个MLM损失和一个ITM损失。由于ITM任务是基于ground truth的二分类任务，且已经引入了硬负样本，与动量模型有冲突，所以没有采用自训练版本。

作者还通过图例证明了伪标签比真实标签更有利于监督训练。



### 2、VLMo

 **贡献1** - dual-encoder (双塔模型，如CLIP) 解决了检索问题，而fusion encoder，也叫单塔模型，解决了不同模态之间的交互问题，VLMo就把2种的好处都结合了起来，一个模型，想当双塔和单塔 (论文命名为vision-language expert, language expert, vision expert，其实就是不共享参数的FC层) 用都可以，具体可以看论文的图	

贡献2 - stage-wise pre-training, 简单来说就是多模态的数据集不够大，那我就先预训练单独的一个模态

![img](https://i0.hdslb.com/bfs/note/9cbd34253d8216d7c4f3d6f61e41d51e937646ee.png@690w_!web-note.webp)



注意这里由于loss的计算，和ALBEF类似，每次iteration也要forward几次





多模态学习领域的论文 **VLMo**，该团队以推出多项重要研究成果闻名（如BEiT、LayoutLM等），在多模态研究中具有很高的权威性。**VLMo** 的主要创新点在于模型结构和训练方式的改进，具体分为以下两部分：



1. **模型结构的改进：Mixture-of-Modality-Experts**

   VLMo 引入了一种新的模型架构，称为**Mixture-of-Modality-Experts**（模态专家混合）。该架构结合了当前多模态学习中的两大主流模型结构：

   - **双塔模型（Dual-encoder）**：如 CLIP，图像和文本分别使用不同的编码器处理，两者之间的交互非常简单，通常通过计算**余弦相似度**来完成。这种结构在检索任务上表现极好，因其高效性，特别适用于大规模检索场景。但其交互方式过于浅显，无法处理复杂的多模态任务，如视觉语言分类任务（VR、VQA等）。
   - **单塔模型（Fusion-encoder）**：==如 ViLT，图像和文本先经过独立处理，之后通过**Transformer**进行深度模态交互==。该方法在复杂的视觉语言任务上表现优秀，但在大规模数据检索任务中效率低下，因为每对图像-文本对必须同时进行编码，推理时间极长。

   为了解决双塔模型和单塔模型的各自局限性，VLMo 提出了**Mixture-of-Modality-Experts**：
   - 模态交互的**自注意力层（Self-Attention）**是共享的，适用于所有模态。
   - **前馈层（Feed Forward Layer, FC 层）**采用不同的专家模块：图像有视觉专家（Vision Expert），文本有语言专家（Language Expert），多模态则有多模态专家（Multi-modal Expert）。

   这种设计使得 VLMo 能在推理过程中灵活切换：
   - 当需要高效检索时，可以像双塔模型一样工作，使用双编码器进行快速特征提取和相似度计算。
   - 当需要处理复杂多模态交互时，可以像单塔模型一样工作，通过专家模块进行深度交互。这种灵活性使得 VLMo 在不同任务中表现优越。



2. **训练方法的改进：分阶段预训练策略（Stagewise Pre-training Strategy）**

   论文的另一个创新点是**分阶段预训练策略**，其动机来源于数据集的限制与多模态学习的需求。多模态数据（同时包含视觉和文本）相比单模态数据来说相对稀缺。传统的大规模多模态数据集，如 CLIP 使用的 WIT 数据集，并没有公开，而 LAION 这样的大规模开源数据集当时也未推出。因此，构建一个规模足够大的多模态数据集用于训练具有挑战性。

   为了解决数据集不足的问题，VLMo 作者提出了一种创新的分阶段预训练策略：
   1. **单模态预训练**：首先，利用现有的大规模单模态数据集（如 ImageNet 等视觉数据集和大规模文本数据集）分别训练视觉和语言的专家模块。这样，视觉专家会在视觉数据上得到充分的预训练，语言专家则在文本数据上进行预训练。
   2. **多模态联合预训练**：接着，利用多模态数据（如包含图像和文本的配对数据集）进一步在这些已初始化的模型基础上进行预训练。通过这种方式，模型在视觉和语言模态上已经具有较好的初始化，因此在多模态学习时能快速适应并表现出更好的性能。

   这种**分阶段的训练方法**有以下优势：
   - **有效利用单模态的大规模数据**：即使多模态数据较少，仍然能通过单模态的丰富数据进行预训练，为多模态任务打下良好基础。
   - **模型参数良好初始化**：由于视觉和语言专家已经在各自领域得到了良好的训练，多模态学习时只需进行微调，显著提升模型的效果。

   实验结果表明，这种策略显著提升了 VLMo 在多模态任务上的表现。



3. **对大规模数据训练的展望**

   VLMo 的分阶段预训练策略还解决了多模态学习中的另一个问题：**大规模数据集的扩展性**。在 NLP 中，随着数据量的增加，模型性能往往会不断提升，但在视觉领域，这种数据规模的扩展效应没有那么显著。然而，在多模态学习中，文本的参与可能带来类似的扩展效果。VLMo 通过分阶段预训练，能够在未来利用更大规模的多模态数据集（如后来推出的 LAION 数据集）进一步提高性能。



### 3、BLIP



1. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

代码: https://github.com/salesforce/BLIP



本文是 ALBEF 原班人马做的，基本可以看做吸收了 VLMo 思想的 ALBEF。训练的 loss 和技巧都与 ALBEF 一致，属于 ALBEF 的后续工作



关键的改进：

1. <font color=red>模型结构上整合了 ALBEF 和和 VLMo</font>。VLMo 参数共享，但是不存在单独编码器；ALBEF 存在单独编码器但是部分参数不共享。这篇论文存在单独的 vision encoder 和 text encoder。多模态的参数是以 cross-attention 模块插入到文本编码器实现的，cross-attention 模块享受文本编码器的参数（可以看 col 2 和 col3）
2. 增加了解码器（参考 col 4），为了做生成任务。解码器拿到视觉特征和未掩码的语言特征，过一个 casual self-attention 层，做 GPT 用的那种 lm 任务。这里区别于 MLM 的那种 mask 机制，是通过 causal self-attention 来实现因果推理的，我此时还不熟悉这个过程。

![img](https://i0.hdslb.com/bfs/note/473ad2f82afdf7fe31fbbfb512bd2fed62fd9d67.png@690w_!web-note.webp)

上图中  左边三个就是ALBEF

3. 除了上面的主要部分，还有一个重要的部分是利用训练好的模型生成伪标签。将训练好的模型里的不同的部分拿出来在 COCO 上稍微微调一下，decoder 部分可以生成文本，算 ITM loss 的那个模块可以做 image-text pair 的过滤，通过输出打分、置信度的方式。在实验中，BLIP 的解码能力似乎很强，用这种范式生成的文本不仅人看着觉得不错，用于自训练后也可以涨点 2-3，非常显著。

 一个例子是 stable diffusion 的官方博文里提到了，他们在做微调时，会遇到数据集只有图片没有 caption 的情况，比如 pokeman 数据。他们用 BLIP 来做caption生成，然后微调 stable diffusion 发现效果很好。

 另一个例子是知名的开源多模态数据集 LAION，他们也用了 BLIP 来辅助制作数据集。他们的过程在官网公布了，可以参考。

![img](https://i0.hdslb.com/bfs/note/48274ecb0a30b82fc4576998880324ccf1114874.png@690w_!web-note.webp)

---

Cap Filter Model 的研究动机主要是为了解决大规模图像-文本对数据集中的噪声问题，尤其是在从网络爬取的数据集中的图像和文本匹配不良的情况下，提升模型的预训练质量。具体可以分为几个关键点：

1. **数据集中的噪声问题**

许多现代多模态模型（如 CLIP 等）使用了大量从网络爬取的图像-文本对作为训练数据。然而，这些爬取的数据常常包含不匹配的图像和文本描述（TW），导致数据质量不高。而相对来说，手工标注的数据集（如 COCO 数据集）通常具有更高的文本匹配度。

2. **噪声数据对模型性能的影响**

从不匹配的噪声数据中训练出的模型，其效果通常不理想。为了提高预训练的效果，需要对这些噪声数据进行筛选，使得图像和文本对更加匹配。Cap Filter Model 的提出正是为了清理数据集，提升训练效果。

3. **Cap Filter Model 的核心思想**

Cap Filter Model 通过预训练的多模态模型（BLIP）来评估图像和文本对之间的相似性，并使用这个相似度得分来筛选掉不匹配的图像-文本对（红色的 TW），从而保留高质量的匹配（绿色的 TW）。该过程称为 "filter"。

4. **Filter 模块的训练**

Filter 模块的核心在于使用一个预训练好的 BLIP 模型（包含图像模型和两个文本模型）。作者通过在 COCO 数据集上对该模型进行快速微调，得到一个能评估图像和文本匹配度的模型。使用这个 Filter 模块，可以将从网络爬取的噪声数据过滤，使其更为干净和匹配。

5. **Captioner 的引入**

作者发现，经过训练的 BLIP 模型生成的文本（caption）有时比原始的图像-文本对描述得更好。因此，作者引入了一个 Captioner 模块，利用这个模块为图像生成更高质量的文本描述（TS），进一步优化数据集质量。

6. **生成合成数据集**

通过 Filter 和 Captioner 两个模块，Cap Filter Model 不仅清理了噪声数据，还生成了新的、更优质的图像-文本对数据集。这些数据包括：
   - **IWTW**：原始从网络爬取的数据集，经过 Filter 筛选后，保留了更高质量的部分。
   - **IWTS**：使用 Captioner 生成的合成图像-文本对。
   - **IHTH**：手工标注的高质量数据集（如 COCO）。

7. **结果提升和创新**

经过 Cap Filter 模型的处理，数据集不仅质量更高，规模也更大。这为后续的模型预训练提供了更好的基础。重新训练的 BLIP 模型表现显著提升，同时带来了许多新的应用场景。通过这种数据集自举（bootstrapping）策略，Cap Filter Model 成为了提升多模态模型训练效果的重要创新。

总结来说，Cap Filter Model 的核心动机是解决网络爬取数据中图像-文本匹配不良的问题，通过过滤噪声数据和生成高质量的文本描述，来提高多模态模型的训练效果。

总结：

个人感觉模型部分的改进可能有用可能没有用，但是解码器输出的 caption 确实是不错。以至于很多下游任务都拿 BLIP 来生成 caption。



### 4、CoCa



2. CoCa: Contrastive Captioners are Image-Text Foundation Models

github: https://github.com/lucidrains/CoCa-pytorch



它也是 ALBEF 的后续工作，模型非常像。区别在于：

1. 图像用了 attentional pooling，这在本文的实验中有效
2. 去掉了 ITM loss，目的是加快训练，原本文本需要 forward 2-3 次，去掉 ITM loss 之后只需要 forward 一次就可以了。在 ALBEF 中，ITM 需要完整的 text，而 MLM 需要掩码，所以是两次输入。在 BLIP 中，ITC 一次，ITM 因为在文本模型中插入了新的模块，所以得单独做前向。而 LM 因为用了既多了新的模块又得用 causal self-attention 所以又得单独做一次。在 CoCa 中，为了完成 captioning loss 和 ITC loss，只需要做一次前向即可。GPT 中把 cls-token 放在最后面就可以得到全局表征来做 ITC loss 了。

​	简单快速的方法可以有效地 scale，而我们知道复杂的模型设计、loss 设计经常不如简单地放大模型、增加数据有效。参考凯明的 FLYP。

![img](https://i0.hdslb.com/bfs/note/61528597bfcf062fcec9794a280f25707a5c98e8.png@690w_!web-note.webp)





这种画图的方式很不错，很直观。可以参考，以后也画成这样。

![img](https://i0.hdslb.com/bfs/note/180d98850dacef21cb2ca66eaffb01133fefd91d.png@690w_!web-note.webp)





总结：

简单有效的结构设计，我对 CoCa 的印象是简单有效。它的峰值性能我没有感觉很炸裂，可能是模型、数据 scale 之后自然的结果。但是它的 zero-shot 性能让我印象很深刻，在 imagenet 上微调不微调的差距很小，这一点非常非常关键。

读到 coca，我对多模态的疑问还有两点：

1. mixture of experts 的结构没有在本文中得到应用，但我感觉是个相当有前途的结构
2. 双向的生成 loss 还是没人做，谁说只能图像辅助文本?



### 5、BeiTV



3. (BEiT-3) Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks

论文的卖点是大一统。在 introduction 章节详细介绍了大一统指的是统一模型、loss 和数据。我觉得可以简单地概括为：用统一的 multi-way transformer (mixture of experts ) 架构和单个 masked modeling loss，将任意模态看做是同一个模态来建模。

==这篇文章的引言部分非常好 foundation==

具体而言，它指的是在将任意模态输入网络后，都表现为 list of tokens，直接将它们看做是相同的模态来做 masked modeling 就好了。如果想要拿过去做下游任务的话，直接将需要的那部分模型拿出来即可。比如做视觉任务就拿视觉模型，做语言任务就拿语言模型。如果是做多模态任务，可以灵活地模拟不同的需求，比如：1. 做生成任务可以拿多模态部分的参数出来 2. 做图文检索可以单独取出视觉部分和语言部分来模拟 CLIP。不仅仅是能做任意任务，还继承了前作的优点，比如 CLIP 这种弱跨模态交互带来的计算效率的优势。

![img](https://i0.hdslb.com/bfs/note/b44d0b350e43cd0460b7327d7a201c66fcef8ee6.png@690w_!web-note.webp)



### 6、总结



![image-20241007141539205](/Users/zhihongli/Documents/Course/MachineLearningNotes-master/pic/image-20241007141539205.png)

这段文字讨论了多模态学习的发展历程和关键模型的演变，重点分析了不同模型在视觉和文本多模态任务中的作用和改进。以下是详细的总结概括：

1. **Vision Transformer (ViT) 的引入**

传统多模态学习模型（如 Oscar 和 Uniter）依赖于 Object Detection 模型来提取视觉特征，导致训练效率低且计算成本高。Vision Transformer (ViT) 的引入改变了这一局面，通过简化模型结构，仅使用一个嵌入层就可以处理视觉信息。这种方法让模型更加高效，因此，ViLT 模型应运而生，它结合了 Vision Transformer 并去掉了复杂的视觉特征提取过程。

2. **ViLT 和 CLIP 的对比与 ALBEF 的诞生**

ViLT 简化了结构，而 CLIP 模型则通过对比学习方法，在图像和文本检索任务上表现出色。ALBEF 模型则结合了 ViLT 和 CLIP 的优势，通过融合编码器（Fusion Encoder）的模式，在多模态任务中表现优异。ALBEF 的简单性和高效性使其成为多模态研究中的基础模型，带来了许多后续工作。

3. **SimVLM、CoCa 和 VLMO 的发展**

SimVLM 基于 ALBEF，使用编码器-解码器架构进行多模态学习，并推出了 CoCa 模型，通过对比学习和生成描述的损失函数提升了性能。与此同时，微软推出的 VLMO 则使用参数共享的方式，创建了统一的多模态框架。随着研究的进展，ALBEF 的作者推出了 Blip 模型，进一步强化了图像描述生成能力。

4. **BEIT 系列和 BEITv3 的创新**

Vision Transformer 推动了自监督学习的发展，其中 Masked Data Modeling (MDM) 成为关键技术。微软提出的 BEIT 被誉为计算机视觉领域的 "BERT 时刻"。在 BEIT 的基础上，作者结合了视觉和语言的 Masked Modeling，推出了多模态的 BEITv3，它在单模态和多模态任务上超越了之前的模型（如 CoCa 和 Blip）。

5. **Masked Autoencoder (MAE) 和 Fast Language Image Prediction (FLIP)**

MAE 使用了 Masked Data Modeling 的方法，通过掩盖大量的视觉 Patch，减少计算量，同时提高了训练效率。基于这一思路，Facebook 推出了 FLIP 模型，它结合了 CLIP 和 MAE 的优点，通过在视觉端掩盖部分 token，降低了序列长度，加快了训练速度。

6. **多模态学习的未来趋势：Unified Framework 和 Generalist Model**

多模态学习的发展极为迅速，当前的研究方向之一是使用语言作为界面（Language as Interface）来构建统一的多模态框架。例如，微软的 MetaLM 和 Google 的 Poli 通过语言提示（prompt）来控制模型的行为，让模型在执行不同任务时（如 VQA 或图像分类）都以文本形式输出结果。  
此外，"Generalist Model"（通才模型）也正在发展中，它试图通过单一模型解决所有任务，而无需为每个任务调整模型结构。这类模型（如 Unified IO 和 UniPerceiver）正在逐渐取得进展，尽管性能尚未达到最优，但预计在未来的发展中将超过专用任务模型。

总结：

这段讨论展示了多模态学习领域从依赖复杂特征提取到使用简化架构的转变，以及通过融合编码器和 Masked Modeling 等技术的创新提升模型性能的过程。研究的重点逐渐转向统一的、多任务模型，以减少不同任务间的模型结构调整。