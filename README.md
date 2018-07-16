# awesome-ocr

 A curated list of promising OCR resources



## Librarys

* [百度api store](http://apistore.baidu.com/astore/servicesearch?word=ocr&searchType=null)
```
有2个api
都支持图片
百度自家的 ：基本可以放弃
化验单识别：也只能提取化验单上三个字段的一个
```

* [阿里云市场](https://market.aliyun.com/products/#ymk=%7B%22keywords%22:%22ocr%22,%22pageSize%22:10,%22saleMode%22:0,%22pageIndex%22:1,%22categoryId%22:%22%22%7D)
```
第三方和阿里自己提供的 API 集中在身份证、银行卡、驾驶证、护照、电商商品评论文本、车牌、名片、贴吧文本、视频中的文本，多输出字符及相应坐标，卡片类可输出成结构化字段，价格在0.01左右
另外有三家提供了简历的解析，输出结果多为结构化字段，支持文档和图片格式 价格在0.1-0.3次不等
```


* [腾讯云](https://cloud.tencent.com/document/product/641/12399)
```
目前无第三方入驻，仅有腾讯自有的api 涵盖车牌、名片、身份证、驾驶证、银行卡、营业执照、通用印刷体，价格最高可达0.2左右。
```

* [ Codes And Documents For OcrKing Api ](https://github.com/AvensLab/OcrKing)
```
OcrKing 从哪来?

OcrKing 源自2009年初 Aven 在数据挖掘中的自用项目，在对技术的执着和爱好的驱动下积累已近七载经多年的积累和迭代，如今已经进化为云架构的集多层神经网络与深度学习于一体的OCR识别系统2010年初为方便更多用户使用，特制作web版文字OCR识别，从始至今 OcrKing一直提供免费识别服务及开发接口，今后将继续提供免费云OCR识别服务。OcrKing从未做过推广，

但也确确实实默默地存在，因为他相信有需求的朋友肯定能找得到。欢迎把 OcrKing 在线识别介绍给您身边有类似需求的朋友！希望这个工具对你有用，谢谢各位的支持！

OcrKing 能做什么?

OcrKing 是一个免费的快速易用的在线云OCR平台，可以将PDF及图片中的内容识别出来，生成一个内容可编辑的文档。支持多种文件格式输入及输出，支持多语种（简体中文，繁体中文，英语，日语，韩语，德语，法语等）识别，支持多种识别方式， 支持多种系统平台， 支持多形式API调用！
```

* [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)           

* [Tesseract.js is a pure Javascript port of the popular Tesseract OCR engine. ](http://tesseract.projectnaptha.com/)

* [tesseract is an R package providing bindings to Tesseract.](https://github.com/ropensci/tesseract)

* [List of Tesseract add-ons including wrappers in different languages.](https://github.com/tesseract-ocr/tesseract/wiki/AddOns)

* [ Ocular is a state-of-the-art historical OCR system. ](https://github.com/tberg12/ocular/) 

* [sfhistory  Making a map of historical SF photos -博文4所带库 ](https://github.com/danvk/sfhistory)                              

* [ocropy-论文1所带库 by Adnan Ul-Hasan](https://github.com/tmbdev/ocropy)              

* [ A small C++ implementation of LSTM networks, focused on OCR.by Adnan Ul-Hasan ](https://github.com/tmbdev/clstm)

* [ End to end OCR system for Telugu. Based on Convolutional Neural Networks. ](https://github.com/TeluguOCR/banti_telugu_ocr )    

* [ Telugu OCR framework using RNN, CTC in Theano & Python3. ](https://github.com/rakeshvar/chamanti_ocr)

* [ Recurrent Neural Network and Long Short Term Memory (LSTM) with Connectionist Temporal Classification implemented in Theano. Includes a Toy training example. ](https://github.com/rakeshvar/rnn_ctc )

* [ implement CTC with keras? #383 ](https://github.com/fchollet/keras/issues/383#issuecomment-166850153)         

* [mxnet and ocr ](https://github.com/dmlc/mxnet/issues/1023#issuecomment-167189233)        

* [ An OCR-system based on Torch using the technique of LSTM/GRU-RNN, CTC and referred to the works of rnnlib and clstm.](https://github.com/edward-zhu/umaru)

* [ pure javascript lstm rnn implementation based on ocropus ](https://github.com/naptha/ocracy)

* ['caffe-ocr - OCR with caffe deep learning framework' by pannous ](https://github.com/pannous/caffe-ocr)     


* [ A implementation of LSTM and CTC to recognize image without splitting ](https://github.com/aaron-xichen/cnn-lstm-ctc)

* [ RNNSharp is a toolkit of deep recurrent neural network which is widely used for many different kinds of tasks, such as sequence labeling. It's written by C# language and based on .NET framework 4.6 or above version. RNNSharp supports many different types of RNNs, such as BPTT and LSTM RNN, forward and bi-directional RNNs, and RNN-CRF. ](https://github.com/zhongkaifu/RNNSharp)           

* [warp-ctc A fast parallel implementation of CTC, on both CPU and GPU. by  BAIDU](https://github.com/baidu-research/warp-ctc)        
```
Connectionist Temporal Classification is a loss function useful for performing supervised learning on sequence data, without needing an alignment between input data and labels. For example, CTC can be used to train end-to-end systems for speech recognition, which is how we have been using it at Baidu's Silicon Valley AI Lab.

Warp-CTC是一个可以应用在CPU和GPU上高效并行的CTC代码库 （library） 介绍 CTCConnectionist Temporal Classification作为一个损失函数，用于在序列数据上进行监督式学习，不需要对齐输入数据及标签。比如，CTC可以被用来训练端对端的语音识别系统，这正是我们在百度硅谷试验室所使用的方法。 端到端 系统 语音识别
```    


* [ Test mxnet with own trained model,用训练好的网络模型进行数字，少量汉字，特殊字符（./等）的识别（总共有210类）](https://github.com/mittlin/mxnet_test)           

* [ An expandable and scalable OCR pipeline ](https://github.com/OpenPhilology/nidaba)         

* [OpenOCR makes it simple to host your own OCR REST API.](http://www.openocr.net/)          

* [ OCRmyPDF   uses Tesseract for OCR, and relies on its language packs. ](https://github.com/jbarlow83/OCRmyPDF)       

* [ OwncloudOCR uses tesseract OCR and OCRmyPDF for reading text from images and images in PDF files. ](https://github.com/Pogij/owncloudOCR)

* [ Nextcloud OCR (optical character recoginition) processing for images and PDF with tesseract-ocr, OCRmyPDF and php-native message queueing for asynchronous purpose. http://janis91.github.io/ocr/ ](https://github.com/janis91/ocr)

* [ 多标签分类,端到端的中文车牌识别基于mxnet, End-to-End Chinese plate recognition base on mxnet](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)      
* [中国二代身份证光学识别 ](https://github.com/KevinGong2013/ChineseIDCardOCR)      
* [ SwiftOCR:Fast and simple OCR library written in Swift ](https://github.com/garnele007/SwiftOCR)     
* [Attention-OCR :Visual Attention based OCR ](https://github.com/da03/Attention-OCR)            
* [ Added support for CTC in both Theano and Tensorflow along with image OCR example. #3436](https://github.com/fchollet/keras/blob/master/examples/image_ocr.py)     
* [EasyPR是一个开源的中文车牌识别系统，其目标是成为一个简单、高效、准确的车牌识别库。](https://github.com/liuruoze/EasyPR)      
* [Deep Embedded Clustering  for OCR based on caffe](https://github.com/piiswrong/dec)       
* [ Deep Embedded Clustering  for OCR based on  MXNet](https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py)     
* [ The minimum OCR server by Golang The minimum OCR server by Golang, and a tiny sample application of gosseract.](https://github.com/otiai10/ocrserver)        
* [ A comparasion among different variant of gradient descent algorithm This script implements and visualizes the performance the following algorithms, based on the MNIST hand-written digit recognition dataset:](https://github.com/mazefeng/sgd-opt)     
* [ A curated list of resources dedicated to scene text localization and recognition ](https://github.com/chongyangtao/Awesome-Scene-Text-Recognition)    
* [ Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition. ](https://github.com/bgshih/crnn)   

* [ Implementation of the method proposed in the papers " TextProposals: a Text-specific Selective Search Algorithm for Word Spotting in the Wild" and "Object Proposals for Text Extraction in the Wild" (Gomez & Karatzas), 2016 and 2015 respectively. ]( https://github.com/lluisgomez/TextProposals)          

* [ Word Spotting and Recognition with Embedded Attributes http://www.cvc.uab.es/~almazan/index/projects/words-att/index.html ](https://github.com/almazan/watts)          

* [Part of eMOP: Franken+ tool for creating font training for Tesseract OCR engine from page images.](https://github.com/Early-Modern-OCR/FrankenPlus)      

* [NOCR NOCR is an open source C++ software package for text recognition in natural scenes, based on OpenCV. The package consists of a library, console program and GUI program for text recognition.](https://github.com/honzatran/nocr)      

* [An OpenCV based OCR system, base to other projects Uses Histogram of Oriented Gradients (HOG) to extract characters features and Support Vector Machines as a classifier. It serves as basis for other projects that require OCR functionality.](https://github.com/eduardohenriquearnold/OCR/)    

* [Recognize bib numbers from racing photos](https://github.com/gheinrich/bibnumber)

* [Automatic License Plate Recognition library http://www.openalpr.com](https://github.com/openalpr/openalpr)        

* [汽车挡风玻璃VIN码识别](https://github.com/DoctorDYL/VINOCR)
 
* [](https://github.com/matthill/DemoOpenCV)

* [Image Recognition for the Democracy Project with codes](https://github.com/democraciaconcodigos/recon)    

* [Tools to be evaluated prior to integration into Newman](https://github.com/Sotera/newman-research)

* [Text Recognition in Natural Images in Python](https://github.com/FraPochetti/ImageTextRecognition)

* [ 运用tensorflow实现自然场景文字检测,keras/pytorch实现crnn+ctc实现不定长中文OCR识别](https://github.com/chineseocr/chinese-ocr)

* [A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs](https://github.com/vicariousinc/science_rcn)

* [STN-OCR: A single Neural Network for Text Detection and Text Recognition](https://github.com/Bartzi/stn-ocr)    

* [ Digit Segmentation and Recognition using OpenCV and MLP test ](https://github.com/kyper999/OCR)

* [ctpn based on tensorflow](https://github.com/eragonruan/text-detection-ctpn)

* [ctpn based on caffe](https://github.com/tianzhi0549/CTPN)

* [A Python/OpenCV-based scene detection program, using threshold/content analysis on a given video. http://pyscenedetect.readthedocs.org](https://github.com/Breakthrough/PySceneDetect)    

* [Implementation of the seglink alogrithm in paper Detecting Oriented Text in Natural Images by Linking Segments](https://github.com/dengdan/seglink)
>检测单词，而不是检测出一个文本行

* [ Arbitrary-Oriented Scene Text Detection via Rotation Proposals](https://github.com/mjq11302010044/RRPN)

* [通过旋转候选框实现任意方向的场景文本检测  Arbitrary-Oriented Scene Text Detection via Rotation Proposals ](http://www.jianshu.com/p/379dede5979c)

* [Seven Segment Optical Character Recognition](https://github.com/auerswal/ssocr)
![](https://www.unix-ag.uni-kl.de/~auerswal/ssocr/six_digits.png)

* [SVHN yolo-v2 digit detector](https://github.com/penny4860/Yolo-digit-detector)     

* [Reads Scene Text in Tilted orientation.](https://github.com/jumzzz/Tilt-Scene-Text-OCR)     

* [ocr, cnn+lstm (CTPN/CRNN) for image text detection](https://github.com/Li-Ming-Fan/OCR-CTPN-CRNN)     

* [A stand alone character recognition micro-service with a RESTful API](https://github.com/gunthercox/ocr-service)    

* [Single Shot Text Detector with Regional Attention](https://github.com/BestSonny/SSTD)    

* [gocr is a go based OCR module](https://github.com/eaciit/gocr)      

* [GOCR is an optical character recognition program, released under the](https://github.com/SureChEMBL/gocr)

* [UFOCR (User-Friendly OCR). It is YAGF fork: https://github.com/andrei-b/YAGF  Supported input format: PDF, TIFF, JPEG, PNG, BMP, PBM, PGM, PPM, XBM, XPM.](https://github.com/ZaMaZaN4iK/ufocr)


## Papers

* [论文1 can we build language-independent ocr using lstm networks by Adnan Ul-Hasan](https://www.google.co.jp/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjS4uDg1Y3MAhUDn6YKHe-YAqcQFggbMAA&url=http%3A%2F%2Fdl.acm.org%2Fcitation.cfm%3Fid%3D2505394&usg=AFQjCNHvV9kiHl181IaXAUC1zZLkd2LFdg)                  

* [Adnan Ul-Hasan的博士论文](https://github.com/wanghaisheng/awesome-ocr/raw/master/papers/Generic%20Text%20Recognition%20using%20Long%20Short-Term%20Memory%20Networks-PhD_Thesis_Ul-Hasan.pdf)             
* [Applying OCR Technology for Receipt Recognition](http://pan.baidu.com/s/1qXQBQiC)     

* [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](http://arxiv.org/pdf/1507.05717v1.pdf)

* [Reading Scene Text in Deep Convolutional Sequences](http://arxiv.org/pdf/1506.04395v2.pdf)           

* [What You Get Is What You See:A Visual Markup Decompiler](http://arxiv.org/pdf/1609.04938v1.pdf)
>     Building on recent advances in image caption generation and optical character recognition (OCR), we present a general-purpose, deep learning-based system to decompile an image into presentational markup. While this task is a well-studied problem in OCR, our method takes an inherently different, data-driven approach. Our model does not require any knowledge of the underlying markup language, and is simply trained end-to-end on real-world example data. The model employs a convolutional network for text and layout recognition in tandem with an attention-based neural machine translation system. To train and evaluate the model, we introduce a new dataset of real-world rendered mathematical expressions paired with LaTeX markup, as well as a synthetic dataset of web pages paired with HTML snippets. Experimental results show that the system is surprisingly effective at generating accurate markup for both datasets. While a standard domain-specific LaTeX OCR system achieves around 25% accuracy, our model reproduces the exact rendered image on 75% of examples. 

* [ Recursive Recurrent Nets with Attention Modeling for OCR in the Wild](https://arxiv.org/abs/1603.03101)
>   We present recursive recurrent neural networks with attention modeling (R2AM) for lexicon-free optical character recognition in natural scene images. The primary advantages of the proposed method are: (1) use of recursive convolutional neural networks (CNNs), which allow for parametrically efficient and effective image feature extraction; (2) an implicitly learned character-level language model, embodied in a recurrent neural network which avoids the need to use N-grams; and (3) the use of a soft-attention mechanism, allowing the model to selectively exploit image features in a coordinated way, and allowing for end-to-end training within a standard backpropagation framework. We validate our method with state-of-the-art performance on challenging benchmark datasets: Street View Text, IIIT5k, ICDAR and Synth90k.     

* [#ICML 2016#【通过DNN把数据空间映射到latent的特征空间做聚类，目标函数是最小化软分配与辅助分布直接的KL距离，来迭代优化，思想类似于t-SNE，只不过这里使用了DNN】《Unsupervised Deep Embedding for Clustering Analysis》](https://arxiv.org/pdf/1511.06335.pdf)
> Clustering is central to many data-driven application domains and has been studied extensively in terms of distance functions and grouping algorithms.  Relatively little work has focused on learning  representations  for  clustering.   In  this paper,  we  propose  Deep  Embedded  Clustering (DEC), a method that simultaneously learns feature representations and cluster assignments using  deep  neural  networks.   DEC  learns  a  mapping from the data space to a lower-dimensional feature space in which it iteratively optimizes a
clustering  objective.   Our  experimental  evaluations on image and text corpora show significant improvement over state-of-the-art methods

* [SEE: Towards Semi-Supervised End-to-End Scene Text Recognition](https://github.com/Bartzi/see)

* [A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs](http://science.sciencemag.org/content/early/2017/10/26/science.aag2612.full)


* [EXTENDING THE PAGE SEGMENTATION ALGORITHMS OF THE OCROPUS DOCUMENTATION LAYOUT ANALYSIS SYSTEM](http://scholarworks.boisestate.edu/cgi/viewcontent.cgi?article=1122&context=td)


* [ Text Recognition in Scene Image and Video Frame using Color Channel Selection](https://arxiv.org/abs/1707.06810)
>In recent years, recognition of text from natural scene image and video frame has got increased attention among the researchers due to its various complexities and challenges. Because of low resolution, blurring effect, complex background, different fonts, color and variant alignment of text within images and video frames, etc., text recognition in such scenario is difficult. Most of the current approaches usually apply a binarization algorithm to convert them into binary images and next OCR is applied to get the recognition result. In this paper, we present a novel approach based on color channel selection for text recognition from scene images and video frames. In the approach, at first, a color channel is automatically selected and then selected color channel is considered for text recognition. Our text recognition framework is based on Hidden Markov Model (HMM) which uses Pyramidal Histogram of Oriented Gradient features extracted from selected color channel. From each sliding window of a color channel our color-channel selection approach analyzes the image properties from the sliding window and then a multi-label Support Vector Machine (SVM) classifier is applied to select the color channel that will provide the best recognition results in the sliding window. This color channel selection for each sliding window has been found to be more fruitful than considering a single color channel for the whole word image. Five different features have been analyzed for multi-label SVM based color channel selection where wavelet transform based feature outperforms others. Our framework has been tested on different publicly available scene/video text image datasets. For Devanagari script, we collected our own data dataset. The performances obtained from experimental results are encouraging and show the advantage of the proposed method. 

* [Scene Text Detection via Holistic, Multi-Channel Prediction](https://arxiv.org/pdf/1606.09002.pdf)
>Recently, scene text detection has become an active research topic in computer vision and document analysis, because of its great importance and significant challenge. However, vast majority of the existing methods detect text within local regions, typically through extracting character, word or line level candidates followed by candidate aggregation and false positive elimination, which potentially exclude the effect of wide-scope and long-range contextual cues in the scene. To take full advantage of the rich information available in the whole natural image, we propose to localize text in a holistic manner, by casting scene text detection as a semantic segmentation problem. The proposed algorithm directly runs on full images and produces global, pixel-wise prediction maps, in which detections are subsequently formed. To better make use of the properties of text, three types of information regarding text region, individual characters and their relationship are estimated, with a single Fully Convolutional Network (FCN) model. With such predictions of text properties, the proposed algorithm can simultaneously handle horizontal, multi-oriented and curved text in real-world natural images. The experiments on standard benchmarks, including ICDAR 2013, ICDAR 2015 and MSRA-TD500, demonstrate that the proposed algorithm substantially outperforms previous state-of-the-art approaches. Moreover, we report the first baseline result on the recently-released, large-scale dataset COCO-Text.

* [Joint Line Segmentation and Transcription for End-to-End Handwritten Paragraph Recognition]()

* [Scene Text Recognition with Sliding Convolutional Character Models]()

* [Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Network]()


## Blogs

* [Tesseract-OCR引擎入门](http://blog.csdn.net/xiaochunyong/article/details/7193744)             

* OCR引擎Ocropus实战指南              
* [博文1 Training an Ocropus OCR model ](http://www.danvk.org/2015/01/11/training-an-ocropus-ocr-model.html)          
* [博文2  Extracting text from an image using Ocropus](http://www.danvk.org/2015/01/09/extracting-text-from-an-image-using-ocropus.html)   
* [博文3 Working with Ground Truth ](https://github.com/tmbdev/ocropy/wiki/Working-with-Ground-Truth)                         
* [博文4 Finding blocks of text in an image using Python, OpenCV and numpy](http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html)                

* [Applying OCR Technology for Receipt Recognition]( http://rnd.azoft.com/applying-ocr-technology-receipt-recognition/ )         

* [Writing a Fuzzy Receipt Parser in Python](http://tech.trivago.com/2015/10/06/python_receipt_parser/)
* [Number plate recognition with Tensorflow](http://matthewearl.github.io/2016/05/06/cnn-anpr/)      
* [车牌识别中的不分割字符的端到端(End-to-End)识别](http://m.blog.csdn.net/article/details?id=52174198)         
* [端到端的OCR：基于CNN的实现](http://blog.xlvector.net/2016-05/mxnet-ocr-cnn/)
* [ 腾讯OCR—自动识别技术，探寻文字真实的容颜 ](http://dataunion.org/17291.html)    
>特征描述的完整过程 http://dataunion.org/wp-content/uploads/2015/05/640.webp_2.jpg

* [验证码识别](https://github.com/100steps/Blogs/issues/43)     

* [Bank check OCR with OpenCV and Python (Part I)](https://www.pyimagesearch.com/2017/07/24/bank-check-ocr-with-opencv-and-python-part-i/)

* [Common Sense, Cortex, and CAPTCHA](https://www.vicarious.com/2017/10/26/common-sense-cortex-and-captcha/)

## Presentations

* 学霸君archsubmit上的演讲提到了他们的ocr算法 使用cnn来识别中文  链接:http://pan.baidu.com/s/1kUoSyV1 密码: p92n



## Projects

* [ Project Naptha :highlight, copy, search, edit and translate text in any image](https://github.com/naptha)      


## Commercial products

* [ABBYY](http://www.baidu.com/link?url=p0G_qRjgD7aGhNS9gLT6i83as15p7aYyTY7xXM2dEKC&wd=&eqid=defb39aa000a84da000000065730a048)
```
作者：chenqin
链接：https://www.zhihu.com/question/19593313/answer/18795396
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1，识别率极高。我使用过现在的答案总结里提到的所有软件，但遇到下面这样的表格，除了ABBYY还能保持95%以上的识别率之外（包括秦皇岛三个字），其他所有的软件全部歇菜，数字认错也就罢了，中文也认不出。血泪的教训。
![](https://pic3.zhimg.com/a1b8009516c105556d2a2df319c72d72_b.jpg)
2，自由度高。可以在同一页面手动划分不同的区块，每一个区块也可以分别设置表格或文字；简体繁体英文数字。而此时大部分软件还只能对一个页面设置一种识别方案，要么表格，要么文字。
3，批量操作方便。对于版式雷同的年鉴，将一页的版式设计好，便可以应用到其他页，省去大量重复操作。
4，可以保持原有表格格式，省去二次编辑。跨页识别表格时，选择“识别为EXCEL”，ABBYY可以将表格连在一起，产出的是一整个excel文件，分析起来就方便多了。
5，包括梯形校正，歪斜校正之类的许多图片校正方式，即使扫描得歪了，或者因为书本太厚而导致靠近书脊的部分文字扭曲，都可以校正回来。
```

* IRIS               
```
 真正能把中文OCR做得比较专业的，一共也没几家，国内2家，国外2家。国内是文通和汉王，国外是ABBYY和IRIS（台湾原来有2家丹青和蒙恬，这两年没什么动静了）。像大家提到的紫光OCR、CAJViewer、MS Office、清华OCR、包括慧视小灵鼠，这些都是文通的产品或者使用文通的识别引擎，尚书则是汉王的产品，和中晶扫描仪捆绑销售的。这两家的中文识别率都是非常不错的。而国外的2家，主要特点是西方语言的识别率很好，而且支持多种西欧语言，产品化程度也很高，不过中文方面速度和识别率还是有差距的，当然这两年人家也是在不断进步。Google的开源项目，至少在中文方面，和这些家相比，各项性能指标水平差距还蛮大的呢。 

作者：张岩
链接：https://www.zhihu.com/question/19593313/answer/14199596
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

* 
```
https://github.com/cisocrgroup
```

* [OCR.space is a service of a9t9 software GmbH. The goal of OCR.space is to bring fresh ideas, methods and products to the OCR community.](https://ocr.space/)               
```
目前看到最棒的免费的API  当然也提供商业版
```

## OCR Databases



## OTHERS

* [pics for testing 测试用图片 ]() 链接: http://pan.baidu.com/s/1jGNIjPG 密码: 3izf


## Discussion and Feedback

欢迎扫码加入 参与讨论分享 过期请添加个人微信 edwin_whs
![](https://user-images.githubusercontent.com/2363295/34026309-0d1f4e8e-e190-11e7-8415-c73405cb25e7.jpeg){:height="50%" width="50%"}

             


