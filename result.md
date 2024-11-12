---
layout: page
---

## Evaluation metrics
We evaluate the performances of the models using the following metrics, Accuracy (Acc), Sensitivity (Sens), Specificity (Spec), F1, Area Under the Curve (AUC) 

### Ranking of singletask approaches
The singletask approaches are trained on the benchmark dataset with 3,641 BUS images. Click on a metric to sort approaches based on that metric.

|Rank|Approaches|Acc|Sens|Spec|F1|AUC|
|--- |--- |--- |--- |--- |--- |--- |
|1|[VGG16](https://arxiv.org/abs/1409.1556)|74.5|86.7|62.6|0.77|74.7|
|2|[MobileNet](https://arxiv.org/abs/1704.04861?context=cs)|74.0|87.4|61.3|0.77|74.4|
|3|[Xception](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)|73.7|88.5|59.6|0.77|74.0|
|4|[EfficientNetB0](http://proceedings.mlr.press/v97/tan19a.html)|73.8|86.8|61.2|0.77|74.0|
|5|[InceptionV3](https://arxiv.org/abs/1512.00567)|73.0|88.4|57.6|0.77|73.0|
|6|[DenseNet121](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)|72.7|90.1|55.7|0.77|72.9|
|7|[Tanaka](https://pubmed.ncbi.nlm.nih.gov/31645021/)|77.8|74.6|81.2|0.76|77.9|
|8|[ResNet50](https://arxiv.org/abs/1512.03385v1)|72.6|86.2|59.4|0.76|72.8|
|9|[Shia](https://pubmed.ncbi.nlm.nih.gov/33302247/)|74.6|75.5|74.0|0.75|74.7|
|10|[Xie](https://iopscience.iop.org/article/10.1088/1361-6560/abc5c7/meta)|62.2|48.6|75.8|0.55|62.2|


### Ranking of multitask approaches
The multitask approaches are trained on BUSI and BUSIS datasets combined with 1,209 BUS images. 

|Rank|Approaches|Acc|Sens|Spec|F1|AUC|
|--- |--- |--- |--- |--- |--- |--- |
|1|[MT-ESTAN (ours)]()|90.0|90.4|89.8|0.88|90.1|
|2|[MobileNet](https://arxiv.org/abs/1704.04861?context=cs)|87.0|81.1|91.0|0.83|86.1|
|3|[VGG16](https://arxiv.org/abs/1409.1556)|87.1|81.3|90.9|0.83|86.1|
|4|[EfficientNetB0](http://proceedings.mlr.press/v97/tan19a.html)|87.5|81.0|91.2|0.83|86.1|
|5|[Zhang](https://pubmed.ncbi.nlm.nih.gov/34254225/)|87.4|81.4|91.4|0.83|86.4|
|6|[ResNet50](https://arxiv.org/abs/1512.03385v1)|86.1|80.9|89.2|0.81|85.0|
|7|[DenseNet121](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)|85.0|79.1|88.9|0.80|84.0|
|8|[Shi](https://rc.signalprocessingsociety.org/conferences/isbi-2022/SPSISBI22VID0276.html?source=IBP)|83.9|87.3|81.7|0.80|84.5|
|9|[Vakanski](https://ieeexplore.ieee.org/abstract/document/9596501)|83.6|77.4|87.8|0.78|82.6|
