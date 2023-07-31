---
layout: page
---
## Evaluation metrics
We evaluate the performances of the models using the following metrics, Accuracy (Acc), Sensitivity (Sens), Specificity (Spec), F1, Area Under the Curve (AUC) 

### Ranking of singletask approaches
The singletask approaches are trained on the benchmark dataset with 3,641 BUS images. Click on a metric to sort approaches based on that metric.

<style>
      table,
      th,
      td {
        padding: 0px;
        border: 1px solid black;
        border-collapse: collapse;
      }
    </style>
<table id="" name= table class="display">
  <thead>
      
    <tr>
      <th>Rank</th>
      <th>Approaches</th>
      <th>Acc</th>
      <th>Sens</th>
      <th>Spec</th>
      <th>F1</th>
      <th>AUC</th>

    </tr>
  </thead>
  <tbody>
     <tr>
	      <td style="text-align:center">1</td>
      <td style="text-align:center" ><a href = "https://arxiv.org/abs/1409.1556"> VGG16 </a></td>

        <td style="text-align:center">74.5</td>
        <td style="text-align:center">86.7</td>
        <td style="text-align:center">62.6</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">74.7 </td>
    </tr>
    <tr>

	      <td style="text-align:center">2</td>
      <td style="text-align:center" ><a href = "https://arxiv.org/abs/1704.04861?context=cs">MobileNet</a></td>

        <td style="text-align:center">74.0</td>
        <td style="text-align:center">87.4</td>
        <td style="text-align:center">61.3</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">74.4 </td>
    </tr>
	    <tr>
	      <td style="text-align:center">3</td>
      <td style="text-align:center" ><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html">Xception</a></td>

        <td style="text-align:center">73.7</td>
        <td style="text-align:center">88.5</td>
        <td style="text-align:center">59.6</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">74.0 </td>
    </tr>	    <tr>

	      <td style="text-align:center">4</td>
      <td style="text-align:center" ><a href = "http://proceedings.mlr.press/v97/tan19a.html">EfficientNetB0</a></td>

        <td style="text-align:center">73.8</td>
        <td style="text-align:center">86.8</td>
        <td style="text-align:center">61.2</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">74.0 </td>
    </tr>
	    <tr>
	  <td style="text-align:center">5</td>
      <td style="text-align:center" ><a href= "https://arxiv.org/abs/1512.00567"> InceptionV3</a></td>
    
        <td style="text-align:center">73.0</td>
        <td style="text-align:center">88.4</td>
        <td style="text-align:center">57.6</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">73.0 </td>
    </tr>	    <tr>
      <td style="text-align:center">6</td>
      <td style="text-align:center" ><a href ="https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf">DenseNet121</a></td>
	    
        <td style="text-align:center">72.7</td>
        <td style="text-align:center"><b>90.1</b></td>
        <td style="text-align:center">55.7</td>
        <td style="text-align:center">0.77</td>
        <td style="text-align:center">72.9 </td>
    </tr>

    <tr>
	      <td style="text-align:center">7</td>
      <td style="text-align:center"><a href="https://pubmed.ncbi.nlm.nih.gov/31645021/"> Tanaka</a></td>

        <td style="text-align:center"><b>77.8</b></td>
        <td style="text-align:center">74.6</td>
        <td style="text-align:center"><b>81.2</b></td>
        <td style="text-align:center">0.76</td>
        <td style="text-align:center"><b>77.9 </b></td>
    </tr>




    <tr>
	      <td style="text-align:center">8</td>
      <td style="text-align:center" ><a href="https://arxiv.org/abs/1512.03385v1"> ResNet50</a></td>

        <td style="text-align:center">72.6</td>
        <td style="text-align:center">86.2</td>
        <td style="text-align:center">59.4</td>
        <td style="text-align:center">0.76</td>
        <td style="text-align:center">72.8 </td>
    </tr>

 







    <tr>
	      <td style="text-align:center">9</td>
		        <td style="text-align:center"><a href = "https://pubmed.ncbi.nlm.nih.gov/33302247/">Shia</a></td>


        <td style="text-align:center">74.6</td>
        <td style="text-align:center">75.5</td>
        <td style="text-align:center">74.0</td>
        <td style="text-align:center">0.75</td>
        <td style="text-align:center">74.7 </td>
    </tr>


    <tr>
	      <td style="text-align:center">10</td>

     <td style="text-align:center" ><a href = "https://iopscience.iop.org/article/10.1088/1361-6560/abc5c7/meta"> Xie</a></td>

        <td style="text-align:center">62.2</td>
        <td style="text-align:center">48.6</td>
        <td style="text-align:center">75.8</td>
        <td style="text-align:center">0.55</td>
        <td style="text-align:center">62.2 </td>
    </tr>
	
  </tbody>
</table>


### Ranking of multitask approaches
The multitask approaches are trained on BUSI and BUSIS datasets combined with 1,209 BUS images. 

<style>
      table,
      th,
      td {
        padding: 0px;
        border: 1px solid black;
        border-collapse: collapse;
      }
    </style>
<table id="" name= table class="display">
  <thead>
      
    <tr>
      <th>Rank</th>
      <th>Approaches</th>
      <th>Acc</th>
      <th>Sens</th>
      <th>Spec</th>
      <th>F1</th>
      <th>AUC</th>

    </tr>
  </thead>
  <tbody>
     <tr>      <td style="text-align:center">1</td>
      <td style="text-align:center" ><a href = "">MT-ESTAN (ours)</a></td>
        <td style="text-align:center">90.0</td>
        <td style="text-align:center">90.4</td>
        <td style="text-align:center">89.8</td>
        <td style="text-align:center">0.88</td>
        <td style="text-align:center">90.1</td>
    </tr>
    <tr>      <td style="text-align:center">2</td>
      <td style="text-align:center" ><a href= "https://arxiv.org/abs/1704.04861?context=cs"> MobileNet</a></td>
        <td style="text-align:center">87.0</td>
        <td style="text-align:center">81.1</td>
        <td style="text-align:center">91.0</td>
        <td style="text-align:center">0.83</td>
        <td style="text-align:center">86.1 </td>
    </tr>
	    <tr>      <td style="text-align:center">3</td>
      <td style="text-align:center" ><a href="https://arxiv.org/abs/1409.1556"> VGG16</a></td>
        <td style="text-align:center">87.1</td>
        <td style="text-align:center">81.3</td>
        <td style="text-align:center">90.9</td>
        <td style="text-align:center">0.83</td>
        <td style="text-align:center">86.1 </td>
    </tr>
	    <tr>      <td style="text-align:center">4</td>
      <td style="text-align:center"><a href = "http://proceedings.mlr.press/v97/tan19a.html">EfficientNetB0</a></td>
        <td style="text-align:center">87.5</td>
        <td style="text-align:center">81.0</td>
        <td style="text-align:center">91.2</td>
        <td style="text-align:center">0.83</td>
        <td style="text-align:center">86.1 </td>
    </tr>
	   <tr>      <td style="text-align:center">5</td>
      <td style="text-align:center"><a href="https://pubmed.ncbi.nlm.nih.gov/34254225/">Zhang</a></td>
        <td style="text-align:center">87.4</td>
        <td style="text-align:center">81.4</td>
        <td style="text-align:center">91.4</td>
        <td style="text-align:center">0.83</td>
        <td style="text-align:center">86.4</td>
    </tr>
	    <tr>      <td style="text-align:center">6</td>
      <td style="text-align:center" ><a href = "https://arxiv.org/abs/1512.03385v1">ResNet50</a></td>
        <td style="text-align:center">86.1</td>
        <td style="text-align:center">80.9</td>
        <td style="text-align:center">89.2</td>
        <td style="text-align:center">0.81</td>
        <td style="text-align:center">85.0 </td>
    </tr>
	      <tr>      <td style="text-align:center">7</td>
      <td style="text-align:center" ><a href ="https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf">DenseNet121</a></td>
        <td style="text-align:center">85.0</td>
        <td style="text-align:center">79.1</td>
        <td style="text-align:center">88.9</td>
        <td style="text-align:center">0.80</td>
        <td style="text-align:center">84.0 </td>
    </tr>

    <tr>
	      <td style="text-align:center">8</td>
      <td style="text-align:center"><a href="https://rc.signalprocessingsociety.org/conferences/isbi-2022/SPSISBI22VID0276.html?source=IBP">Shi</a></td>
        <td style="text-align:center">83.9</td>
        <td style="text-align:center">87.3</td>
        <td style="text-align:center">81.7</td>
        <td style="text-align:center">0.80</td>
        <td style="text-align:center">84.5</td>
    </tr>


 
    <tr >
	      <td style="text-align:center">9</td>
      <td style="text-align:center" ><a href = "https://ieeexplore.ieee.org/abstract/document/9596501">Vakanski</a></td>
        <td style="text-align:center">83.6</td>
        <td style="text-align:center">77.4</td>
        <td style="text-align:center">87.8</td>
        <td style="text-align:center">0.78</td>
        <td style="text-align:center">82.6</td>
    </tr>




  </tbody>
</table>
