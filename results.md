---
layout: page
---
## <left><span style="color:Black"> Single-Task and Multi-Task results </span></left>

### <left><span style="color:Black"> Ranking of Multi-Task and Single-Task approaches </span></left>
The Multi-task approaches are trained on BUSI and BUSIS datasets combined with 1192 BUS images and Single-task approaches are trained on benchmark dataset with 3600 BUS images. Click on a metric to sort approaches based on that metric
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
  <thead><tr><th></th><th></th>
      <th colspan="5" >Classification </th>
      <th colspan="3" >Segmentation </th></tr>
    <tr>
      <th>Rank</th>
      <th>Classifers</th>
      <th>Acc</th>
      <th>Sens</th>
      <th>Spec</th>
      <th>F1</th>
      <th>AUC</th>
	  <th>Dice</th>
      <th>IOU</th>
      <th>Sens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><a href ="https://ieeexplore.ieee.org/document/8099726">DenseNet121_Multi</a></td>
      <td>0.85</td>
      <td>0.88</td>
      <td>0.83</td>
      <td>0.82</td>
      <td>0.86</td>
	  <td>0.83</td>
      <td>0.82</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>2</td>
      <td><a href= "https://pubmed.ncbi.nlm.nih.gov/34254225/"> SHA-MTL_Multi</a></td>
      <td>0.87</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
	  <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>3</td>
      <td><a href = "http://proceedings.mlr.press/v97/tan19a.html">EfficientNetB0_Multi</a></td>
      <td>0.87</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
	  <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>4</td>
      <td><a href="https://arxiv.org/abs/1704.04861v1"> MobileNet_Multi</a></td>
      <td>0.87</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
	  <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>5</td>
      <td><a href = "https://arxiv.org/abs/1409.1556">VGG16_Multi</a></td>
      <td>0.87</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
	        <td>0.91</td>
      <td>0.83</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>6</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0169260719307059?via%3Dihub">Moon_Multi</a></td>
      <td>0.87</td>
      <td>0.78</td>
      <td>0.93</td>
      <td>0.82</td>
      <td>0.85</td>
	  <td>0.93</td>
      <td>0.82</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>7</td>
      <td><a href = "https://ieeexplore.ieee.org/document/7780459">ResNet50_Multi</a></td>
      <td>0.86</td>
      <td>0.8</td>
      <td>0.89</td>
      <td>0.81</td>
      <td>0.85</td>
	  <td>0.89</td>
      <td>0.81</td>
      <td>0.85</td>
    </tr>
	 <tr>
      <td>8</td>
      <td><a href="https://arxiv.org/abs/1704.04861v1"> MobileNet_Single</a></td>
      <td>0.81</td>
      <td>0.85</td>
      <td>0.78</td>
      <td>0.81</td>
      <td>0.81</td>
	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>9</td>
      <td><a href = "https://arxiv.org/abs/1512.00567"> InceptionV3_Single</a></td>
      <td>0.82</td>
      <td>0.82</td>
      <td>0.82</td>
      <td>0.82</td>
      <td>0.82</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>10</td>
      <td><a href = "https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf">Xception_Single</a></td>
      <td>0.80</td>
      <td>0.84</td>
      <td>0.77</td>
      <td>0.81</td>
      <td>0.80</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>11</td>
      <td><a href = "https://ieeexplore.ieee.org/document/7780459">ResNet50_Single</a></td>
      <td>0.80</td>
      <td>0.81</td>
      <td>0.80</td>
      <td>0.80</td>
      <td>0.80</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>12</td>
      <td><a href = "https://arxiv.org/abs/1409.1556">VGG16_Single</a></td>
      <td>0.79</td>
      <td>0.79</td>
      <td>0.80</td>
      <td>0.79</td>
      <td>0.79</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>13</td>
      <td><a href ="https://ieeexplore.ieee.org/document/8099726">DenseNet121_Single</a></td>
      <td>0.81</td>
      <td>0.81</td>
      <td>0.82</td>
      <td>0.81</td>
      <td>0.81</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>14</td>
      <td><a href = "http://proceedings.mlr.press/v97/tan19a.html">EfficientNet_Single</a></td>
      <td>0.79</td>
      <td>0.82</td>
      <td>0.76</td>
      <td>0.79</td>
      <td>0.79</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>15</td>
      <td><a href ="https://pubmed.ncbi.nlm.nih.gov/31645021/">Tanaka_Single</a></td>
      <td>0.77</td>
      <td>0.74</td>
      <td>0.81</td>
      <td>0.76</td>
      <td>0.77</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>16</td>
      <td><a href = "https://www.sciencedirect.com/science/article/pii/S0895611120301245?via%3Dihub">Shia_Single</a></td>
      <td>0.76</td>
      <td>0.74</td>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>17</td>
      <td><a href = "https://pubmed.ncbi.nlm.nih.gov/33120380/">DSCNN_Single </a></td>
      <td>0.64</td>
      <td>0.58</td>
      <td>0.71</td>
      <td>0.61</td>
      <td>0.64</td>
	  	  <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
