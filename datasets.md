---
layout: page
---
## <left><span style="color:Black">Benchmark Dataset </span></left>
In the benchmark, we used 3,614 breast ultrasound images from five public datasets.  To prepare the benchmark, 
please obtain permission of the original owners of each dataset, follow their copyright and usage 
policies, and cite and acknowledge their great efforts in your research. 
Any redistribution and commercial use of the benchmark are prohibited. 
 

<style>
      table,
      th,
      td {
        padding: 0px;
        border: 1px solid black;
        border-collapse: collapse;
      }
    </style>

<table id=""  class="display">
  <thead>
    <tr>
      <th style="text-align: left">Datasets</th>
      <th style="text-align: center">Total BUS Images</th>
	  <th style="text-align: center">Class Distribution</th>
		<th style="text-align: center">Location</th>
		<th style="text-align: center">Devices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><a href ="https://www.ultrasoundcases.info/"> HMSS </a></td>
      <td style="text-align: center">2,006</td>
	  <td style="text-align: center">Benign: 846   Malignant: 1,160</td>
	  <td style="text-align: center">Netherlands/Europe</td>
	  <td style="text-align: center">Fujifilm  Ultrasound</td>
	  
    </tr>

    <tr>
      <td style="text-align: center"><a href ="https://scholar.cu.edu.eg/?q=afahmy/pages/dataset"> BUSI</a></td>
      <td style="text-align: center">647</td>
	  <td style="text-align: center">Benign: 437    Malignant: 210</td>
		<td style="text-align: center">Egypt/Africa</td>
<td style="text-align: center">  LOGIQ E9, LOGIQ E9 Agile </td>
    </tr>

    <tr>
      <td style="text-align: center"><a href ="http://cvprip.cs.usu.edu/busbench/"> BUSIS</a></td>
      <td style="text-align: center">562</td>
	  <td style="text-align: center">Benign: 306   Malignant: 256</td>
<td style="text-align: center">China/Asia</td>
<td style="text-align: center"> GE VIVID 7, LOGIQ E9, Hitachi EUB-6500, Philips iU22, Siemens ACUSON S2000</td>
    </tr>

    <tr>
      <td style="text-align: center"><a href ="http://www.onlinemedicalimages.com/index.php/en/81-site-info/73-introduction"> Thammasat </a></td>
      <td style="text-align: center">263</td>
	  <td style="text-align: center">Benign: 120   Malignant: 143</td>
<td style="text-align: center">Thailand/Asia</td>
<td style="text-align: center">Samsung RS80A, Philips iU22</td>
    </tr>

    <tr>
      <td style="text-align: center"><a href ="http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php"> Dataset B</a></td>
      <td style="text-align: center">163</td>
	  <td style="text-align: center">Benign: 109   Malignant: 54</td>
<td style="text-align: center">Spain/Europe</td>
<td style="text-align: center">Siemens ACUSON  Sequoia C512</td>
    </tr>

    <tr>
      <td style="text-align: center" colspan=2>Total # of images:</td>
      <td style="text-align: center" colspan=1>3,641</td>
    </tr>
  </tbody>
</table>

<h2 id="Lorem_Ipsum"> Dataset preparation</h2>
To generate the *benchmark* dataset, we first download and unzip the ["Benchmark"](http://bus.midalab.net/filesharing/download/7e54bb0d-b328-42cf-83b6-61d29846dc6e) folder. The Benchmark folder contains all the preprocessing code, and "Datasets" 
folders, where later we introduce all the files and folders. 

<ol>
<li> Download the five datasets individually and follow the instructions:</li> 

	<ul><li> <b>  <font size="+1">HMSS dataset: </font></b></li>
	<ul><li>The raw images of HMSS dataset can be found at: <a href=" https://www.ultrasoundcases.info/">https://www.ultrasoundcases.info/ </a> </li>
	
	<li> To download the HMSS dataset, please go to "Benchmark" folder and run the "Download_HMSS.ipynb" script.</li>

	<li> The HMSS images are saved under the 'Benchmark/Datasets/HMSS/imgs/' folder, and the HMSS.csv file under 'Benchmark/Datasets/HMSS/'</li> </ul></ul>
  	 
	 <left><span style="color:Red">Note:</span> 
		<i> The HMSS website contains images of different organs and cancer types with different modalities. 
		We have used only 2,006 HMSS breast ultrasound images. </i>
				</left>
	<br>	<br>

	
<ul><li> <b> <font size="+1">Thammasat dataset:</font></b></li><ul><li>The raw images of Thammasat dataset can be found at: <a href=""> http://www.onlinemedicalimages.com/ </a></li>
<li> To download the Thammasat dataset, please go to "Benchmark" folder and run the "Download_Thammasat.ipynb" script.</li> 

<li> The Thammasat images are saved under the 'Benchmark/Datasets/Thammasat/imgs/' folder, 
and the Thammasat.csv file under 'Benchmark/Benchmark/Datasets/Thammasat/' </li></ul></ul>
	
		<left><span style="color:Red">Note:</span> 
		<i> We use only bus images from Thammasat website and we do not download the elastography, nor the colored images. </i>
				</left>	<br>	<br>

	<ul> <li><b> <font size="+1">BUSIS dataset:</font></b> </li> <ul> <li>To download the BUSIS dataset, please go to this website:
 <a href ="http://cvprip.cs.usu.edu/busbench/index.html">http://cvprip.cs.usu.edu/busbench/index.html</a>, and fill in the Licence Agreement
  and send to <a href="mailto:hengda.cheng@usu.edu">hengda.cheng@usu.edu</a> or <a href="mailto:mxian@uidaho.edu">mxian@uidaho.edu</a>. Once approved they will 
  send access information for you to download the dataset.</li>
<li> Put all raw ultrasound images under the 'Benchmark/Datasets/BUSIS/imgs/' folder, 
the sementation ground truths under the 'Benchmark/Datasets/BUSIS/GT/' 
folder, and the BUSIS.xlsx file under 'Benchmark/Datasets/BUSIS/' </li> </ul></ul>
		

	<ul> <li><b> <font size="+1">Dataset B:</font></b> </li><ul><li>To download the Dataset B, please go to this website:
<a href=" http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php"> http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php </a>, and fill in the Licence 
 Agreement and send to <a href="mailto:M.Yap@mmu.ac.uk">M.Yap@mmu.ac.uk</a> or <a href="mailto:robert.marti@udg.edu">robert.marti@udg.edu</a>.
  Once approved they will send access information for you to download the dataset.</li>
<li> Put all raw ultrasound images under the 'Benchmark/Datasets/DatasetB/imgs/' folder, 
the sementation ground truths under the 'Benchmark/Datasets/DatasetB/GT/' folder, 
and the DatasetB.xlsx file under 'Benchmark/Benchmark/Datasets/DatasetB/'</li> </ul>
</ul>
	<ul> <li> <b> <font size="+1">BUSI dataset:</font></b> </li><ul><li>To download the BUSI dataset, please go to this website:
 <a href="https://scholar.cu.edu.eg/?q=afahmy/pages/dataset"> https://scholar.cu.edu.eg/?q=afahmy/pages/dataset </a>, and follow the instructions.
</li>
<li> Unzip the BUSI dataset and put "benign" and "malignant" folders under the 'Benchmark/Datasets/BUSI/imgs/' folder.</li> </ul></ul>

<span style="color:Red">Note:</span> 
		<i> The BUSI dataset comprises of "benign", "malignant", and "normal" bus images. In this study, we did not use the normal images.</i>
				<br><br>

	<li> Generate the benchmark dataset</li> 
<ul><li> To generate the benchmark dataset and perform preprocessing on the images, run "Preprocess_All.ipynb" script, 
please note that prior to running this file, you must have completed the <a href="#Lorem_Ipsum">step 1.</a></li> </ul>
	<left><span style="color:Red">Note:</span> </left>
	In this step we generate two datasets, the benchmark dataset for singletask approaches which consists of 3,641 images,
	and a multitask dataset for multitask training which combines the BUSI and BUSIS datasets with total 2,009 images.
Please refer to the "readme.txt" files at each dataset's folder for instructions to prepare the benchmark dataset.<br><br>
<li> Predefined folds. To split the dataset based on "Cases" and build 5-fold train, test, and validation, run the "Predefined folds.ipynb" script</li>
<left><span style="color:Red">Note:</span> <ul>
<li> To prevent data bias and leakage, we conduct our experiments based on <b>BUS cases</b>, the images of the same case go to either "train", "test", or "valid" sets, and we do not accept results based on random splitting </li>
	</ul></left>
<li> To reproduce our published results, <a href="download/folds.zip" download>click here</a>  to download our predefined 5-fold train, test, and validation indices,   </li>
	
</ol>

