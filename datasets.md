---
layout: page
---

## Benchmark Dataset
In the benchmark, we used 3,614 breast ultrasound images from five public datasets.  To prepare the benchmark, please obtain permission of the original owners of each dataset, follow their copyright and usage policies, and cite and acknowledge their great efforts in your research. Any redistribution and commercial use of the benchmark are prohibited. 

| Datasets           | Total BUS Images | Class Distribution             | Location           | Devices                                                                    | 
|--------------------|------------------|--------------------------------|--------------------|----------------------------------------------------------------------------|
|  [HMSS](https://www.ultrasoundcases.info/)              | 2,006            | Benign: 846   Malignant: 1,160 | Netherlands/Europe | Fujifilm  Ultrasound                                                       | 
|  [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)              | 647              | Benign: 437    Malignant: 210  | Egypt/Africa       | LOGIQ E9, LOGIQ E9 Agile                                                   |
|  [BUSIS](http://cvprip.cs.usu.edu/busbench/)             | 562              | Benign: 306   Malignant: 256   | China/Asia         | GE VIVID 7, LOGIQ E9, Hitachi EUB-6500, Philips iU22, Siemens ACUSON S2000 |
|  [Thammasat](http://www.onlinemedicalimages.com/index.php/en/81-site-info/73-introduction)         | 263              | Benign: 120   Malignant: 143   | Thailand/Asia      | Samsung RS80A, Philips iU22                                                | 
|  [Dataset B](http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php)         | 163              | Benign: 109   Malignant: 54    | Spain/Europe       | Siemens ACUSON  Sequoia C512                                               | 
| Total # of images: | 3,641            |                                |                    |                                                                            | 


## Dataset preparation
To generate the *benchmark* dataset, we first download and unzip the ["Benchmark"](http://bus.midalab.net/filesharing/download/7e54bb0d-b328-42cf-83b6-61d29846dc6e) folder. The Benchmark folder contains all the preprocessing code, and "Datasets" 
folders, where later we introduce all the files and folders. 

1. Download the five datasets individually and follow the instructions:
   - **HMSS dataset**:
     - The raw images of HMSS dataset can be found at: [https://www.ultrasoundcases.info/](https://www.ultrasoundcases.info/)
     - To download the HMSS dataset, please go to "Benchmark" folder and run the "Download_HMSS.ipynb" script.
     - The HMSS images are saved under the 'Benchmark/Datasets/HMSS/imgs/' folder, and the HMSS.csv file under 'Benchmark/Datasets/HMSS/'
     - *<span style="color: red">Note</span>: The HMSS website contains images of different organs and cancer types with different modalities. We have used only 2,006 HMSS breast ultrasound images.*
  
   - **Thammasat dataset**:
     - The raw images of Thammasat dataset can be found at: [http://www.onlinemedicalimages.com/](http://www.onlinemedicalimages.com/)
     - To download the Thammasat dataset, please go to "Benchmark" folder and run the "Download_Thammasat.ipynb" script.
     - The Thammasat images are saved under the 'Benchmark/Datasets/Thammasat/imgs/' folder, and the Thammasat.csv file under 'Benchmark/Benchmark/Datasets/Thammasat/' 
     - *<span style="color: red">Note</span>: We use only bus images from Thammasat website and we do not download the elastography, nor the colored images.*
  
   - **BUSIS dataset**:
     - To download the BUSIS dataset, please go to this website: [http://cvprip.cs.usu.edu/busbench/index.html](http://cvprip.cs.usu.edu/busbench/index.html), and fill in the Licence Agreement and send to [hengda.cheng@usu.edu](mailto:hengda.cheng@usu.edu) or [mxian@uidaho.edu](mailto:mxian@uidaho). Once approved they will send access information for you to download the dataset.
     - Put all raw ultrasound images under the 'Benchmark/Datasets/BUSIS/imgs/' folder, the sementation ground truths under the 'Benchmark/Datasets/BUSIS/GT/' folder, and the BUSIS.xlsx file under 'Benchmark/Datasets/BUSIS/'
   - **Dataset B**:
     - To download the Dataset B, please go to this website: [http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php](http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php), and fill in the Licence Agreement and send to [M.Yap@mmu.ac.uk](mailto:M.Yap@mmu.ac.uk) or [robert.marti@udg.edu](mailto:robert.marti@udg.edu). Once approved they will send access information for you to download the dataset.
     - Put all raw ultrasound images under the 'Benchmark/Datasets/DatasetB/imgs/' folder, the sementation ground truths under the 'Benchmark/Datasets/DatasetB/GT/' folder, and the DatasetB.xlsx file under 'Benchmark/Benchmark/Datasets/DatasetB/'
   - **BUSI dataset**:
     - To download the BUSI dataset, please go to this website: [https://scholar.cu.edu.eg/?q=afahmy/pages/dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), and follow the instructions.
     - Unzip the BUSI dataset and put "benign" and "malignant" folders under the 'Benchmark/Datasets/BUSI/imgs/' folder.
     - *<span style="color: red">Note</span>: The BUSI dataset comprises of "benign", "malignant", and "normal" bus images. In this study, we did not use the normal images.*
2. Generate the benchmark dataset
   - To generate the benchmark dataset and perform preprocessing on the images, run "Preprocess_All.ipynb" script, please note that prior to running this file, you must have completed the step 1.
   - *<span style="color: red">Note</span>*: In this step we generate two datasets, the benchmark dataset for singletask approaches which consists of 3,641 images, and a multitask dataset for multitask training which combines the BUSI and BUSIS datasets with total 2,009 images. Please refer to the "readme.txt" files at each dataset's folder for instructions to prepare the benchmark dataset.
3. Predefined folds. To split the dataset based on "Cases" and build 5-fold train, test, and validation, run the "Predefined folds.ipynb" script
   - *<span style="color: red">Note</span>*: To prevent data bias and leakage, we conduct our experiments based on **BUS cases**, the images of the same case go to either "train", "test", or "valid" sets, and we do not accept results based on random splitting.
4. To reproduce our published results, [click here]({{ site.baseurl }}/download/folds.zip) to download our predefined 5-fold train, test, and validation indices.
