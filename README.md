# <font size=6> DivAugGAN </font>

<img src='images/afhq-transfer.png' width="960px" align="middle">   <br><br>
 
<img src='images/regularization-framework.png' width="960px" align="middle">   <br><br>

<br><br><br>

We provide our PyTorch implementation of unpaired image-to-image translation based on

## <font size=5> Example Results </font>

### <font size=4> Two-domain paired image-to-image translation </font>
<font size=3>  Aeriel  &rarr;  Map  </font> <br>
<img src="images/comparison-results/PI2I-01-maps/148.png" width="800px"/>  <br><br>

<img src="images/comparison-results/PI2I-01-maps/154.png" width="800px"/>  <br><br>

<img src="images/comparison-results/PI2I-01-maps/246.png" width="800px"/>  <br><br>

<!-- ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/cat2dog%26summer2winter.png)  -->

<br>

### <font size=4> Two-domain unpaired image-to-image translation </font>

<font size=3>  Cat  &rarr;  Dog </font> <br><br>
<img src="images/comparison-results/UI2I-01-cat2dog/cat2dog-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-01-cat2dog/cat2dog-2.png" width="800px"/>  <br><br>


<font size=3>  Dog  &rarr; Cat  </font> <br><br>

<img src="images/comparison-results/UI2I-02-dog2cat/dog2cat-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-02-dog2cat/dog2cat-2.png" width="800px"/>  <br><br>


<font size=3>  Monet  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/UI2I-03-monet2photo/00030.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-03-monet2photo/00150.png" width="800px"/>  <br><br>


<font size=3>  Photo  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-03-09:47:19.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-06-19:34:34.png" width="800px"/>  <br><br>


<font size=3>  Photo  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-03-09:47:19.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-06-19:34:34.png" width="800px"/>  <br><br>


<font size=3>  Photo  &rarr; Portrait  </font> <br><br>

<img src="images/comparison-results/UI2I-05-photo2portrait/photo2portrait-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-05-photo2portrait/photo2portrait-2.png" width="800px"/>  <br><br>


<font size=3>  Portrait  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/UI2I-06-portrait2photo/portrait2photo-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-06-portrait2photo/portrait2photo-2.png" width="800px"/>  <br><br>

<font size=3>  Summer  &rarr; Winter  </font> <br><br>

<img src="images/comparison-results/UI2I-07-summer2winter/summer2winter-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-07-summer2winter/summer2winter-2.png" width="800px"/>  <br><br>

<font size=3>  Winter  &rarr; Summer  </font> <br><br>

<img src="images/comparison-results/UI2I-08-winter2summer/winter2summer-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-08-winter2summer/winter2summer-2.png" width="800px"/>  <br><br>



### <font size=4>  multi-domains image translation </font>
<!--  ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/afhq-transfer.png)  -->
<!--   ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/image-weather-conditions.png)  -->


<font size=3>  Alps seasonal transfer </font> <br><br>




<font size=3>  Arts </font> <br><br>


<font size=3>  AFHQ </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-afhq/afhq.png" width="960px"/>  <br><br>




## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN