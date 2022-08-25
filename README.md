# <font size=6> DivAugGAN </font>

<img src='images/afhq-transfer.png' width="960px" align="middle">   <br><br>
 
<img src='images/regularization-framework.png' width="960px" align="middle">   <br><br>

<br><br><br>

We provide our PyTorch implementation of DivAugGAN for multimodal image-to-image translation. DivAugGAN functions as a regularizer to simultaneously maximize the distinction of the generating samples and maintain the relative variation consistency in the translation process as well.　<br>　

## <font size=5> Example Results </font>

### <font size=4> Two-domain paired image-to-image translation   </font>
<font size=3>  **Aeriel  &rarr;  Map**  </font> <br><br>

<img src="images/comparison-results/PI2I-01-maps/148.png" width="800px"/>  <br><br>

<img src="images/comparison-results/PI2I-01-maps/154.png" width="800px"/>  <br><br>

<img src="images/comparison-results/PI2I-01-maps/246.png" width="800px"/>  <br><br>

<!-- ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/cat2dog%26summer2winter.png)  -->

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with *vanilla* BicycleGAN (first row), MSGAN (second row), DSGAN (third row), and DivCoBicycleGAN (fourth row) on  **Aeriel &rarr;  Maps**  dataset. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this paired multimodal image-to-image translation. DivAugGAN generates images with superior contents preservation and adequate variation (diverse color and light) with different input latent vectors. </font> <br><br>

### <font size=4> Two-domain unpaired image-to-image translation qualitative comparisons </font>

<font size=3>  **Cat  &rarr;  Dog**  </font> <br><br>

<img src="images/comparison-results/UI2I-01-cat2dog/cat2dog-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-01-cat2dog/cat2dog-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Cat  &rarr;  Dog**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>

<font size=3>  **Dog  &rarr; Cat**  </font> <br>

<img src="images/comparison-results/UI2I-02-dog2cat/dog2cat-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-02-dog2cat/dog2cat-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Dog  &rarr; Cat**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>

<font size=3>  **Monet  &rarr; Photo**  </font> <br><br>

<img src="images/comparison-results/UI2I-03-monet2photo/00030.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-03-monet2photo/00150.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Monet  &rarr; Photo**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>

<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Monet  &rarr; Photo**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   78.73    |    0.197     |       0.664        |     0.028      |       0.555      |      0.162      |
| MSGAN     |   **68.21**    |    **0.280**     |       <u>0.712</u>        |     **0.062**      |       **0.754**      |      **0.255**      |
| DSGAN     |   79.45    |    0.182     |       0.576        |     0.025      |       0.429      |      0.145      |
| DivCo     |   72.44    |    0.244     |       **0.726**        |     0.029      |       <u>0.673</u>      |      <u>0.233<u>      |
| DivAugGAN |   <u>70.51</u>    |    <u>0.236</u>     |       0.690        |     <u>0.044</u>      |       0.566      |      0.213      |



<font size=3>  **Photo  &rarr; Monet**  </font> <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-03-09:47:19.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-04-photo2monet/2014-08-06-19:34:34.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Photo  &rarr; Monet**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>


<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Photo  &rarr; Monet**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   **51.98**    |    0.193     |       **0.623**        |     0.063      |       **0.482**      |      0.854      |
| MSGAN     |   <u>52.16</u>    |    **0.356**     |       <u>0.599</u>        |     **0.209**      |       <u>0.463</u>      |      **0.891**      |
| DSGAN     |   55.98    |    <u>0.321</u>     |       0.596        |     0.142      |       0.456      |      </u>0.857<u>      |
| DivCo     |   55.31    |    0.303     |       0.522        |     0.152      |       0.331      |      0.809      |
| DivAugGAN |   55.58    |    0.306     |       0.548        |     <u>0.157</u>      |       0.413      |      0.820      |
 
 

<font size=3>  **Photograph  &rarr; Portrait**  </font> <br><br>

<img src="images/comparison-results/UI2I-05-photo2portrait/photo2portrait-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-05-photo2portrait/photo2portrait-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Photo  &rarr; Portrait**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>

<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Photograph  &rarr; Portrait**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   **49.21**    |    0.401     |       0.761        |     0.174      |       <u>0.938</u>      |      <u>0.295</u>      |
| MSGAN     |   50.43    |    **0.496**     |       0.768        |     **0.216**      |       0.934      |      **0.302**      |
| DSGAN     |   <u>49.86</u>    |    <u>0.474</u>     |       0.768        |     **0.216**      |       0.906      |      0.282      |
| DivCo     |   52.88    |    0.270     |       <u>0.771</u>        |     0.111      |       **0.950**      |      0.273      |
| DivAugGAN |   57.21    |    0.310     |       **0.772**        |     0.182      |       0.754      |      0.225      |
 
 

<font size=3>  **Portrait  &rarr; Photograph**  </font> <br><br>

<img src="images/comparison-results/UI2I-06-portrait2photo/portrait2photo-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-06-portrait2photo/portrait2photo-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Portrait  &rarr; Photo**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>

<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Portrait  &rarr; Photograph**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   59.06    |    0.448     |       <u>0.913</u>        |     0.204      |       **1.791**      |      <u>0.819</u>      |
| MSGAN     |   **49.43**    |    <u>0.581</u>     |       **0.945**        |     0.204      |       1.243      |      **0.856**      |
| DSGAN     |   <u>45.32</u>    |    **0.594**     |       0.900        |     **0.260**      |       <u>1.748</u>      |      0.818      |
| DivCo     |   67.71    |    0.237     |       0.796        |     0.093      |       0.940      |      0.564      |
| DivAugGAN |   54.96    |    0.403     |       0.796        |     <u>0.253</u>      |       0.952      |      0.694      |

 

<font size=3>  **Summer  &rarr; Winter**  </font> <br><br>

<img src="images/comparison-results/UI2I-07-summer2winter/summer2winter-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-07-summer2winter/summer2winter-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Summer  &rarr; Winter**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>


<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Summer  &rarr; Winter**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   52.57    |    0.117     |       0.761        |     0.025      |       1.008      |      0.865      |
| MSGAN     |   47.78    |    0.231     |       0.786        |     0.049      |       1.001      |      0.905      |
| DSGAN     |   48.89    |    0.128     |       0.775        |     0.007      |       1.028      |      0.899      |
| DivCo     |   50.82    |    0.097     |       0.751        |     0.001      |       0.948      |      0.843      |
| DivAugGAN |   48.87    |    0.189     |       0.754        |     0.036      |       1.044      |      0.931      |
 

<font size=3>  **Winter  &rarr; Summer**  </font> <br><br>

<img src="images/comparison-results/UI2I-08-winter2summer/winter2summer-1.png" width="800px"/>  <br><br>

<img src="images/comparison-results/UI2I-08-winter2summer/winter2summer-2.png" width="800px"/>  <br><br>

<font size=4> Qualitative diversity comparisons of DivAugGAN (fifth row) with DRIT (first row), MSGAN (second row), DSGAN (third row), and DivCo (fourth row) on **Winter  &rarr; Summer**. The first column shows the input images and the remaining 10 columns presents the generated multimodal images with 10 different input latent vectors for this unpaired multimodal image-toimage translation. DivAugGAN generates images with promising content preservation performance (cloud shape) and adequate variation (diverse color and light) for the different input latent vectors. </font>  <br><br>


<font size=4> Quantitative comparisons of DivAugGAN with DRIT, MSGAN, DSGAN, and DivCo on **Winter  &rarr; Summer**. <br> 

| Methods   | FID &darr; | LPIPS &uarr; |  Precision &uarr;  |  Recall &uarr; |   Density &uarr; | Coverage &uarr; |
|-----------|:----------:|--------------|:------------------:|:--------------:|:----------------:|:---------------:|
| DRIT      |   53.77    |    0.062     |       0.751        |     0.006      |       0.977      |      0.717      |
| MSGAN     |   41.02    |    0.217     |       0.788        |     0.063      |       0.910      |      0.828      |
| DSGAN     |   44.81    |    0.144     |       0.783        |     0.020      |       0.989      |      0.821      |
| DivCo     |   47.58    |    0.125     |       0.755        |     0.016      |       0.782      |      0.757      |
| DivAugGAN |   42.33    |    0.179     |       0.749        |     0.039      |       0.925      |      0.849      |
 

 

### <font size=4>  Multi-domain unpaired image-to-image translation qualitative comparisons </font>
<!--  ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/afhq-transfer.png)  -->
<!--   ![Image text](https://github.com/anonymous-gan/DivAugGAN/blob/master/images/image-weather-conditions.png)  -->


<font size=3>  Alps seasonal transfer </font> <br><br>

<font size=3>  Spring  &rarr; Spring  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/spring2spring-summer-autumn-winter/spring2spring/7195527734_c8e4e84d00_z.png" width="800px"/>  <br><br>

<font size=3>  Spring  &rarr; Summer  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/spring2spring-summer-autumn-winter/spring2summer/7195527734_c8e4e84d00_z.png" width="800px"/>  <br><br>


<font size=3>  Spring  &rarr; Autumn  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/spring2spring-summer-autumn-winter/spring2autumn/7195527734_c8e4e84d00_z.png" width="800px"/>  <br><br>

<font size=3>  Spring  &rarr; Winter  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/spring2spring-summer-autumn-winter/spring2winter/7195527734_c8e4e84d00_z.png" width="800px"/>  <br><br>

<font size=3>  Summer  &rarr; Spring  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/summer2spring-summer-autumn-winter/summer2spring/7352618582-0fc508e670-z.png" width="800px"/>  <br><br>

<font size=3>  Summer  &rarr; Summer  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/summer2spring-summer-autumn-winter/summer2summer/7352618582_0fc508e670_z.png" width="800px"/>  <br><br>

<font size=3>  Summer  &rarr; Autumn  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/summer2spring-summer-autumn-winter/summer2autumn/7352618582_0fc508e670_z.png" width="800px"/>  <br><br>

<font size=3>  Summer  &rarr; Winter  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/summer2spring-summer-autumn-winter/summer2winter/7352618582_0fc508e670_z.png" width="800px"/>  <br><br>

<font size=3>  Autumn  &rarr; Spring  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/autumn2spring-summer-autumn-winter/autumn2spring/21848230103_69b75ef4f9_z.png" width="800px"/>  <br><br>

<font size=3>  Autumn  &rarr; Summer  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/autumn2spring-summer-autumn-winter/autumn2summer/21848230103_69b75ef4f9_z.png" width="800px"/>  <br><br>

<font size=3>  Autumn  &rarr; Autumn  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/autumn2spring-summer-autumn-winter/autumn2autumn/21848230103_69b75ef4f9_z.png" width="800px"/>  <br><br>

<font size=3>  Autumn  &rarr; Winter  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/autumn2spring-summer-autumn-winter/autumn2winter/21848230103_69b75ef4f9_z.png" width="800px"/>  <br><br>

<font size=3>  Winter  &rarr; Spring  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/winter2spring-summer-autumn-winter/winter2spring/13803289324_f3ca101524_z.png" width="800px"/>  <br><br>

<font size=3>  Winter  &rarr; Summer  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/winter2spring-summer-autumn-winter/winter2summer/13803289324_f3ca101524_z.png" width="800px"/>  <br><br>

<font size=3>  Winter  &rarr; Autumn  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/winter2spring-summer-autumn-winter/winter2autumn/13803289324_f3ca101524_z.png" width="800px"/>  <br><br>

<font size=3>  Winter  &rarr; Winter  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-alps/winter2spring-summer-autumn-winter/winter2winter/13803289324_f3ca101524_z.png" width="800px"/>  <br><br>

<font size=3>  Arts </font> <br><br>

<font size=3>  Cezanne  &rarr; Cezanne  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/cezanne2cezanne-monet-photo-ukiyoe-vangogh/cezanne2cezanne/00220.png" width="800px"/>  <br><br>

<font size=3>  Cezanne  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/cezanne2cezanne-monet-photo-ukiyoe-vangogh/cezanne2monet/00220.png" width="800px"/>  <br><br>

<font size=3>  Cezanne  &rarr; Ukiyoe  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/cezanne2cezanne-monet-photo-ukiyoe-vangogh/cezanne2ukiyoe/00220.png" width="800px"/>  <br><br>

<font size=3>  Cezanne  &rarr; Van Gogh  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/cezanne2cezanne-monet-photo-ukiyoe-vangogh/cezanne2vangogh/00220.png" width="800px"/>  <br><br>

<font size=3>  Cezanne  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/cezanne2cezanne-monet-photo-ukiyoe-vangogh/cezanne2photo/00220.png" width="800px"/>  <br><br>

<font size=3>  Monet  &rarr; Cezanne  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/monet2cezanne-monet-photo-ukiyoe-vangogh/monet2cezanne/00480.png" width="800px"/>  <br><br>

<font size=3>  Monet  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/monet2cezanne-monet-photo-ukiyoe-vangogh/monet2monet/00480.png" width="800px"/>  <br><br>

<font size=3>  Monet  &rarr; Ukiyoe  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/monet2cezanne-monet-photo-ukiyoe-vangogh/monet2ukiyoe/00480.png" width="800px"/>  <br><br>

<font size=3>  Monet  &rarr; Van Gogh  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/monet2cezanne-monet-photo-ukiyoe-vangogh/monet2vangogh/00480.png" width="800px"/>  <br><br>

<font size=3>  Monet  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/monet2cezanne-monet-photo-ukiyoe-vangogh/monet2photo/00480.png" width="800px"/>  <br><br>


<font size=3>  Ukiyoe  &rarr; Cezanne  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/ukiyoe2cezanne-monet-photo-ukiyoe-vangogh/ukiyoe2cezanne/01202.png" width="800px"/>  <br><br>

<font size=3>  Ukiyoe  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/ukiyoe2cezanne-monet-photo-ukiyoe-vangogh/ukiyoe2monet/01202.png" width="800px"/>  <br><br>

<font size=3>  Ukiyoe  &rarr; Ukiyoe  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/ukiyoe2cezanne-monet-photo-ukiyoe-vangogh/ukiyoe2ukiyoe/01202.png" width="800px"/>  <br><br>

<font size=3>  Ukiyoe  &rarr; Van Gogh  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/ukiyoe2cezanne-monet-photo-ukiyoe-vangogh/ukiyoe2vangogh/01202.png" width="800px"/>  <br><br>

<font size=3>  Ukiyoe  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/ukiyoe2cezanne-monet-photo-ukiyoe-vangogh/ukiyoe2photo/01202.png" width="800px"/>  <br><br>


<font size=3>  Van Gogh  &rarr; Cezanne  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/vangogh2cezanne-monet-photo-ukiyoe-vangogh/vangogh2cezanne/00010.png" width="800px"/>  <br><br>

<font size=3>  Van Gogh  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/vangogh2cezanne-monet-photo-ukiyoe-vangogh/vangogh2monet/00010.png" width="800px"/>  <br><br>

<font size=3>  Van Gogh  &rarr; Ukiyoe  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/vangogh2cezanne-monet-photo-ukiyoe-vangogh/vangogh2ukiyoe/00010.png" width="800px"/>  <br><br>

<font size=3>  Van Gogh  &rarr; Van Gogh  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/vangogh2cezanne-monet-photo-ukiyoe-vangogh/vangogh2vangogh/00010.png" width="800px"/>  <br><br>

<font size=3>  Van Gogh  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/vangogh2cezanne-monet-photo-ukiyoe-vangogh/vangogh2photo/00010.png" width="800px"/>  <br><br>


<font size=3>  Photo  &rarr; Cezanne  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/photo2cezanne-monet-photo-ukiyoe-vangogh/photo2cezanne/2014-08-01-22:38:22.png" width="800px"/>  <br><br>

<font size=3>  Photo  &rarr; Monet  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/photo2cezanne-monet-photo-ukiyoe-vangogh/photo2monet/2014-08-01-22:38:22.png" width="800px"/>  <br><br>

<font size=3>  Photo  &rarr; Ukiyoe  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/photo2cezanne-monet-photo-ukiyoe-vangogh/photo2ukiyoe/2014-08-01-22:38:22.png" width="800px"/>  <br><br>

<font size=3>  Photo  &rarr; Van Gogh  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/photo2cezanne-monet-photo-ukiyoe-vangogh/photo2vangogh/2014-08-01-22:38:22.png" width="800px"/>  <br><br>

<font size=3>  Photo  &rarr; Photo  </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-arts/photo2cezanne-monet-photo-ukiyoe-vangogh/photo2photo/2014-08-01-22:38:22.png" width="800px"/>  <br><br>

<font size=3>  AFHQ </font> <br><br>

<img src="images/comparison-results/Multidomain-I2I-afhq/afhq.png" width="960px"/>  <br><br>

<br>



## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN