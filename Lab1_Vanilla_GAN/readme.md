# Lab1_Vanilla_GAN

<br/>

## Lab Overview

<div align='center'>
    <img src='../figures/overview.png' width='600'/>
</div>

<br/>

<li><b>Input</b> : 10 dimensional noise</li>
<li><b>Generator</b></li>
<li><b>Discriminator</b></li>
<li><b>Loss</b></li>

<br/>
<br/>

## Datasets 

<li><b>FashionMNIST</b>: </li>
<li><b>Korean Font</b>: In progress ... </li>
<li><b>Custom</b>: In progress ... </li>

<br/>
<br/>

## How to train

```
python VanillaGAN.py --noise --input_size --batch_size --epochs --model_dir --images_dir --loss_dir
```
- <b>noise</b> Define noise value
- <b>input_size</b> Define input size eg. 28x28, 256x256, ...
- <b>batch_size</b> Define bacth size eg. 32, 64, 128, ...
- <b>epochs</b> Define training epochs value
- <b>model_dir</b> Directory to save checkpoint
- <b>images_dir</b> Directory to save generated images
- <b>loss_dir</b> Directory to save loss output

<br/>
<br/>

## Result

<br/>

<div align='center'>
    <b>FasionMNIST</b>
</div>

<br/>

<div align='center'>
    <img src='../figures/result_1-4_reduced.gif' width='500'/>
</div>

<br/>
<br/>
<br/>

<div align='center'>
    <b>Korean</b>
</div>

<br/>

<div align='center'>
    <img src='../figures/result_2-2_reduced.gif' width='500'/>
</div>

<br/>
<br/>
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>


