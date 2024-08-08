# Lab1 Vanilla GAN

<br/>

## Lab Overview

<div align='center'>
    <img src='../figures/overview.png' width='600'/>
</div>

<br/>

<li><b>Input</b>: 100 dimensional noise</li>
<li><b>Generator</b>: Input is above noise value.</li>
<li><b>Discriminator</b>: Input is input size as you defined (eg. 28x28, 256x256, etc).</li>
<li><b>Loss</b>: BCELoss</li>
<li><b>Generator optimizer</b>: Adam</li>
<li><b>Discriminator optimizer</b>: Adam</li>

<br/>
<br/>

## Datasets 

<li>FashionMNIST</li>

<br/>
<br/>

## How to train

```
python VanillaGAN.py --noise 10 --input_size 28x28 --batch_size 64 --epochs 100 --model_dir result/model --images_dir result/images --loss_dir result/loss
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
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>


