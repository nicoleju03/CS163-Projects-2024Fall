---
layout: post
comments: true
title: Fashion Image Editing
author: Antara Chugh, Joy Cheng, Caroline DebBaruah, & Nicole Ju
date: 2024-12-11
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Introduction
Computer vision has played an increasingly large role in fashion-related problems such as the recommendation of similar clothing items, the recognition of garments, and the virtual try-on of outfits. There have been several efforts to expand on virtual try-on and employ computer vision-based editing of fashion images, which involves generating realistic fashion designs on images of models using a variety of inputs such as text prompts and sketches. Fashion image editing with computer vision can help a fashion designer efficiently visualize clothing items on any number of people without requiring their physical presence. Here, we examine three different approaches to fashion image editing:

1. **Multimodal Garment Designer (MGD)**, a latent diffusion-based solution that takes a model image, its pose map, a textual description of a garment, and a sketch of the garment as the inputs and produces an image of the model wearing the target garment as the output.
1. **Fashion Image CLIP Editing (FICE)**, a generative adversarial network-based solution that takes a model image and a textual description of a garment as the inputs and produces an image of the model wearing the target garment as the output.
1. **Multi-Garment Virtual Try-On and Editing (M&M VTO)**, a diffusion- and transformer-based solution that takes a model image, multiple garment images, and a textual description as the inputs and produces an image of the model wearing the target garments as the output.

## Multimodal Garment Designer (MGD)
Many existing works explore the conditioning of diffusion models on various modalities, like text descriptions and sketches, to allow for more control over image generation. **Multimodal Garment Designer (MGD)**, in particular, focuses on the fashion domain and is a human-centric architecture that builds on latent diffusion models. It is conditioned on multiple modalities: textual sentences, human pose maps, and garment sketches.

Given an input image $$I \in \mathbb{R}^{H \times W \times 3}$$, MGD’s goal is to generate a new image $$I$$' of the same dimensions that retains the input model’s information, while replacing the existent garment with a target garment.



### Stable Diffusion Model
MGD builds off of the **Stable Diffusion Model**, which is a latent diffusion model that involves an encoder $$E$$ to convert the image $$I$$ into a latent space of dimension $$\frac{H}{8} \times \frac{W}{8} \times 4$$, and a decoder $$D$$ that converts back into the image space. It uses a CLIP-based text encoder $$T_E$$, which takes input $$Y$$, and a text-time-conditional U-Net denoising model $$\epsilon_{\theta}$$. The denoising network $$\epsilon_{\theta}$$ minimizes the loss:

$$
L = \mathbb{E}_{\epsilon(1), Y, \epsilon \sim 𝒩(0,1), t} \left[ \left\| \epsilon - \epsilon_0(\gamma, \psi) \right\|\ _2^2 \right]
$$

where $$t$$ is the time step, $$\gamma$$ is the spatial input to the denoising network, $$\psi = \begin{bmatrix} t; T_E(Y) \end{bmatrix}$$, and $$\epsilon \sim 𝒩(0, 1)$$ is the Gaussian noise added to the encoded image.

The Stable Diffusion Model is a state-of-the art text-to-image model widely known for its ability to generate high-quality, realistic images from textual descriptions. MGD broadens its scope to focus on human-centric fashion image editing, maintaining the body information of the input model while also incorporating an input sketch.

### Conditioning Constraints
Instead of employing a standard text-to-image model, we also need to perform inpainting to replace the input model’s garment using the multimodal inputs. The denoising network input is concatenated with an encoded mask image and binary inpainting mask. Because the encoder and decoder are fully convolutional, the model can preserve spatial information in the latent space; thus, we can add constraints to the generation process, in addition to the textual information. 

First, we can condition on the pose map $$P$$, which represents various body keypoints, to preserve the input model’s pose information. MGD proposes to improve the garment inpainting by utilizing the pose map in addition to the segmentation mask. Specifically, it adds 18 additional channels to the first convolution layer of the denoising layer (one for each keypoint). 

MGD also utilizes a garment sketch $$S$$ to capture spatial characteristics that text descriptions may not fully be able to describe. Similar to the pose map, additional channels are added for the garment sketches. The final input to the denoising network is:

$$\gamma = \begin{bmatrix} z_t ; m ; E(I_M) ; p ; s \end{bmatrix}, \quad [p; s] \in \mathbb{R}^{H/8 \times W/8 \times (18+1)}
$$

where $$z_t$$ is the convolutional input, $$m$$ is the binary inpainting mask, $$E(I_M)$$ is the encoded masked image, and $$p$$ and $$s$$ are resized versions of $$P$$ and $$S$$ to match the latent dimensions.

### Training
**Classifier-free guidance** is used during the training process, meaning the denoising network is trained to work both with a condition and without any condition. This process aims to slowly move the unconditional model toward the conditional model by modifying its predicted noise. Since MGD uses multiple conditions, it computes the direction using joint probability of all conditions. We also use **unconditional training**, where we replace one or more of the conditions with a null value according to a set probability. This improves model versatility, as it must learn to produce results when certain conditions are taken away. 

### Evaluation and Metrics
Many different metrics can be used to assess the performance of MGD. Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) measure the differences between real and generated images, and thus help represent the realism and quality of images. The CLIP Score (CLIP-S) captures the adherence of the image to the textual input. MGD also uses novel metrics: pose distance (PD), which compares the human pose of the original image to the generated one, and sketch distance (SD) which reflects how closely the generated image adheres to the sketch constraint. 

#### Comparison with Other Methods

![Results]({{ '/assets/images/30/MGD_Results.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig INSERT. Comparison of results on the Dress Code Multimodal and VITON-HD Multimodal datasets* [INSERT].

MGD was tested for paired and unpaired settings; in the paired settings, the conditions refer to the garment the model is wearing, while in the unpaired settings, the target garment differs from the worn one. The results on the Dress Code Multimodal and VITON-HD Multimodal datasets outperform competitors in both realism and adherence to the inputs (text, pose map, garment sketch). It produces much lower FID and KID scores compared to other models, slightly higher CLIP scores, and lower PD and SD scores due to the pose map and garment sketch conditioning. Stable Diffusion produces realistic results but fails to preserve the model’s pose information because such data is not included in the inputs to the model.



## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
