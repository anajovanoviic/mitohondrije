# mitohondrije

*patchify_to_generate.py*

![patches](https://user-images.githubusercontent.com/51513732/226105240-99ce385c-6dad-4166-a667-eb8908c380d4.jpg)

Above is the graphical example of one image in training.tif dataset and how it is divided into patches. Dimension of the patches above (256x256) are dimensions of every image inside patches/images folder. <br>

In the first dataset *training.tif* there is 165 images, so in the *patches/images* folder there will be 165*12=1980 images. <br>

*simple_unet.py* <br>

By adding *axis=1* dimension of the image is expanded by 1, because we are only dealing with grayscale images
```
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
```
Result of the above command:
```
image_dataset.shape
```
Out: *(1980, 256, 256, 1)* <br>

**If it is colour images you would have (256,256,3)**
