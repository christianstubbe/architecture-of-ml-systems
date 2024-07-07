# Setup
- We assume a Debian/Ubuntu Distribution to run the code on
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```


#  Links for Satellite Images


Good for looking at bounding boxes:

http://bboxfinder.com

---
(52.230095,12.760088,52.819950,13.979952)

![alt text](img/image.png)

Because the initial Bounding box is far wider than the actual city bounds, we need to reduce the bounds to the actual city


## Images

#### Buildings
![alt text](img/buildings.png)


#### RGB Bands Sentinel-2 L2A
![alt text](img/rgb_sentinel.png)

#### False Color Image
![alt text](img/false_color.png)

#### Buildings Mask
![alt text](img/buildings_mask.png)


and in blue
![alt text](img/buildings_mask_blues.png)



Greyscale IMG with buildings mask
![alt text](img/greyscale_with_build_mask.png)
