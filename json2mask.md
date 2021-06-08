# json2mask

In order to accomplish the json2mask task, we should understand the list and dictionary python datatypes well. This task will be easy peasy (lemon squeezy); if you know/learn the data types mentioned above.

In our annotation files, JSON files have the structure as in the following:

```
json_dict = {
    "description": empty string
    "tags": list of tag_dict
    "size": size_dict
    "objects": list of object_dict
		}
```

In this dictionary, we will be working on `objects`' values. Other key-value pairs will not be significant. tags & size are not essential but shared below for those interested:

```
tag_dict = {
    "id": int
    "tagId": int
    "name": string
    "value": string
    "labelerLogin": string
    "createdAt": string
    "updatedAt": string
}

size_dict = {
    "height": int
    "width": ing
}
```

Our primary focus is to find freespace objects: we have to go through every object and check out if the object belongs to the `Freespace` by checking `classTitle`. We need to get every point of the polygon by calling it the “exterior” of points.

```
object_dict = {
    "id": int
    "classId": int,
    "description": empty string
    "geometryType": string
    "labelerLogin": string,
    "createdAt": string,
    "updatedAt": string,
    "tags": empty list,
    "classTitle": string
    "points": point_dict
}

point_dict = {
    "exterior": list of point_list
    "interior": empty list
}

point_list : [x, y]
```

There is an example below. In the example, if an object with an object id of 61 is found among the `objects`, the text "I like it" is displayed.

```python
import numpy as np
import cv2
import json
import os

# Path to mask
MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to jsons
JSON_DIR  = '../data/jsons'
json_name = 'ozan.json'

# Access and open json file
json_path = os.path.join(JSON_DIR, json_name)
json_file = open(json_path, 'r')

# Load json data
json_dict = json.load(json_file)

# Extract "objects" information
json_objs = json_dict["objects"]

# Read every "objects"
for obj in json_objs:
    obj_id = obj['id']
    if obj_id > 61:
        # Condition is fulfilled
        print('I like it')
    else:
        continue
```