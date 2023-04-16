## Second Stage Data Preparation

Our second stage dataset can be downloaded from 
[here](https://drive.google.com/file/d/1nJXhoEcy3KTExr17I7BXqY5Y9Lx_-n-9/view?usp=share_link) 
After extraction, you will get a data follder with the following structure:

```
cc_sbu_align
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...   
```

Put the folder to any path you want.
Then, set up the dataset path in the dataset config file 
[here](../minigpt4/configs/datasets/cc_sbu/align.yaml#L5) at Line 5.

