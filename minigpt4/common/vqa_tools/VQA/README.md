Python API and Evaluation Code for v2.0 and v1.0 releases of the VQA dataset.
===================
## VQA v2.0 release ##
This release consists of
- Real 
	- 82,783 MS COCO training images, 40,504 MS COCO validation images and 81,434 MS COCO testing images (images are obtained from [MS COCO website] (http://mscoco.org/dataset/#download))
	- 443,757 questions for training, 214,354 questions for validation and 447,793 questions for testing
	- 4,437,570 answers for training and 2,143,540 answers for validation (10 per question)

There is only one type of task
- Open-ended task

## VQA v1.0 release ##
This release consists of
- Real 
	- 82,783 MS COCO training images, 40,504 MS COCO validation images and 81,434 MS COCO testing images (images are obtained from [MS COCO website] (http://mscoco.org/dataset/#download))
	- 248,349 questions for training, 121,512 questions for validation and 244,302 questions for testing (3 per image)
	- 2,483,490 answers for training and 1,215,120 answers for validation (10 per question)
- Abstract
	- 20,000 training images, 10,000 validation images and 20,000 MS COCO testing images
	- 60,000 questions for training, 30,000 questions for validation and 60,000 questions for testing (3 per image)
	- 600,000 answers for training and 300,000 answers for validation (10 per question)

There are two types of tasks
- Open-ended task
- Multiple-choice task (18 choices per question)

## Requirements ##
- python 2.7
- scikit-image (visit [this page](http://scikit-image.org/docs/dev/install.html) for installation)
- matplotlib (visit [this page](http://matplotlib.org/users/installing.html) for installation)

## Files ##
./Questions
- For v2.0, download the question files from the [VQA download page](http://www.visualqa.org/download.html), extract them and place in this folder.
- For v1.0, both real and abstract, question files can be found on the [VQA v1 download page](http://www.visualqa.org/vqa_v1_download.html).
- Question files from Beta v0.9 release (123,287 MSCOCO train and val images, 369,861 questions, 3,698,610 answers) can be found below
	- [training question files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Questions_Train_mscoco.zip)
	- [validation question files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Questions_Val_mscoco.zip)
- Question files from Beta v0.1 release (10k MSCOCO images, 30k questions, 300k answers) can be found [here](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.1/Questions_Train_mscoco.zip).

./Annotations
- For v2.0, download the annotations files from the [VQA download page](http://www.visualqa.org/download.html), extract them and place in this folder.
- For v1.0, for both real and abstract, annotation files can be found on the [VQA v1 download page](http://www.visualqa.org/vqa_v1_download.html).
- Annotation files from Beta v0.9 release (123,287 MSCOCO train and val images, 369,861 questions, 3,698,610 answers) can be found below
	- [training annotation files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Annotations_Train_mscoco.zip)
	- [validation annotation files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Annotations_Val_mscoco.zip)
- Annotation files from Beta v0.1 release (10k MSCOCO images, 30k questions, 300k answers) can be found [here](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.1/Annotations_Train_mscoco.zip).

./Images
- For real, create a directory with name mscoco inside this directory. For each of train, val and test, create directories with names train2014, val2014 and test2015 respectively inside mscoco directory, download respective images from [MS COCO website](http://mscoco.org/dataset/#download) and place them in respective folders.
- For abstract, create a directory with name abstract_v002 inside this directory. For each of train, val and test, create directories with names train2015, val2015 and test2015 respectively inside abstract_v002 directory, download respective images from [VQA download page](http://www.visualqa.org/download.html) and place them in respective folders.

./PythonHelperTools
- This directory contains the Python API to read and visualize the VQA dataset
- vqaDemo.py (demo script)
- vqaTools (API to read and visualize data)

./PythonEvaluationTools
- This directory contains the Python evaluation code
- vqaEvalDemo.py (evaluation demo script)
- vqaEvaluation (evaluation code)

./Results
- OpenEnded_mscoco_train2014_fake_results.json (an example of a fake results file for v1.0 to run the demo)
- Visit [VQA evaluation page] (http://visualqa.org/evaluation) for more details.

./QuestionTypes
- This directory contains the following lists of question types for both real and abstract questions (question types are unchanged from v1.0 to v2.0). In a list, if there are question types of length n+k and length n with the same first n words, then the question type of length n does not include questions that belong to the question type of length n+k.
- mscoco_question_types.txt
- abstract_v002_question_types.txt

## References ##
- [VQA: Visual Question Answering](http://visualqa.org/)
- [Microsoft COCO](http://mscoco.org/)

## Developers ##
- Aishwarya Agrawal (Virginia Tech)
- Code for API is based on [MSCOCO API code](https://github.com/pdollar/coco).
- The format of the code for evaluation is based on [MSCOCO evaluation code](https://github.com/tylin/coco-caption).
