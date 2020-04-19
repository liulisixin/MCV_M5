# Master in computer vision (MCV): M5 Visual Recognition Module

Team 06 repository

# Documentation

Link to overleaf paper: https://www.overleaf.com/read/rphxcrxhpgmd

Link to presentation slides: https://docs.google.com/presentation/d/1mnWuenpjyBr6ViBmbNpcGeiCicX1kMv7wvfGmLtuBk0/edit?usp=sharing

## Reports

### Week 1
* Slides: https://docs.google.com/presentation/d/169s5x1e9TEeDZlBNCVuwIWeYEQtlGO_WIN30iOxkPE0/edit?usp=sharing
* Code: the source code is located on folder Week1. 
* Run code: You can run it using `sbatch job_ruben` and it will tigger the Pytorch ruben version same with the Keras version.

### Week 2
* Slides: https://docs.google.com/presentation/d/1mHO-So5vzC89FCnZvbcdVUSzPMOTXorG-V8YU6ntmSA/edit?usp=sharing
* Code: Week 2
* Run code: `sbatch job_task_x` where x could be B,C or D.

### Week 3
* Team 06 slides: https://docs.google.com/presentation/d/1KcJaLKY47Jq1Qr72WDtIQA2ZUnPr_JiVQbtodnIZLB8/edit?usp=sharing
* Slides: https://docs.google.com/presentation/d/1rppl8bJZF5lnt4Qxvoe_KrF_eDC2S-eNhT6g58L_NlE/edit#slide=id.g711500f7df_0_11
* Code: Week 3

### Week 4
* Team 06 slides: https://docs.google.com/presentation/d/1y0-lhYYyhSCk8liCfAdRRBXJltzeedJMmWCdd1GY3Ic/edit#slide=id.g6fda4ed841_0_0
* Slides: https://docs.google.com/presentation/d/1Wxv_nS51v2C9CKlNpzeHORPC9lifEhkCmpZSD9jJOXA/edit#slide=id.g7f4da1844a_0_12
* Code: Week 4

### Week 5
* Team 06 slides: https://docs.google.com/presentation/d/1qqR4o1meUzZMC8N6lFSGuocggcT9y1Gy_QTy-YR7ktk/edit?usp=sharing
* Slides: https://docs.google.com/presentation/d/1GoxeIPR7aRU02mNyxeSnqRkAa7uV55FtJaIlOdWdFMM/edit#slide=id.p
* Code: Week 5

### Week 6
* Slides:  https://docs.google.com/presentation/d/1ydBIwr2Vx4eIkHH6BRrn0nSDtqjCqCaG16S4zq_4Se8/edit#slide=id.g7392858802_30_0
* Code: Week 6

# DeepLabv3+ Experimental setup
We have used the tensorflow framework used in the original work to get the experimental setup done. 
To run training with the DeepLabV3+ and to reproduce the results obtained in the paper [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf) with CityScapes dataset one needs to follow these steps:

  - Clone the tensorflow model from the given repository link: https://github.com/tensorflow/models
  - From tensorflow/models/research/deeplab/datasets: We copy the already downloaded cityscapes dataset from the path home/mcv/datasets/cityscapes. Other option is to download the [cityscapes dataset]( https://www.cityscapes-dataset.com/)  from the official website. 
  - Clone https://github.com/mcordts/cityscapesScripts.git 
  - Install cityscapeScripts with the following instruction:
     ```sh
        $ pip install cityscapesscripts
     ```
  - Then run the following command to generate the tfrecords for generating the train and val records to get the data prepared:
     ```sh
        $ sh convert_cityscapes.sh
     ```

>  datasets  
├── cityscapes
├── leftImg8bit
├── gtFine
├── tfrecord
├── exp
├── ├── train_on_train_set
├── ├── eval
├── ├── vis

  - Inside the tfrecord folder rename the train* files to train_fine* file and val* to val_fine* to be able to train on train_fine dataset.
  -  export PYTHONPATH from tensorflow/models/research  as 
     ```sh
        $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
     ```
  - To train the deeplab model you need to run the following training script:
    ```sh
        $ python deeplab/train.py \
            --logtostderr \
            --training_number_of_steps=90000 \
            --train_split="train_fine" \
            --model_variant="xception_65 OR xception_71" \
            --atrous_rates=6 \ THESE ATROUS RATES ACTIVATE ASPP
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \ IF DECODER IS ACTIVATED 
            --add_image_level_feature=True OR False \
            --train_crop_size="769,769" \
            --train_batch_size=1 \
            --dataset="cityscapes" \
            --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
            --train_logdir=${PATH_TO_TRAIN_DIR} \
            --dataset_dir=${PATH_TO_DATASET}
     ```
    where the model_variant is between xception_65 and xception_71, --decoder_output_stride=4 must be provided if Decoder is present, and --add_image_level_feature is either True or False depending on the presence of Image Level Features. The Initial checkpoint is chosen to be xception_65 or xception_71 pretrained on imageNet taken from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md. Please note that PATH_TO_DATASET must point to the tfrecord/ folder.
    
  - To evaluate the deeplab model you need to run the following evaluation script:
     ```sh
        $ python deeplab/eval.py \
            --logtostderr \
            --eval_split="val_fine" \
            --model_variant="xception_65 OR xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \ IF DECODER IS ACTIVATED 
            --add_image_level_feature=True OR False \
            --eval_crop_size="769,769" \
            --dataset="cityscapes" \
            --checkpoint_dir=${PATH_TO_CHECKPOINT} \
            --eval_logdir=${PATH_TO_EVAL_DIR} \
            --dataset_dir=${PATH_TO_DATASET}
     ```
  - To visualize the results of the deeplab model qualitatively you need to run the following script:
    ```sh
        $ python deeplab/vis.py \
            --logtostderr \
            --vis_split="val_fime" \
            --model_variant="xception_65 OR xception_71" \
            --atrous_rates=6 \
            --atrous_rates=12 \
            --atrous_rates=18 \
            --output_stride=16 \
            --decoder_output_stride=4 \ IF DECODER IS ACTIVATED 
            --add_image_level_feature=True OR False \
            --vis_crop_size="1025,2049" \
            --dataset="cityscapes" \
            --colormap_type="cityscapes" \
            --checkpoint_dir=${PATH_TO_CHECKPOINT} \
            --vis_logdir=${PATH_TO_VIS_DIR} \
            --dataset_dir=${PATH_TO_DATASET}
     ```
    
  
## Credits

| Name | Email |
|:----:|:------|
| Yixiong Yang | yangyxwork@163.com |
| Sanket Biswas | sanketbiswas1995@gmail.com |
| Ruben Bagan | ruben.bagan@gmail.com |

