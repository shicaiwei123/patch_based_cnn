# patch_based_cnn
the  implement  of  Face Anti-Spoofing Using Patch and Depth-Based CNNs


English|[中文](https://github.com/shicaiwei123/patch_based_cnn/blob/master/ReadMe_CH.md)

# Introduction
- data
    - The data after dividing into different patchs.
- lib
    - Shared basic functions: dataset, data processing and model training 
- model
    - the code of net
- output
    - save model and logs
- src
    - main function and other auxiliary functions
- test
    - test a single image, which is used for deployment 
- config.py
    - parameter configuration


# User guide
- patch_based_cnn
    - Firstly, run generate.py to divide the living and spoofing img into different patches according to paper and save them as the test set and training set in the data folder. In this section, users need to prepare their own data or public data sets. We did not upload the data. You need to modify the path to your own data path.
    - Modify the configuration file: patch_based_cnn.py
    - Then train and test.

    ## Run
    
    - Divide img to different patchs(64 is used in article)
        ```
        python3 data_generate.py
        ```
    
    - Train and test
        ```
        cd src
        python3 patch_cnn_main.py
        ```
    - Single image test. In this section,you can test a single image,which is used for deployment.
        ```
        cd test 
        python patch_cnn_test.py
        ```
    
    ## Result
    - the data enhancement such as learning rate decay and mixup has not been used for the follow results.
    
        | Dataset    | Average accuracy(%) |
        | :----------| --- |
        | CASIA-FASD |  93.52 |
        | CASIA-SURF |  88.88 |
        
- depth_based_cnn
    - Firstly, run [PRNET](https://github.com/YadiraF/PRNet) to get depth map of living faces. We did not upload the data. You need to modify the path to your own data path. it is worth noting that  acquiring depth maps from PRNET may take  much time, so we take a 52x52 matrix with 1 as the ground truth of living faces here.
    - Modify the configuration file: patch_based_cnn.py
    - Then train and test.

    ## Run
    
    - Train and test
        ```
        cd src
        python3 depth_cnn_main.py
        ```
    - Single image test. In this section,you can test a single image,which is used for deployment.
        ```
        cd test 
        python depth_cnn_test.py
        ```
