# patch_based_cnn
the  implement  of  Face Anti-Spoofing Using Patch and Depth-Based CNNs

[English](https://github.com/shicaiwei123/patch_based_cnn/blob/master/README.md)|中文

# 结构介绍
- data
    - 用于存存放数据，数据格式如下：
    - train/spoofing(欺骗数据)   or train/living(真人数据)
- lib
    - 共用的辅助文件，数据集，数据处理，模型训练
- model
    - 模型文件
- output
    - 保存输出模型和log的地方
- src
    - 主文件和辅助文件。训练代码和patch 生成代码就在这
- test
    - 用于单张照片的测试，可以用于模型的部署
- config.py
    - 配置文件，唯一需要修改的地方

# 使用手册

- patch_based_cnn
    - 首先，运行generate.py，根据论文将真人和欺骗人脸分为不同的patch，并将它们另存为数据集中的测试集和训练集。
     在本环境，用户需要准备自己的数据或公共数据集。 我们没有上传数据。 您需要将路径修改为自己的数据路径。
     - 修改配置文件
     - 训练和测试
    - 运行
        - - 划分patch
        ```
        cd src
        python3 data_generate.py
        ```
    
    - 训练和测试
        ```
        cd src
        python3 patch_cnn_main.py
        ```
    - 对其他的单张照片进行测试，可以用于做交叉测试，部署
        ```
        cd test 
        python patch_cnn_test.py
        ```
    - 结果
        - 以下结果没有舒勇诸如学习率衰减和混合等数据增强功能
    
        | Dataset    | Average accuracy(%) |
        | :----------| --- |
        | CASIA-FASD |  93.52 |
        | CASIA-SURF |  88.88 |
        
- depth_based_cnn

    - 首先，运行[PRNET](https://github.com/YadiraF/PRNet)生成真人人脸的深度图，并将它们另存为数据集中的测试集和训练集。
     在本环境，用户需要准备自己的数据或公共数据集。 我们没有上传数据。 您需要将路径修改为自己的数据路径。
     - 修改配置文件
     - 训练和测试

    - 运行

        - 训练和测试
            ```
            cd src
            python3 patch_cnn_main.py
            ```
        - 对其他的单张照片进行测试，可以用于做交叉测试，部署
            ```
            cd test 
            python patch_cnn_test.py
            ```