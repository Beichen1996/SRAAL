# State-Relabeling Adversarial Active Learning
 Code for SRAAL [2020 CVPR Oral]

# Requirements
 torch >= 1.6.0

 numpy >= 1.19.1

 tqdm >= 4.31.1

# AL Results
  The AL sampling starts from 10% initial labeled pool(10.npy) and selects 5% data to label at each iteration.

  The result files locate in ./results_cifar100/





# To Train the Model

  python main.py

# To Evaluate the Results

  python acc100.py

