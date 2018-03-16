# json file does not allow comments, so the comment is bellow

{
  "batch_size": 128,  //
  "test_size": 300,
  "learning_rate": 0.001, // 学习率
  "expected_accuracy": 0.95,  // 期望达到的识别率
  "exp_name": "image_classfication", // 项目的名称，供选择Model和图的路径用
  "train_data_dir": "data/train.tfrecords", // 训练数据
  "test_data_dir": "data/test.tfrecords",  // 测试数据
  "log_dir": "./logpath", // Log地址
  "image_size": [28,28], // 图片size， 手写是28*28， 人脸是128*128
  "label_size": 10, // label个数， 手写识别是10， 人脸是人的个数 44
  "max_to_keep":5, // 保存Model的最大个数
  "num_epochs": 10, // 回合数
  "num_iter_per_epoch": 20, // 每个回合的步数
  "dir_af":"test_num" // 用于验证的图片路径
}