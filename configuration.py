# dirs
train_dir = r'./data_set/train'
test_dir = r'./data_set/test'
valid_dir = r'./data_set/valid'


num_classes = 2
img_size = (224, 224)
total_epochs = 1000
batch_size = 128


TRAIN_SET_RATIO = 0.8
TEST_SET_RATIO = 0.1
VALID_SET_RATIO = 0.1


is_training = True
is_validating = False
is_testing = False
is_save_weight = True
continued_training = False
is_distribution = False


model_name = "efficientnet-b0"
load_official_weight = False
load_downed_weight = True

