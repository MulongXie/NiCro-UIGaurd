import cv2
import numpy as np
from os.path import join as pjoin
from glob import glob
from tqdm import tqdm


class Data:
    def __init__(self, data_dir=None, cls=None):
        '''
        :param data_dir: Data root directory
        :param cls: 'compo' or 'icon', the type of classification
        '''
        self.cls_type = cls
        if cls == 'compo':
            self.data_dir = '/home/ml/Data/rico/component'
            self.class_map = ['Text Button', 'Input', 'Switch', 'Image', 'Icon', 'Checkbox']
        elif cls == 'icon':
            self.data_dir = '/home/ml/Data/rico/icon'
            self.class_map = [cls.replace('\\', '/').split('/')[-1] for cls in glob(pjoin(self.data_dir, '*'))]
        else:
            self.data_dir = data_dir
            self.class_map = []
        self.class_number = len(self.class_map)
        self.class_dirs = glob(pjoin(self.data_dir, '*')) if self.data_dir is not None else None

        self.image_shape = (32, 32, 3)
        self.images = []
        self.labels = []
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None

    def get_class_map_from_directory(self):
        classes = glob(pjoin(self.data_dir, '*'))
        self.class_map = [cls.replace('\\', '/').split('/')[-1] for cls in classes]
        self.class_number = len(self.class_map)
        self.count_data_in_data_dir()

    def count_data_in_data_dir(self, match_cls_map=True):
        '''
        Check the data amount in each category under the data directory
        '''
        for class_dir in self.class_dirs:
            c_dir = pjoin(class_dir, '*')
            if match_cls_map:
                if class_dir.replace('\\', '/').split('/')[-1] in self.class_map:
                    print(len(glob(c_dir)), '\t', class_dir)
            else:
                print(len(glob(c_dir)), '\t', class_dir)

    def view_images_in_class(self, class_name, batch_num=5):
        class_dir = pjoin(self.data_dir, class_name)
        image_files = glob(pjoin(class_dir, '*'))
        i = 0
        num = len(image_files)
        while i < num:
            for k, j in enumerate(range(i, i + batch_num)):
                img = cv2.imread(image_files[j])
                cv2.imshow(class_name + str(k), img)
            key = cv2.waitKey()
            if key == ord('q'):
                break
            i += batch_num
        cv2.destroyAllWindows()

    def load_data_in_class_map(self):
        for class_dir in self.class_dirs:
            cls = class_dir.replace('\\', '/').split('/')[-1]
            if cls in self.class_map:
                label = self.class_map.index(cls)
                img_files = glob(pjoin(class_dir, '*'))
                print('Total Images: %d; \t Load image in class of %s [%d]' % (len(img_files), cls, label))
                for img_file in tqdm(img_files):
                    img = cv2.imread(img_file)
                    img = cv2.resize(img, self.image_shape[:2])
                    self.images.append(img)
                    self.labels.append(label)

    def generate_training_data(self, train_data_ratio=0.8):
        # transfer int into c dimensions one-hot array
        def expand(label, class_number):
            # return y : (num_class, num_samples)
            y = np.eye(class_number)[label]
            y = np.squeeze(y)
            return y

        # reshuffle
        np.random.seed(0)
        self.images = np.random.permutation(self.images)
        np.random.seed(0)
        self.labels = np.random.permutation(self.labels)
        Y = expand(self.labels, self.class_number)

        # separate dataset
        cut = int(train_data_ratio * len(self.images))
        self.X_train = (self.images[:cut] / 255).astype('float32')
        self.X_test = (self.images[cut:] / 255).astype('float32')
        self.Y_train = Y[:cut]
        self.Y_test = Y[cut:]
        print('X_train:%d, Y_train:%d' % (len(self.X_train), len(self.Y_train)))
        print('X_test:%d, Y_test:%d' % (len(self.X_test), len(self.Y_test)))
