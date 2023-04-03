
import os
import random
import shutil
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 /Result
    :param target_data_folder: 目标文件夹 /to/data/
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    split_names = ['train', 'val', 'test']

    # 在目标目录下创建类别文件夹
    for class_name in class_names:
        class_split_path = os.path.join(target_data_folder, class_name)
        if os.path.exists(class_split_path):
            pass
            # shutil.rmtree(class_split_path)
        else:
            os.mkdir(class_split_path)
        # 然后在 类别文件夹下创建 'train'/'val'/'test'文件夹
        for split_name in split_names:
            split_path = os.path.join(class_split_path, split_name)
            if os.path.exists(split_path):
                # pass
                # 如果该文件夹本来存在，则删除该文件夹下所有文件
                shutil.rmtree(split_path)
            os.mkdir(split_path)


    # 按照比例划分数据集，并进行数据图片的复制
    # 首先对A进行分类遍历，同时相应的将B的源文件夹中的文件放入B的目标文件中
    A_class_data_path = os.path.join(src_data_folder, 'A')
    B_class_data_path = os.path.join(src_data_folder, 'B')
    A_all_data = os.listdir(A_class_data_path)
    A_data_length = len(A_all_data)
    A_data_index_list = list(range(A_data_length))
    random.shuffle(A_data_index_list)

    A_train_folder = os.path.join(os.path.join(
        target_data_folder, 'A'), 'train')
    A_val_folder = os.path.join(os.path.join(
        target_data_folder, 'A'), 'val')
    A_test_folder = os.path.join(os.path.join(
        target_data_folder, 'A'), 'test')

    B_train_folder = os.path.join(os.path.join(
        target_data_folder, 'B'), 'train')
    B_val_folder = os.path.join(os.path.join(
        target_data_folder, 'B'), 'val')
    B_test_folder = os.path.join(os.path.join(
        target_data_folder, 'B'), 'test')

    train_stop_flag = A_data_length * train_scale
    val_stop_flag = A_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in A_data_index_list:
        A_src_img_path = os.path.join(
            A_class_data_path, A_all_data[i])
        B_src_img_path = os.path.join(
            B_class_data_path, A_all_data[i])

        if current_idx <= train_stop_flag:
            copy2(A_src_img_path, A_train_folder)
            copy2(B_src_img_path, B_train_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(A_src_img_path, A_val_folder)
            copy2(B_src_img_path, B_val_folder)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
        else:
            copy2(A_src_img_path, A_test_folder)
            copy2(B_src_img_path, B_test_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1

    print("A类按照{}：{}：{}的比例划分完成，一共{}张图片".format(
        train_scale, val_scale, test_scale, A_data_length))
    print("训练集{}：{}张".format(A_train_folder, train_num))
    print("验证集{}：{}张".format(A_val_folder, val_num))
    print("测试集{}：{}张".format(A_test_folder, test_num))
    print("B 类的训练集、验证集、测试集完全按照 A 类的文件名称对应分类！")


if __name__ == '__main__':
    src_data_folder = "Result"
    target_data_folder = "to/data"

    # 如果目标目录不存在，则创建该目录。
    if os.path.exists(target_data_folder):
        pass
    else:
        os.makedirs(target_data_folder)
    data_set_split(r'./datasets/tar_data', r'./datasets/tarrr_data')

