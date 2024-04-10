import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff



def calculate_iou(predict_image, label_image):
    """
    计算IoU（Intersection over Union）
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: IoU值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), label_image))  # 假反例（False Negative）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    iou = tp / (tp + fn + fp + 1e-7)  # 计算IoU
    return iou


def calculate_accuracy(predict_image, label_image):
    """
    计算准确率
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: 准确率值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(label_image)))  # 真负例（True Negative）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), label_image))  # 假负例（False Negative）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)  # 计算准确率值
    return accuracy


def calculate_precision(predict_image, label_image):
    """
    计算精确度
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: 精确度值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    precision = tp / (tp + fp + 1e-7)  # 计算精确度值
    return precision


def calculate_recall(predict_image, label_image):
    """
    计算召回率
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: 召回率值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), label_image))  # 假负例（False Negative）

    recall = tp / (tp + fn + 1e-7)  # 计算召回率值
    return recall


def calculate_dice_coefficient(predict_image, label_image):
    """
    计算Dice系数
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: Dice系数值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), label_image))  # 假负例（False Negative）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)  # 计算Dice系数值
    return dice_coefficient


def calculate_f1(predict_image, label_image):
    """
    计算F1分数
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: F1分数值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tp = np.sum(np.logical_and(predict_image, label_image))  # 真正例（True Positive）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), label_image))  # 假负例（False Negative）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    precision = tp / (tp + fp + 1e-7)  # 计算精确度值
    recall = tp / (tp + fn + 1e-7)  # 计算召回率值

    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # 计算F1分数值
    return f1


def calculate_specificity(predict_image, label_image):
    """
    计算特异性
    :param predict_image: 预测图像，类型为numpy数组
    :param label_image: 标签图像，类型为numpy数组
    :return: 特异性值，类型为float
    """
    predict_image = np.array(predict_image, dtype = bool)
    label_image = np.array(label_image, dtype = bool)

    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(label_image)))  # 真负例（True Negative）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(label_image)))  # 假正例（False Positive）

    specificity = tn / (tn + fp + 1e-7)  # 计算特异性值
    return specificity

def calculate_hausdorff_distance(predict_image, label_image):
    predict_image = np.array(predict_image, dtype=bool)
    label_image = np.array(label_image, dtype=bool)

    hd1 = directed_hausdorff(predict_image, label_image)[0]
    hd2 = directed_hausdorff(label_image, predict_image)[0]
    hausdorff_distance = max(hd1, hd2)
    return hausdorff_distance


def calculate_all_for_images(pred_folder, label_folder):
    """
    遍历计算分割结果文件夹和真实标签文件夹中的图像，并将图像名与对应的参数存储。
    :param pred_folder: 分割结果文件夹路径，类型为字符串。
    :param label_folder: 真实标签文件夹路径，类型为字符串。
    :return: 包含图像名和对应数的字典，类型为dict。
    """
    # 初始化各个指标的字典
    iou_dict, accuracy_dict, precision_dict, recall_dict, dice_coefficient_dict, f1_dict, specificity_dict,hausdorff_distance_dict ={}, {}, {}, {}, {}, {}, {}, {}

    # 遍历预测结果文件夹
    for filename in os.listdir(pred_folder):
        if filename.endswith('.png'):
            # 获取预测结果和真实标签的路径
            pred_path = os.path.join(pred_folder, filename)
            label_path = os.path.join(label_folder, filename)

            # 读取图像并转换为numpy数组
            pred_img = np.array(Image.open(pred_path))
            label_img = np.array(Image.open(label_path))

            # 计算各个指标并存储到对应的字典中
            metrics = [calculate_iou, calculate_accuracy, calculate_precision, calculate_recall,
                       calculate_dice_coefficient, calculate_f1, calculate_specificity,calculate_hausdorff_distance]
            result_dicts = [iou_dict, accuracy_dict, precision_dict, recall_dict, dice_coefficient_dict, f1_dict,
                            specificity_dict,hausdorff_distance_dict]
            for metric, result_dict in zip(metrics, result_dicts):
                value = metric(pred_img, label_img)
                result_dict[filename] = "{:.4f}".format(round(value, 4))

    return iou_dict, accuracy_dict, precision_dict, recall_dict, dice_coefficient_dict, f1_dict, specificity_dict,hausdorff_distance_dict


pred_folder = 'D:/xuanxiujie/Desktop/PyStudy/SAR/test/img_out'
label_folder = 'D:/xuanxiujie/Desktop/PyStudy/SAR/test/labels'
iou_dict, accuracy_dict, precision_dict, recall_dict, dice_coefficient_dict, f1_dict, specificity_dict,hausdorff_distance_dict = calculate_all_for_images(
    pred_folder, label_folder)

print("iou_dict\n", iou_dict)
print("accuracy_dict\n", accuracy_dict)
print("precision_dict\n", precision_dict)
print("recall_dict\n", recall_dict)
print("dice_coefficient_dict\n", dice_coefficient_dict)
print("f1_dict\n", f1_dict)
print("specificity_dict\n", specificity_dict)
print("hausdorff_distance_dict\n", hausdorff_distance_dict)
