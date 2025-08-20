import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 导入层次映射关系，用于数据增强后的一致性检查
from hierarchy_dict import map_3_to_2, map_2_to_1


def random_rot_flip_images(image, label, is_rot=True):
    if is_rot:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a


def RGBtoHSVTransform(image, hue=.1, sat=1.2, val=1.2):
    '''
        对图像进行HSV变换
        :param
        image: numpy, b,g,r
        :return: 有颜色色差的图像image
        '''
    image = image[:, :, ::-1].copy()
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(image / 255.)  # RGB
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image = hsv_to_rgb(x) * 255  # 0 to 1
    return image.astype(np.uint8)[:, :, ::-1].copy()


def random_rot_flip_images_both(image, label_checked, label_uncheck, is_rot=True):
    if is_rot:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label_checked = np.rot90(label_checked, k)
        label_uncheck = np.rot90(label_uncheck, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label_checked = np.flip(label_checked, axis=axis).copy()
    label_uncheck = np.flip(label_uncheck, axis=axis).copy()
    return image, label_checked, label_uncheck


def random_rot_flip_images_triple(image, primary_label, secondary_label, tertiary_label, is_rot=True):
    """
    对图像、一级标签、二级标签和三级标签同时进行旋转和翻转操作
    
    参数:
        image: 输入图像，形状为 [H, W, C]
        primary_label: 一级标签，形状为 [H, W]
        secondary_label: 二级标签，形状为 [H, W]
        tertiary_label: 三级标签，形状为 [H, W]
        is_rot: 是否进行旋转，默认为 True
        
    返回:
        增强后的图像和三级标签
    """
    if is_rot:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        primary_label = np.rot90(primary_label, k)
        secondary_label = np.rot90(secondary_label, k)
        tertiary_label = np.rot90(tertiary_label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    primary_label = np.flip(primary_label, axis=axis).copy()
    secondary_label = np.flip(secondary_label, axis=axis).copy()
    tertiary_label = np.flip(tertiary_label, axis=axis).copy()
    
    return image, primary_label, secondary_label, tertiary_label


def check_label_consistency(secondary_label, tertiary_label):
    """
    检查二级标签和三级标签之间的一致性
    
    参数:
        secondary_label: 二级标签，形状为 [H, W]
        tertiary_label: 三级标签，形状为 [H, W]
        
    返回:
        一致性掩码，形状为 [H, W]，值为 1 表示一致，值为 0 表示不一致
    """
    # 创建映射数组
    max_key = max(map_3_to_2.keys())
    mapping = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in map_3_to_2.items():
        mapping[k] = v
    
    # 将三级标签映射到二级标签
    mapped_secondary = mapping[tertiary_label]
    
    # 检查一致性
    consistency_mask = (mapped_secondary == secondary_label).astype(np.uint8)
    
    return consistency_mask


def apply_data_augmentation(image, primary_label, secondary_label, tertiary_label, 
                           apply_rot_flip=True, apply_hsv=True, rot_flip_prob=0.8, hsv_prob=0.5):
    """
    应用数据增强到图像和标签
    
    参数:
        image: 输入图像，形状为 [H, W, C]
        primary_label: 一级标签，形状为 [H, W]
        secondary_label: 二级标签，形状为 [H, W]
        tertiary_label: 三级标签，形状为 [H, W]
        apply_rot_flip: 是否应用旋转和翻转，默认为 True
        apply_hsv: 是否应用 HSV 变换，默认为 True
        rot_flip_prob: 应用旋转和翻转的概率，默认为 0.8
        hsv_prob: 应用 HSV 变换的概率，默认为 0.5
        
    返回:
        增强后的图像和标签
    """
    # 旋转和翻转 (概率控制)
    if apply_rot_flip and np.random.random() < rot_flip_prob:
        image, primary_label, secondary_label, tertiary_label = random_rot_flip_images_triple(
            image, primary_label, secondary_label, tertiary_label
        )
    
    # HSV 变换 (仅应用于图像，概率控制)
    if apply_hsv and np.random.random() < hsv_prob:
        image = RGBtoHSVTransform(image)
    
    # 检查标签一致性
    consistency_mask = check_label_consistency(secondary_label, tertiary_label)
    
    # 如果存在不一致，可以选择修复或记录
    if np.sum(1 - consistency_mask) > 0:
        # 创建映射数组
        max_key = max(map_3_to_2.keys())
        mapping = np.zeros(max_key + 1, dtype=np.int32)
        for k, v in map_3_to_2.items():
            mapping[k] = v
        # 根据三级标签更新二级标签
        secondary_label = mapping[tertiary_label]
    
    return image, primary_label, secondary_label, tertiary_label
