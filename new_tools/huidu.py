import numpy as np

def illumination_invariant_transform(image, alpha):
    """
    对RGB图像应用照明不变变换。

    参数:
    - image: 代表RGB图像的3D numpy数组。
    - alpha: 变换公式中的alpha参数。

    返回:
    - illumination_invariant_image: 代表照明不变灰度图像的2D numpy数组。
    """
    # 将图像分割为RGB通道
    R1 = image[:, :, 0]  # 红色通道
    R2 = image[:, :, 1]  # 绿色通道
    R3 = image[:, :, 2]  # 蓝色通道

    # 应用照明不变强度的公式进行计算
    I = np.log(R2 + 1e-8) - alpha * np.log(R1 + 1e-8) - (1 - alpha) * np.log(R3 + 1e-8)

    # 可选：将结果归一化到合适的灰度范围
    I = (I - np.min(I)) / (np.max(I) - np.min(I)) * 255.0
    illumination_invariant_image = I.astype(np.uint8)

    return illumination_invariant_image

# 示例使用，读取图像并给定alpha值
if __name__ == "__main__":
    import cv2
    # 读取图像
    image = cv2.imread("/home/qk/data/new_test/images/10_2073.jpg")  # 替换为你的图像路径
    alpha = 0.7  # 可以根据需要设置alpha值
    
    # 应用变换
    illumination_invariant_image = illumination_invariant_transform(image, alpha)

    # 保存或显示结果
    cv2.imwrite("/home/qk/data/new_test/10_2073.jpg", illumination_invariant_image)
    cv2.imshow("/home/qk/data/new_test/10_2073.jpg", illumination_invariant_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
