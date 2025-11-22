import numpy as np
import cv2
import matplotlib.pyplot as plt

def adaptive_filter(image, delta):
    h, w, c = image.shape
    freq_x = np.fft.fftfreq(w)
    freq_y = np.fft.fftfreq(h)
    freq = np.sqrt(freq_x[:, np.newaxis]**2 + freq_y[np.newaxis, :]**2)
    z = np.fft.fft2(image)
    freq_mask = np.power(freq, -delta, where=(freq != 0))
    filtered_image = z * freq_mask[..., np.newaxis]
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.clip(np.real(filtered_image), 0, 255).astype(np.uint8)
    return filtered_image

def generate_mask(image, threshold=0.5):
    grayscale = np.mean(image, axis=2)
    mask = grayscale > (threshold * 255)
    return mask.astype(np.float32)

def mixup_adaptive(im1, im2, delta=1):
    filtered_im1 = adaptive_filter(im1, delta)
    filtered_im2 = adaptive_filter(im2, delta)
    mask = generate_mask(filtered_im1)
    mixed_image = (mask[..., None] * filtered_im1 + (1 - mask[..., None]) * filtered_im2).astype(np.uint8)
    return mixed_image, mask

def mixup_basic(im1, im2):
    r = np.random.beta(32.0, 32.0)
    mixed_image = (im1 * r + im2 * (1 - r)).astype(np.uint8)
    return mixed_image

# 加载图像（替换为你的图像路径）
a = cv2.imread('/home/qk/data/new_test/images/10_2073.jpg')
b = cv2.imread("/home/qk/data/new_test/images/11_2030.jpg")

# 调整图像大小以确保相同
a = cv2.resize(a, (640, 640))
b = cv2.resize(b, (640, 640))

# 应用自适应MixUp
mixed_image_adaptive, mask_adaptive = mixup_adaptive(a, b)

# 应用基本MixUp
mixed_image_basic = mixup_basic(a, b)

# 保存混合图像到当前目录
cv2.imwrite('mixed_adaptive.jpg', mixed_image_adaptive)
cv2.imwrite('mixed_basic.jpg', mixed_image_basic)

# 保存生成的掩码
cv2.imwrite('mask_adaptive.jpg', (mask_adaptive * 255).astype(np.uint8))  # 转换为uint8

# 可视化对比
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
plt.title('Image A')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(mixed_image_adaptive, cv2.COLOR_BGR2RGB))
plt.title('Adaptive MixUp')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(mixed_image_basic, cv2.COLOR_BGR2RGB))
plt.title('Basic MixUp')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(mask_adaptive, cmap='gray')
plt.title('Adaptive Mask')
plt.axis('off')

plt.show()
