import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import os

# Question 1 

cameraman_image_8bit = imread(('cameraman.tif'))

image_array = np.array(cameraman_image_8bit)
scaled_image = image_array / 255.0

output_path = 'output_3_noise_power_(7x7)_sigma(0.5)'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def save_images(image, title, filename, output_path):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    image_path = os.path.join(output_path, f'{filename}.png')
    plt.savefig(image_path)
    plt.close()



print('Cameraman Scaled: ', scaled_image)
save_images(image_array, 'Original 8-bit Image', 'cameraman_image_original', output_path)
save_images(scaled_image, 'Scaled Image (0 to 1)', 'cameraman_image_scaled', output_path)


# Question 2

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)

    noisy_image = np.clip(image + noise, 0, 1)

    return noisy_image

sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
noisy_images = []

for sigma in sigma_values:
    noisy_image = add_gaussian_noise(scaled_image, sigma)
    noisy_images.append(noisy_image)

    save_images(noisy_image, f'Noisy Image (σ = {sigma})', f'noisy_image_sigma_{sigma}', output_path)
plt.figure(figsize=(15, 8))
for i, (sigma, nc) in enumerate(zip(sigma_values, noisy_image)):
    plt.subplot(2,3, i+1)
    plt.title(f'Nosiy Image (σ = {sigma})')
    plt.imshow(nc, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Question 3
from scipy.signal import wiener

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1)

def apply_wiener_filter(noisy_image, kernel_size, noise_power=None):
    epsilon = 1e-6 
    effective_noise_power = noise_power if noise_power is not None else epsilon
    return wiener(noisy_image, (kernel_size, kernel_size), noise=effective_noise_power)

sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
kernel_sizes = [3, 5, 7]
noise_powers = [0.001, 0.01, 0.1]

noisy_images = [add_gaussian_noise(scaled_image, sigma) for sigma in sigma_values]

for sigma, noisy_image in zip(sigma_values, noisy_images):
    for kernel_size in kernel_sizes:
        denoised_image = apply_wiener_filter(noisy_image, kernel_size)
        filename = f'denoised_sigma_{sigma}_kernel_{kernel_size}.png'
        title = f'Denoised (σ={sigma}, Kernel={kernel_size}x{kernel_size})'
        save_images(denoised_image, title, filename, output_path)

sigma_to_test = 0.5
if sigma_to_test in sigma_values:
    noisy_image = noisy_images[sigma_values.index(sigma_to_test)]
    for noise_power in noise_powers:
        denoised = apply_wiener_filter(noisy_image, kernel_size=7, noise_power=noise_power)
        filename = f'denoised_sigma_{sigma_to_test}_noise_power_{noise_power}.png'
        title = f'Denoised (σ={sigma_to_test}, Noise Power={noise_power})'
        save_images(denoised, title, filename, output_path)
else:
    print(f"Error: σ={sigma_to_test} not in sigma_values")

























