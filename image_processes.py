from PIL import ImageTk, Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np


def Saturation(image):
    """
    Adjust the saturation of an image.

    Parameters:
    - image: PIL Image object
    - factor: float, saturation factor (0.0 to 1.0)

    Returns:
    - PIL Image object with adjusted saturation
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.5)

def HistogramEqualization(image):
    """
    Apply histogram equalization to an image.

    Parameters:
    - image: PIL Image object

    Returns:
    - PIL Image object with histogram equalization applied
    """
    image = image.convert("L")  # Convert to grayscale
    equalized_image = ImageOps.equalize(image)
    return equalized_image


def GammaCorrection(image, gamma):
    """
    Apply gamma correction to an image.

    Parameters:
    - image: PIL Image object
    - gamma: float, gamma value for correction

    Returns:
    - PIL Image object with gamma correction applied
    """
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = [int(value) for value in table]

    if image.mode in ['L', 'P']:  # Grayscale or palette-based image
        return image.point(table)
    elif image.mode == 'RGB':  # RGB image
        r, g, b = image.split()
        r = r.point(table)
        g = g.point(table)
        b = b.point(table)
        return Image.merge('RGB', (r, g, b))
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    

def BilateralFiltering(image, diameter, sigma_color, sigma_space):
    """
    Apply bilateral filtering to an image.

    Parameters:
    - image: PIL Image object
    - diameter: int, diameter of each pixel neighborhood
    - sigma_color: float, filter sigma in the color space
    - sigma_space: float, filter sigma in the coordinate space

    Returns:
    - PIL Image object with bilateral filtering applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply bilateral filter
    filtered_image_cv = cv2.bilateralFilter(image_cv, diameter, sigma_color, sigma_space)

    # Convert back to PIL Image
    filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_cv, cv2.COLOR_BGR2RGB))
    return filtered_image


def GaussianBlur(image, kernel_size, sigma):

    """
    Apply Gaussian blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the Gaussian kernel (must be odd)
    - sigma: float, standard deviation of the Gaussian distribution

    Returns:
    - PIL Image object with Gaussian blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur
    blurred_image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image


def MedianBlur(image, kernel_size):
    """
    Apply median blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the kernel (must be odd)

    Returns:
    - PIL Image object with median blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply median blur
    blurred_image_cv = cv2.medianBlur(image_cv, kernel_size)

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image


def AverageBlur(image, kernel_size):
    """
    Apply average blur to an image.

    Parameters:
    - image: PIL Image object
    - kernel_size: int, size of the kernel (must be odd)

    Returns:
    - PIL Image object with average blur applied
    """

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply average blur
    blurred_image_cv = cv2.blur(image_cv, (kernel_size, kernel_size))

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image




TECHNIQUES = {
    "Histogram Equalization": HistogramEqualization,
    "Gaussian Blurring": GaussianBlur,
    "Saturation": Saturation,
    "Gamma Correction": GammaCorrection,
    "Bilateral Filtering": BilateralFiltering,
    "Median Filtering": MedianBlur,
    "Average Filtering": AverageBlur,
}