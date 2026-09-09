import numpy as np
import cv2


# gaborFunction was adapted from Wikipedia: https://en.wikipedia.org/wiki/Gabor_filter
# sigma: scale, theta: orientation, Lambda: wavelength, psi: phase offset, gamma: ellipticity ratio, b: half-response spatial frequency bandwidth (in octaves)

# Note: b is optional. If it is entered as argument in gaborTransfrom, sigma is fixed intrinsically in the function. Otherwise b is not functional, sigma and Lambda are independent and they should be fixed manually by the user.

def gaborFunction(theta, Lambda, psi, sigma, gamma=1):
    
    theta = -theta
    sigma_x = sigma
    sigma_y = sigma / gamma

    # Bounding box
    nstds = 3
    xmax = int(np.ceil(nstds * sigma_x))
    ymax = int(np.ceil(nstds * sigma_y))
    xmax = max(xmax, ymax)
    ymax = max(xmax, ymax)
    xmin, ymin = -xmax, -ymax

    y, x = np.meshgrid(
        np.arange(ymin, ymax + 1),
        np.arange(xmin, xmax + 1),
        indexing="ij"
    )

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Raw Gabor
    gabor = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi * x_theta / Lambda + psi)

    # ---- NORMALIZATION ----

    # (1) Remove DC component
    gabor -= np.mean(gabor)

    # (2) Unit L2 norm
    norm = np.sqrt(np.sum(gabor**2))
    if norm > 0:
        gabor /= norm

    return gabor


# Generate the Gabor filter bank to filter a given 2D image
def gaborFilterBank(Lambda, psi, sigma, nOfTheta, gamma=1, periodicity = np.pi):
    
    thetaList = np.arange(0, periodicity, periodicity/nOfTheta)
    out = [gaborFunction(theta, Lambda, psi, sigma, gamma) for theta in thetaList]
    
    return out


# Filter the given 2D image to lift it to the 3D sub-Riemannian space
def gaborTransform(imArray, Lambda, psi, sigma, nOfTheta, gamma=1, periodicity = np.pi):
        
    imSize = np.shape(imArray)[0]
    out = np.empty([nOfTheta,imSize, imSize], dtype=float) # initialize the output array
    filterBank = gaborFilterBank(Lambda, psi, sigma, nOfTheta) # generate the filter bank
    
    # filter the image for each orienation layer
    for i in range(0, nOfTheta):    
        kernel = filterBank[i]
        out[i] = cv2.filter2D(imArray, ddepth= -1, kernel=kernel)
                
    return out    
    
    
# Define the inverse Gabor transform    
def inverseGaborTransform(liftedArray, Lambda, psi, sigma, gamma=1, periodicity=np.pi):

    nOfTheta, imSize, _ = liftedArray.shape
    out = np.zeros((imSize, imSize), dtype=float)

    filterBank = gaborFilterBank(Lambda, psi, sigma, nOfTheta, gamma, periodicity)

    for i, kernel in enumerate(filterBank):
        out += cv2.filter2D(liftedArray[i], -1, kernel)

    # Orientation normalization
    out /= periodicity

    return out
