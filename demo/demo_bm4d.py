from bm4d import bm4d, BM4DProfileBM3DComplex
from bm3d import bm3d
import numpy as np
import time
import matplotlib.pyplot as plt

z = np.load('complex_image.npy')
z2 = np.stack([np.real(z), np.imag(z)], axis=2)

sigma = 0.01

print('Starting BM3D')
tic = time.time()
z2_denoise = bm3d(z2, sigma)
toc = time.time()
print('BM3D:  Elapsed time {:.2f} secs'.format(toc - tic))

print('Starting BM4D')
tic = time.time()
z_denoise = bm4d(z, sigma, profile=BM4DProfileBM3DComplex())
z_denoise = np.squeeze(z_denoise)
toc = time.time()
print('BM4D complex:  Elapsed time {:.2f} secs'.format(toc - tic))

z2_denoise_complex = np.squeeze(z2_denoise[:, :, 0]+1j*z2_denoise[:, :, 1])
nrmse = np.sqrt(np.sum(np.abs(z_denoise - z2_denoise_complex)**2)) / np.sqrt(np.sum(np.abs(z_denoise)))
print('NRMSE of bm3d relative to bm4d complex: {:.4f}'.format(nrmse))

plt.imshow(np.real(z2_denoise_complex))
plt.title('real with bm3d')
plt.show()
plt.imshow(np.imag(z2_denoise_complex))
plt.title('imag with bm3d')
plt.show()

plt.imshow(np.real(z_denoise))
plt.title('real with bm4d')
plt.show()
plt.imshow(np.imag(z_denoise))
plt.title('imag with bm4d')
plt.show()


