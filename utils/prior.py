 from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfile2D, BM4DProfileComplex, BM4DProfileBM3DComplex


 def denoise_cmplx_bm3d(cmplx_img, psd=0.1):
     """
     Denoise single complex image using complex bm4d software.
     :param cmplx_img: noisy complex image.
     :param psd: standard deviation of noise.
     :return: denoised complex image.
     """

     denoised_img = bm4d.bm4d(cmplx_img, psd, profile=BM4DProfileBM3DComplex())
     output = denoised_img[:, :, 0]

     return output

