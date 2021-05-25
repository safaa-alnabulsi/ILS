# point spread function (PSF) is impulse response, describes the response of an imaging system to a point source or point object.
# optical transfer function (OTF) is defined as the Fourier transform of the point spread function 
# ---------------------------------------------------------------------



def _fft(img, interface='single'):
    if interface == 'single':
        out = np.fft.fft2(img,axes=(0,1))
    elif interface == 'parallel_numpy':
        out = pyfftw.interfaces.numpy_fft.fft2(img,axes=(0,1))
    elif interface == 'parallel_scipy':
        out = pyfftw.interfaces.scipy_fftpack.fft2(img,axes=(0,1)) 
    return out

def _ifft(img, interface='single'):
    if interface == 'single':
        out = np.fft.ifft2(img,axes=(0,1)).real
    elif interface == 'parallel_numpy':
        out = pyfftw.interfaces.numpy_fft.ifft2(img,axes=(0,1)).real
    elif interface == 'parallel_scipy':
        out = pyfftw.interfaces.scipy_fftpack.ifft2(img,axes=(0,1)).real
    return out


def psf2otf_Dx(outSize, interface):
    psf = np.zeros(outSize)
    psf = psf.astype('float32')
    psf[0, 0] = -1 
    psf[0, -1] = 1 
    otf =  _fft(psf, interface)
    return otf

def psf2otf_Dy(outSize, interface):
    psf = np.zeros(outSize)
    psf = psf.astype('float32')
    psf[0, 0] = -1 
    psf[-1, 0] = 1 
    otf =  _fft(psf, interface)
    return otf

def ILS_Norm(F,c,lam, interface='single'):
    N, M, D = F.shape # (768, 1024, 3)
    sizeI2D = [N, M]

    # ----------------------------------------------------------------------------------------------------------

    # pre-compute
    otfFx = psf2otf_Dx(sizeI2D,interface)
    otfFy = psf2otf_Dy(sizeI2D,interface)

    Denormin = np.abs(otfFx)**2 + np.abs(otfFy)**2  # take the real part and to power of two
    Denormin = Denormin[:,:, np.newaxis] # add a third axis
    Denormin = np.repeat(Denormin, D, axis=2) # repeat the same data D=3 times on the third axis=2

    denominator = 1 + 0.5 * c * lam * Denormin
    
    # ----------------------------------------------------------------------------------------------------------
    U = F
    normin1 =  _fft(U, interface)
    # ----------------------------------------------------------------------------------------------------------

    for i in range(4):
        ## Step 1:  eq 7 - Intermediate variables \mu update, in x-axis and y-axis direction

        # Gradients delta u on x axis and delta u on y axis

        # image gradient in x direction
        u_extra_col = (U[:,0,:] - U[:,-1,:])[:,np.newaxis,:]
        u_extra_row = (U[0,:,:] - U[-1,:,:])[np.newaxis,:,:]

        u_h = np.hstack((np.diff(U,1,1), u_extra_col))
        u_v = np.vstack((np.diff(U,1,0), u_extra_row))


        mu_h = c * u_h - p * u_h * (u_h * u_h + eps)**gamma
        mu_v = c * u_v - p * u_v * (u_v * u_v + eps)**gamma 


        ### ---------------------------------------------------------------------------- ##
        ## Step 2: eq 9 - Update the smoothed image U

        # The diff causes loss in one columns
        extra_col = (mu_h[:,-1,:] - mu_h[:, 0,:])[:,np.newaxis,:]
        extra_row = (mu_v[-1,:,:] - mu_v[0,:,:])[np.newaxis,:,:]    

        # we calculate the diff - the inverse first order derivative of μxn along x-axis & μyn along y-axis 
        normin2_h = np.hstack((extra_col , - np.diff(mu_h,1,1)))
        normin2_v = np.vstack((extra_row , - np.diff(mu_v,1,0)))
        
        fft_normin_h_v = _fft(normin2_h + normin2_v, interface)

        numerator = normin1 + 0.5 * lam * fft_normin_h_v

        FU = numerator / denominator
        U =  _ifft(FU, interface)

        # U = np.abs(ifft2(FU,axes=(0,1)))
        normin1 = FU
    
    return U


def show_image(img,t=''):
    plt.figure()
    plt.imshow(img)
    plt.title(t)
    
# Read the image and normalize its values between zero and one
def read_image(image_fulll_path):
    image = Image.open(image_fulll_path) # read the image using PIL
    im = np.asarray(image) # convert the image into numpy array
    info = np.iinfo(im.dtype) # get the data type of the input image
    F = im.astype('float32') / info.max # divide all values by the largest possible value in the datatype
    return F

    