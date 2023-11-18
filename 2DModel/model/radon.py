import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
from torch.autograd import Variable
# from scipy.integrate import quad
from Config_doc.logger import get_logger
from torch.nn.modules.loss import _Loss
import numpy as np
# import astra
try:
    import torch_radon_cuda
    from torch_radon_cuda import RaysCfg
except Exception as e:
    print("Importing exception")

# from skimage.transform import radon
# from torch_radon import Radon, RadonFanbeam

#

logger = get_logger('loss')
import numpy as np
import torch

try:
    from _abc import (get_cache_token, _abc_init, _abc_register,
                      _abc_instancecheck, _abc_subclasscheck, _get_dump,
                      _reset_registry, _reset_caches)
except ImportError:
    from _py_abc import ABCMeta, get_cache_token
    ABCMeta.__module__ = 'abc'
else:
    class ABCMeta(type):
        """Metaclass for defining Abstract Base Classes (ABCs).

        Use this metaclass to create an ABC.  An ABC can be subclassed
        directly, and then acts as a mix-in class.  You can also register
        unrelated concrete classes (even built-in classes) and unrelated
        ABCs as 'virtual subclasses' -- these and their descendants will
        be considered subclasses of the registering ABC by the built-in
        issubclass() function, but the registering ABC won't show up in
        their MRO (Method Resolution Order) nor will method
        implementations defined by the registering ABC be callable (not
        even via super()).
        """
        def __new__(mcls, name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, name, bases, namespace, **kwargs)
            _abc_init(cls)
            return cls

        def register(cls, subclass):
            """Register a virtual subclass of an ABC.

            Returns the subclass, to allow usage as a class decorator.
            """
            return _abc_register(cls, subclass)

        def __instancecheck__(cls, instance):
            """Override for isinstance(instance, cls)."""
            return _abc_instancecheck(cls, instance)

        def __subclasscheck__(cls, subclass):
            """Override for issubclass(subclass, cls)."""
            return _abc_subclasscheck(cls, subclass)

        def _dump_registry(cls, file=None):
            """Debug helper to print the ABC registry."""
            print(f"Class: {cls.__module__}.{cls.__qualname__}", file=file)
            print(f"Inv. counter: {get_cache_token()}", file=file)
            (_abc_registry, _abc_cache, _abc_negative_cache,
             _abc_negative_cache_version) = _get_dump(cls)
            print(f"_abc_registry: {_abc_registry!r}", file=file)
            print(f"_abc_cache: {_abc_cache!r}", file=file)
            print(f"_abc_negative_cache: {_abc_negative_cache!r}", file=file)
            print(f"_abc_negative_cache_version: {_abc_negative_cache_version!r}",
                  file=file)

        def _abc_registry_clear(cls):
            """Clear the registry (for debugging or testing)."""
            _reset_registry(cls)

        def _abc_caches_clear(cls):
            """Clear the caches (for debugging or testing)."""
            _reset_caches(cls)


class ABC(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()


try:
    import scipy.fft

    fftmodule = scipy.fft
except ImportError:
    import numpy.fft

    fftmodule = numpy.fft


class FourierFilters:
    def __init__(self):
        self.cache = dict()

    def get(self, size: int, filter_name: str, device):
        key = (size, filter_name)

        if key not in self.cache:
            ff = torch.FloatTensor(self.construct_fourier_filter(size, filter_name)).view(1, -1, 1).to(device)
            self.cache[key] = ff

        return self.cache[key].to(device)

    @staticmethod
    def construct_fourier_filter(size, filter_name):
        """Construct the Fourier filter.

        This computation lessens artifacts and removes a small bias as
        explained in [1], Chap 3. Equation 61.

        Parameters
        ----------
        size: int
            filter size. Must be even.
        filter_name: str
            Filter used in frequency domain filtering. Filters available:
            ram-lak (ramp), shepp-logan, cosine, hamming, hann.

        Returns
        -------
        fourier_filter: ndarray
            The computed Fourier filter.

        References
        ----------
        .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
               Imaging", IEEE Press 1988.

        """
        filter_name = filter_name.lower()

        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                            np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2

        # Computing the ramp filter from the fourier transform of its
        # frequency domain representation lessens artifacts and removes a
        # small bias as explained in [1], Chap 3. Equation 61
        fourier_filter = 2 * np.real(fftmodule.fft(f))  # ramp filter
        if filter_name == "ramp" or filter_name == "ram-lak":
            pass
        elif filter_name == "shepp-logan":
            # Start from first element to avoid divide by zero
            omega = np.pi * fftmodule.fftfreq(size)[1:]
            fourier_filter[1:] *= np.sin(omega) / omega
        elif filter_name == "cosine":
            freq = np.linspace(0, np.pi, size, endpoint=False)
            cosine_filter = fftmodule.fftshift(np.sin(freq))
            fourier_filter *= cosine_filter
        elif filter_name == "hamming":
            fourier_filter *= fftmodule.fftshift(np.hamming(size))
        elif filter_name == "hann":
            fourier_filter *= fftmodule.fftshift(np.hanning(size))
        else:
            print(
                f"[TorchRadon] Error, unknown filter type '{filter_name}', available filters are: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'")

        return fourier_filter


def _normalize_shape(x, d):
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape


def _unnormalize_shape(y, old_shape):
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]

    return y


def normalize_shape(d):
    """
    Input with shape (batch_1, ..., batch_n, s_1, ..., s_d) is reshaped to (batch, s_1, s_2, ...., s_d)
    fed to f and output is reshaped to (batch_1, ..., batch_n, s_1, ..., s_o).
    :param d: Number of non-batch dimensions
    """

    def wrap(f):
        def wrapped(self, x, *args, **kwargs):
            x, old_shape = _normalize_shape(x, d)

            y = f(self, x, *args, **kwargs)

            return _unnormalize_shape(y, old_shape)

        wrapped.__doc__ = f.__doc__
        return wrapped

    return wrap
class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, rays_cfg):
        with torch.no_grad():
            sinogram = torch_radon_cuda.forward(x, angles, tex_cache, rays_cfg)
            sinogram_cpu = sinogram.cpu()
            del sinogram
            # torch.cuda.empty_cache()
        # ctx.tex_cache = tex_cache
        # ctx.rays_cfg = rays_cfg
        # ctx.save_for_backward(angles)

        return sinogram_cpu

    @staticmethod
    def backward(ctx, grad_x):
        # if not grad_x.is_contiguous():
        #     grad_x = grad_x.contiguous()
        #
        # angles, = ctx.saved_variables
        # grad = torch_radon_cuda.backward(grad_x, angles, ctx.tex_cache, ctx.rays_cfg)
        return None, None, None, None#grad, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, rays_cfg):
        image = torch_radon_cuda.backward(x, angles, tex_cache, rays_cfg)
        ctx.tex_cache = tex_cache
        ctx.rays_cfg = rays_cfg
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        angles, = ctx.saved_variables
        grad = torch_radon_cuda.forward(grad_x, angles, ctx.tex_cache, ctx.rays_cfg)
        return grad, None, None, None
class BaseRadon(ABC):
    def __init__(self, angles, rays_cfg):
        self.rays_cfg = rays_cfg

        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        # change sign to conform to Astra and Scikit
        self.angles = -angles

        # caches used to avoid reallocation of resources
        self.tex_cache = torch_radon_cuda.TextureCache(8)
        self.fourier_filters = FourierFilters()

        seed = np.random.get_state()[1][0]
        self.noise_generator = torch_radon_cuda.RadonNoiseGenerator(seed)

    def _move_parameters_to_device(self, device):
        if device != self.angles.device:
            self.angles = self.angles.to(device)

    def _check_input(self, x, square=False):
        if not x.is_contiguous():
            x = x.contiguous()

        if square:
            assert x.size(1) == x.size(2), f"Input images must be square, got shape ({x.size(1)}, {x.size(2)})."

        if x.dtype == torch.float16:
            assert x.size(
                0) % 4 == 0, f"Batch size must be multiple of 4 when using half precision. Got batch size {x.size(0)}"

        return x

    @normalize_shape(2)
    def forward(self, x):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        """
        x = self._check_input(x, square=True)
        self._move_parameters_to_device(x.device)

        return RadonForward.apply(x, self.angles, self.tex_cache, self.rays_cfg)

    @normalize_shape(2)
    def backprojection(self, sinogram):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        """
        sinogram = self._check_input(sinogram)
        self._move_parameters_to_device(sinogram.device)

        return RadonBackprojection.apply(sinogram, self.angles, self.tex_cache, self.rays_cfg)

    @normalize_shape(2)
    def filter_sinogram(self, sinogram, filter_name="ramp"):
        # if not self.clip_to_circle:
        #     warnings.warn("Filtered Backprojection with clip_to_circle=True will not produce optimal results."
        #                   "To avoid this specify clip_to_circle=False inside Radon constructor.")

        # Pad sinogram to improve accuracy
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size

        padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))
        # TODO should be possible to use onesided=True saving memory and time
        sino_fft = torch.rfft(padded_sinogram, 1, normalized=True, onesided=False)

        # get filter and apply
        f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
        filtered_sino_fft = sino_fft * f

        # Inverse fft
        filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)

        # pad removal and rescaling
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)

    def backward(self, sinogram):
        r"""Same as backprojection"""
        return self.backprojection(sinogram)

    @normalize_shape(2)
    def add_noise(self, x, signal, density_normalization=1.0, approximate=False):
        # print("WARN Radon.add_noise is deprecated")

        torch_radon_cuda.add_noise(x, self.noise_generator, signal, density_normalization, approximate)
        return x

    @normalize_shape(2)
    def emulate_readings(self, x, signal, density_normalization=1.0):
        return torch_radon_cuda.emulate_sensor_readings(x, self.noise_generator, signal, density_normalization)

    @normalize_shape(2)
    def emulate_readings_new(self, x, signal, normal_std, k, bins):
        return torch_radon_cuda.emulate_readings_new(x, self.noise_generator, signal, normal_std, k, bins)

    @normalize_shape(2)
    def readings_lookup(self, sensor_readings, lookup_table):
        return torch_radon_cuda.readings_lookup(sensor_readings, lookup_table)

    def set_seed(self, seed=-1):
        if seed < 0:
            seed = np.random.get_state()[1][0]

        self.noise_generator.set_seed(seed)

    def __del__(self):
        self.noise_generator.free()

class RadonFanbeam(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/fanbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Fanbeam geometry.

    :param resolution: The resolution of the input images.
    :param angles: Array containing the list of measuring angles. Can be a Numpy array or a PyTorch tensor.
    :param source_distance: Distance between the source of rays and the center of the image.
    :param det_distance: Distance between the detector plane and the center of the image.
        By default it is =  :attr:`source_distance`.
    :param det_count: Number of rays that will be projected. By default it is = :attr:`resolution`.
    :param det_spacing: Distance between two contiguous rays.
    :param clip_to_circle: If True both forward and backward projection will be restricted to pixels inside the circle
        (highlighted in cyan).

    .. note::
        Currently only support resolutions which are multiples of 16.
    """

    def __init__(self, resolution: int, angles, source_distance: float, det_distance: float = -1, det_count: int = -1,
                 det_spacing: float = -1, clip_to_circle=False):

        if det_count <= 0:
            det_count = resolution

        if det_distance < 0:
            det_distance = source_distance
            det_spacing = 2.0
        if det_spacing < 0:
            det_spacing = (source_distance + det_distance) / source_distance

        rays_cfg = RaysCfg(resolution, resolution, det_count, det_spacing, len(angles), clip_to_circle,
                           source_distance, det_distance)

        super().__init__(angles, rays_cfg)

        self.source_distance = source_distance
        self.det_distance = det_distance
        self.det_count = det_count
        self.det_spacing = det_spacing