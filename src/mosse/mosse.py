from util.util import *
from fft2_wrapper.fft2 import *

import cv2
import numpy as np
import random
import time
import json

class MosseTracker:
    """
        Implementation of Visual Object Tracking using Adaptive Correlation Filters
        See : https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf 
    """
    def __init__(self) -> None:
        
        self.param = {
            "lr" : 0.125, # Learning rate
            "lambda" : 0.1, # Gaussian target sigma (radius)
            "sigma" : 2.0, # Regularization term
            "expansion_factor" : 1.5, # Expand target window
            "psr_thresh" : 10.0 # Psr threshold for updating filter 
        }

        self.training_param = {
            "num_rotation" : 12,
            "num_translation" : 12,
            "num_scale" : 12
        }

        self.time_analysis = { # Measure time in ms
            "init" : 0.0,
            "update" : 0.0,
            "train" : 0.0,
            "detect" : 0.0,
            "pre_process" : 0.0,
            "calc_psr" : 0.0
        }

        self.dbg_mode = True

        self.H_conj : np.ndarray # MOSSE filter
        self.A : np.ndarray
        self.B : np.ndarray
        self.response_map : np.ndarray

        self.window_2d : np.ndarray # hann, welch, cos ...
        self.g_static : np.ndarray # Static gaussian target which peak is centred
        self.G_static : np.ndarray # FFT of g_static
        self.expansion_bbox : list # (cx, cy, width, height)
        self.target_bbox : list # (cx, cy, width, height)
        self.psr : float

    def init(self, image : np.ndarray, target_bbox : list):
        """
            Arguments
            _______

            @param image : Grayscale image in uint8
            @param target_bbox : Initial bounding box for tracking
            Returns
            ______

            @return bbox : Target bounding box (x, y, w, h)
        """
        begin_time = time.time()

        assert(len(image.shape) == 2)
        assert(image.dtype == np.uint8)

        # Where argument target_bbox is formed (x, y, w, h)
        self.target_bbox = [target_bbox[0] + target_bbox[2] // 2,
                            target_bbox[1] + target_bbox[3] // 2,
                            target_bbox[2],
                            target_bbox[3]]
        
        self.expansion_bbox = [self.target_bbox[0],
                               self.target_bbox[1],
                               int(self.target_bbox[2] * self.param["expansion_factor"]),
                               int(self.target_bbox[3] * self.param["expansion_factor"])]
        self.psr = 0.0

        self.H_conj = np.zeros((self.expansion_bbox[3], self.expansion_bbox[2]))
        self.A = np.zeros((self.expansion_bbox[3], self.expansion_bbox[2]))
        self.B = np.zeros((self.expansion_bbox[3], self.expansion_bbox[2]))
        self.response_map = np.zeros((self.expansion_bbox[3], self.expansion_bbox[2]))

        # window_2d can be changed to the welch, hann etc.
        self.window_2d = gen_cos(self.expansion_bbox[3], self.expansion_bbox[2])
        self.g_static = self._gen_gauss_target(self.expansion_bbox[2], self.expansion_bbox[3], 0, 0)
        self.G_static = fft2_forward(self.g_static)

        f = subwindow(image, self.expansion_bbox[0], self.expansion_bbox[1], self.expansion_bbox[2], self.expansion_bbox[3])
        self._train(f, True)

        delta_time = time.time() - begin_time
        self.time_analysis["init"] = delta_time

    def update(self, image : np.ndarray):
        """
            Arguments
            _______

            @param image : Grayscale image in uint8

            Returns
            ______

            @return bbox : Target bounding box (x, y, w, h)
        """
        begin_time = time.time()

        assert(len(image.shape) == 2)
        assert(image.dtype == np.uint8)

        f = subwindow(image, self.expansion_bbox[0], self.expansion_bbox[1], self.expansion_bbox[2], self.expansion_bbox[3])
        self._detect(f)

        max_y, max_x, self.psr = self._calc_psr()
        
        if (self.psr > self.param["psr_thresh"]):
            dx = max_x - self.expansion_bbox[2] // 2
            dy = max_y - self.expansion_bbox[3] // 2

            self.expansion_bbox[0] += dx
            self.expansion_bbox[1] += dy
            self.target_bbox[0] += dx # (cx, cy, width, height)
            self.target_bbox[1] += dy # (cx, cy, width, height)

            f = subwindow(image, self.expansion_bbox[0], self.expansion_bbox[1], self.expansion_bbox[2], self.expansion_bbox[3])
            
            self._train(f, False)
        
        else:
            dx = max_x - self.expansion_bbox[2] // 2
            dy = max_y - self.expansion_bbox[3] // 2

            self.expansion_bbox[0] += dx
            self.expansion_bbox[1] += dy
            self.target_bbox[0] += dx # (cx, cy, width, height)
            self.target_bbox[1] += dy # (cx, cy, width, height)
        
        # bbox (x, y, w, h)
        bbox = [self.target_bbox[0] - self.target_bbox[2] // 2, self.target_bbox[1] - self.target_bbox[3] // 2, self.target_bbox[2], self.target_bbox[3]]

        delta_time = time.time() - begin_time
        self.time_analysis["update"] = delta_time

        if self.dbg_mode:
            print(json.dumps(self.time_analysis, indent=4, sort_keys=True))

        return bbox
    
    def _gen_gauss_target(self, width : int, height : int, dx : int, dy : int):
            """
                dx : x_shift from center
                dy : y_shift from center
            """
            # Create a grid of coordinates
            x = np.arange(width)
            y = np.arange(height)
            xx, yy = np.meshgrid(x, y)

            dx = dx + width // 2
            dy = dy + height // 2

            nom = (xx - dx) ** 2 + (yy - dy) ** 2
            denom = self.param["sigma"] * self.param["sigma"]

            exponent = -nom / denom

            return np.exp(exponent)
    
    def _preprocess(self, image : np.ndarray):
        """
            See : (Sec. 3.1) in https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf  
        """
        begin_time = time.time()

        log = log_transform(image)
        norm = normalize(log)
        windowed = apply_windowing(norm, self.window_2d)

        delta_time = time.time() - begin_time
        self.time_analysis["update"] = delta_time

        return windowed

    def _train(self, f : np.ndarray, initial : bool):
        """
            See : (Eq. 10-11-12) in https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf  
        """
        begin_time = time.time()

        if initial:

            for i in range(self.training_param["num_rotation"]):
                deg = random.randint(-20, 20)
                warped = warp_rotation(f.copy(), deg)
                processed = self._preprocess(warped)
                
                F = fft2_forward(processed)
                F_conj = conj(F)

                A_new = mul_spectrum(self.G_static, F_conj)
                B_new = mul_spectrum(F, F_conj)

                self.A = self.param["lr"] * A_new + (1 - self.param["lr"]) * self.A
                self.B = self.param["lr"] * B_new + (1 - self.param["lr"]) * self.B + self.param["lambda"]

            for i in range(self.training_param["num_translation"]):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)

                g = self._gen_gauss_target(self.expansion_bbox[2], self.expansion_bbox[3], dx, dy)
                G = fft2_forward(g)

                warped = warp_translation(f.copy(), dy, dx)
                processed = self._preprocess(warped)
                
                F = fft2_forward(processed)
                F_conj = conj(F)

                A_new = mul_spectrum(G, F_conj)
                B_new = mul_spectrum(F, F_conj)

                self.A = self.param["lr"] * A_new + (1 - self.param["lr"]) * self.A
                self.B = self.param["lr"] * B_new + (1 - self.param["lr"]) * self.B + self.param["lambda"]

            for i in range(self.training_param["num_scale"]):
                scale_factor = random.randint(-10, 10)
                scale_factor = 1.0 + scale_factor / 100

                warped = warp_scale(f.copy(), scale_factor)
                processed = self._preprocess(warped)
                
                F = fft2_forward(processed)
                F_conj = conj(F)

                A_new = mul_spectrum(self.G_static, F_conj)
                B_new = mul_spectrum(F, F_conj)

                self.A = self.param["lr"] * A_new + (1 - self.param["lr"]) * self.A
                self.B = self.param["lr"] * B_new + (1 - self.param["lr"]) * self.B + self.param["lambda"]
                 
        else:
            processed = self._preprocess(f.copy())
            F = fft2_forward(processed)
            F_conj = conj(F)

            A_new = mul_spectrum(self.G_static, F_conj)
            B_new = mul_spectrum(F, F_conj)

            self.A = self.param["lr"] * A_new + (1 - self.param["lr"]) * self.A
            self.B = self.param["lr"] * B_new + (1 - self.param["lr"]) * self.B + self.param["lambda"]   

        self.H_conj = div_spectrum(self.A, self.B)

        delta_time = time.time() - begin_time
        self.time_analysis["train"] = delta_time
    
    def _detect(self, f : np.ndarray):
        """
            See : (Eq. 1) in https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf  
        """
        begin_time = time.time()

        F = fft2_forward(f)
        G = mul_spectrum(F, self.H_conj)
        self.response_map = real(fft2_backward(G))

        delta_time = time.time() - begin_time
        self.time_analysis["detect"] = delta_time

        if self.dbg_mode:
            g = cv2.normalize(self.response_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
            cv2.imshow("g", g)
    
    def _calc_psr(self):
        """
            See : (Sec. 3.5) in https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
            TODO : Optimization need
        """
        begin_time = time.time()

        # Find the indices of the maximum value
        max_loc = np.argmax(self.response_map)

        # Convert the 1D index to 2D indices
        max_y, max_x = np.unravel_index(max_loc, self.response_map.shape)

        main_lobe_x = max_x
        main_lobe_y = max_y
        main_lobe_w = 11#int(np.sqrt(self.response_map.shape[0] * self.response_map.shape[1]) / 16)
        main_lobe_w = main_lobe_w + 1 if main_lobe_w % 2 == 0 else main_lobe_w
        main_lobe_h = 11#int(np.sqrt(self.response_map.shape[0] * self.response_map.shape[1]) / 16)
        main_lobe_h = main_lobe_h + 1 if main_lobe_h % 2 == 0 else main_lobe_h

        g_max = self.response_map[main_lobe_y, main_lobe_x]

        sl = []
        for i in range(self.response_map.shape[0]):
            for j in range(self.response_map.shape[1]):

                if j <= main_lobe_x - main_lobe_w // 2:
                    sl.append(self.response_map[i, j])
                
                if j > main_lobe_x + main_lobe_w // 2:
                    sl.append(self.response_map[i, j])
                
                if i <= main_lobe_y - main_lobe_h // 2:
                    sl.append(self.response_map[i, j])
                
                if i > main_lobe_y + main_lobe_h // 2:
                    sl.append(self.response_map[i, j])
        
        mean_sl = np.mean(sl)
        std_sl = np.std(sl)

        delta_time = time.time() - begin_time
        self.time_analysis["calc_psr"] = delta_time

        return (max_y, max_x, (g_max - mean_sl) / (std_sl + 1e-5))
        







