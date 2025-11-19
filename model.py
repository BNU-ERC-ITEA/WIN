import numpy as np
import torch

from src.basic_module import *
logabs = lambda x: torch.log(torch.abs(x) + 1e-6)

class INN(nn.Module):
    # Definition of WIN-Naive
    def __init__(self, in_channels, out_channels, n_blocks, scale=1):
        super(INN, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        n_channels = in_channels
        operations = []
        n_channels *= 4
        # Preprocessing -- reducing spatial resolution for lower complexity
        operations.append(PixelShuffler(2, downscale=True))
        # Constructing InvNN for feature representation
        for _ in range(n_blocks):
            operations.append(Flow(n_channels, n_channels//2))
            # Postprocessing -- upscaling to the original spatial resolution
        operations.append(PixelShuffler(2, downscale=False))
        n_channels //= 4

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False):
        if not rev:  # Forward branch
            jacobian = torch.zeros(1).type_as(x)
            y = x
            y_plus = None
            for operation in self.operations:
                y = operation.forward(y, rev)
            y = y.narrow(1, 0, self.out_channels)
            if self.training:
                return y, y_plus, jacobian
            else:
                return y
        else:  # Reverse branch
            y = x
            y = torch.cat((y, torch.randn_like(y)), dim=1)

            for operation in reversed(self.operations):
                y = operation.forward(y, rev)
            return y

class WIN_Naive(nn.Module):
    # Definition of WIN-Naive
    def __init__(self, in_channels, out_channels, n_blocks, scale=2):
        super(WIN_Naive, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        n_channels = in_channels
        operations = []
        if self.scale == 1:  # Reversible image hiding or decolorization
            n_channels *= 4
            # Preprocessing -- reducing spatial resolution for lower complexity
            operations.append(PixelShuffler(2, downscale=True))
            # Constructing InvNN for feature representation
            for _ in range(n_blocks):
                operations.append(Flow(n_channels, n_channels//2))
            # Postprocessing -- upscaling to the original spatial resolution
            operations.append(PixelShuffler(2, downscale=False))
            n_channels //= 4
        else:  # Reversible image rescaling
            for _ in range(int(np.log2(scale))):
                # Preprocessing -- reducing spatial resolution for lower complexity and low-resolution target
                # operations.append(HaarDownsampling(n_channels))
                n_channels *= 4
                operations.append(PixelShuffler(2, downscale=True))
                # Constructing InvNN for feature representation
            for _ in range(n_blocks):
                operations.append(Flow(n_channels, n_channels//2))

        # A single GIC layer to generate the target output
        operations.append(WIC(n_channels, out_channels))
        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False):
        if not rev:  # Forward branch
            idx = 0
            jacobian = 0
            y = x
            y_plus = {}
            for operation in self.operations:
                if isinstance(operation, WIC):
                    if operation.in_channels > operation.out_channels:  # Ill-posed conversion
                        y = operation.forward(y, rev)
                        y_plus['idx{}_chn{}'.format(idx, operation.out_channels)] = y
                        idx += 1
                        y = y[:, :operation.out_channels]
                    else:   # Well-posed conversion
                        y = operation.forward(y, rev)
                    if self.training:
                        jacobian += operation.jacobian()  # Calculate the log-determinant of solving left inverse
                else:
                    y = operation.forward(y, rev)

            if self.training:
                return y, y_plus, jacobian
            else:
                return y
        else:  # Reverse branch
            y = x
            for operation in reversed(self.operations):
                y = operation.forward(y, rev)
            return y

class WIN(nn.Module):
    # Definition of WIN
    def __init__(self, in_channels, out_channels, n_channels, n_modules, n_blocks, splitting_ratio=0.75, scale=2):
        super(WIN, self).__init__()

        self.scale = scale
        SM_channels = int(n_channels*splitting_ratio)  # number of channels in short-term memory
        LM_channels = n_channels - SM_channels  # number of channels in long-term memory
        operations = []
        if scale == 1:  # Reversible image hiding or decolorization
            in_channels *= 4
            # Preprocessing -- reducing spatial resolution for lower complexity
            operations.append(PixelShuffler(2, downscale=True))
            # Feature Expansion \Phi_pre
            operations.append(WIC(in_channels, SM_channels))
            # Stacking WICM modules
            operations.append(WICM(n_channels, n_modules, n_blocks, splitting_ratio))
            in_channels = SM_channels + n_modules * LM_channels
            # Postprocessing -- upscaling to the original spatial resolution
            if in_channels % 4 != 0:
                raise InterruptedError
            operations.append(PixelShuffler(2, downscale=False))
            in_channels //= 4
            # Feature Fusion \Phi_fusion
            operations.append(WIC(in_channels, out_channels))
        else:  # Reversible image rescaling
            for _ in range(int(np.log2(scale))):
                # operations.append(HaarDownsampling(in_channels))
                in_channels *= 4
                if in_channels > n_channels:
                    n_channels = int(2 ** np.ceil(np.log2(in_channels)))
                    SM_channels = int(n_channels * splitting_ratio)  # number of channels in short-term memory
                    LM_channels = n_channels - SM_channels
                # Preprocessing -- reducing spatial resolution for lower complexity and low-resolution target
                operations.append(PixelShuffler(2, downscale=True))
                # Feature Expansion \Phi_pre
                operations.append(WIC(in_channels, SM_channels))
                # Stacking WICM modules
                operations.append(WICM(n_channels, n_modules, n_blocks, splitting_ratio))
                in_channels = SM_channels + n_modules * LM_channels
            # Feature Fusion \Phi_fusion
            operations.append(WIC(in_channels, out_channels))
        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False):
        if not rev:  # Forward branch
            idx = 0
            jacobian = 0
            y = x
            y_plus = {}
            for operation in self.operations:
                if isinstance(operation, PixelShuffler) or isinstance(operation, HaarDownsampling):
                    y = operation.forward(y, rev)
                    # if self.training:
                    #     jacobian += operation.jacobian()  # Calculate the log-determinant of solving left inverse
                elif isinstance(operation, WICM):
                    if self.training:
                        y, jac = operation.forward(y, rev)  # Output and log-determinant of WICM modules
                        jacobian += jac
                    else:
                        y = operation.forward(y, rev)
                elif isinstance(operation, WIC):
                    if operation.in_channels > operation.out_channels:  # Ill-posed conversion
                        y = operation.forward(y, rev)
                        y_plus['idx{}_chn{}'.format(idx, operation.out_channels)] = y
                        idx += 1
                        y = y[:, :operation.out_channels]
                    else:  # Well-posed conversion
                        y = operation.forward(y, rev)
                    if self.training:
                        jacobian += operation.jacobian()  # Calculate the log-determinant of solving left inverse

            if self.training:
                return y, y_plus, jacobian
            else:
                return y
        else:  # Reverse branch
            y = x
            for operation in reversed(self.operations):
                y = operation.forward(y, rev)
            return y