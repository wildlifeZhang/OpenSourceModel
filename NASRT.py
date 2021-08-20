import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable

# optimal architecture found by NASMS in paper submitted to Neuralcomputing

def compute_processed_size(data_size, kernel_size, process_mode='conv', stride=1, padding=0, dilation=1, output_ratio=None):
    if process_mode == 'conv' or process_mode == 'max':
        return math.floor((data_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    elif process_mode == 'average':
        return math.floor((data_size + 2 * padding - kernel_size) / stride + 1)
    elif process_mode == 'fractional':
        return math.floor(data_size * output_ratio)
    elif process_mode == 'lp':
        return math.floor((data_size-kernel_size)/stride + 1)
    elif process_mode == 'unpool':
        return math.floor((data_size - 1) * stride - 2 * padding + kernel_size)


def compute_image_size(image_height, image_width, kernel_size, process_mode='conv', stride=[1,1], padding=[0,0], dilation =[1,1], output_ratio=[1,1]):
    image_height = compute_processed_size(image_height, kernel_size[0], 
                                          process_mode=process_mode, 
                                          stride=stride[0], 
                                          padding=padding[0], 
                                          dilation=dilation[0],
                                          output_ratio=output_ratio[0])
    image_width = compute_processed_size(image_width, kernel_size[1], 
                                          process_mode=process_mode, 
                                          stride=stride[1], 
                                          padding=padding[1], 
                                          dilation=dilation[1],
                                          output_ratio=output_ratio[1])
    return image_height, image_width


def downsample(input_data, output_height, output_width, GPU):    
    """
    Transform the dimensions of input_data, i.e., [N, C, H, W] to
    [N, C, H*, W*] where H*=output_height and W*=output_width,
    notice, H* can be larger OR smaller than the original dimension
    H, so is W*.

    Parameters
    ----------
      (1) input_data : torch.Tensor
        A tensor of dimensions [N, C, H, W] where N, C, H, W respectively
        denote sample Number, Channel number of a single sample, Height
        of a sample and Width of a sample
      (2) output_height : number
        Number denotes the desired sample height
      (2) output_width : number
        Number denotes the desired sample width

    Returns
    -------
      res : torch.Tensor
        The dimension-changed tensor

    Example
    -------
    >>> t=tensor([[[[  1.,   2.,   3.,   4.],
                    [  5.,   6.,   7.,   8.],
                    [  9.,  10.,  11.,  12.]]]])
    >>> downsample(t, 2*t.shape[2], t.shape[3]//2)
    >>> tensor([[[[  1.0000,   4.0000],
                  [  2.6000,   5.6000],
                  [  4.2000,   7.2000],
                  [  5.8000,   8.8000],
                  [  7.4000,  10.4000],
                  [  9.0000,  12.0000]]]])
    """
    if input_data.shape[2]==output_height and input_data.shape[3]==output_width:
        return input_data
    if output_height==1: height_indices = torch.tensor([-1], dtype=torch.float32).unsqueeze(1)
    else: height_indices = torch.linspace(-1, 1, output_height).view(-1, 1).repeat(1, output_width)
    if output_width==1: width_indices = torch.tensor([-1], dtype=torch.float32).unsqueeze(1)
    else: width_indices = torch.linspace(-1, 1, output_width).view(1, -1).repeat(output_height, 1)   
    grid = torch.cat((width_indices.unsqueeze(2), height_indices.unsqueeze(2)), 2)
    grid.unsqueeze_(0)
    if input_data.is_cuda:
        return F.grid_sample(input_data, grid.repeat(input_data.shape[0],1,1,1).cuda(device=GPU), align_corners=True) 
    else:
        return F.grid_sample(input_data, grid.repeat(input_data.shape[0],1,1,1), align_corners=True)


class ConvOperation(nn.Module):
    def __init__(self, image_height, image_width, input_channel, config_channel,
                 kernel_size, stride, padding, dilation, is_depthwise_separable=False):        
        super(ConvOperation, self).__init__()
        if not isinstance(image_height, int): raise Exception('image_height should be an integer')
        elif image_height > 0: self.input_height = image_height
        else: raise Exception('image_height should be positive')
        if not isinstance(image_width, int): raise Exception('image_width should be an integer')
        elif image_width > 0: self.input_width = image_width
        else: raise Exception('image_width should be positive')
        if not isinstance(input_channel, int): raise Exception('input_channel should be an integer')
        elif input_channel > 0: self.input_channel = input_channel
        else: raise Exception('input_channel should be positive')    
        if not isinstance(config_channel, int): raise Exception('config_channel should be an integer')
        elif config_channel <= 0: raise Exception('config_channel should be positive')
        if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, Iterable):            
            raise Exception('kernel_size should be an iterable object containing two integers')    
            if len(kernel_size)==2 and all(isinstance(x, int) for x in kernel_size) and all(x>0 for x in kernel_size):
                self.kernel_size = kernel_size
            else: raise Exception('kernel_size should contain two positive integers')
        if isinstance(stride, int): self.stride = (stride, stride)
        elif not isinstance(stride, Iterable):            
            raise Exception('stride should be an iterable object containing two integers')    
            if len(stride)==2 and all(isinstance(x, int) for x in stride) and all(x>0 for x in stride):
                self.stride = stride
            else: raise Exception('stride should contain two positive integers')
        if isinstance(padding, int): self.padding = (padding, padding)
        elif not isinstance(padding, Iterable):            
            raise Exception('padding should be an iterable object containing two integers')    
            if len(padding)==2 and all(isinstance(x, int) for x in padding) and all(x>0 for x in padding):
                self.padding = padding
            else: raise Exception('padding should contain two positive integers')
        if isinstance(dilation, int): self.dilation = (dilation, dilation)
        elif not isinstance(dilation, Iterable):            
            raise Exception('dilation should be an iterable object containing two integers')    
            if len(dilation)==2 and all(isinstance(x, int) for x in dilation) and all(x>0 for x in dilation):
                self.dilation = dilation
            else: raise Exception('dilation should contain two positive integers')            
        if not isinstance(is_depthwise_separable, bool): raise Exception('is_depthwise_separable should be bool')
        # when current operation is the destination of a skip connection, the stacked channels 
        # are adjusted to the original (config_channel) channel number sepecified in channel setting 
        if isinstance(config_channel, int) and is_depthwise_separable:
            # parameters of conv_layer employ the default values which won't change image size
            if input_channel != config_channel:
                self.adjust_layer1 = nn.Conv2d(input_channel, config_channel, kernel_size=(1, 1), stride=(1,1), padding=(0,0), dilation=(1,1))
                self.adjust_layer2 = nn.BatchNorm2d(config_channel)
                self.adjust_layer3 = nn.ReLU()
        self.is_depthwise_separable  = is_depthwise_separable        
        image_height, image_width = compute_image_size(image_height, image_width, 
                                                        kernel_size=self.kernel_size, 
                                                        stride=self.stride, 
                                                        padding=self.padding, 
                                                        dilation=self.dilation)            
        if is_depthwise_separable:    
            self.depthwise_layer = nn.Conv2d(config_channel, config_channel, 
                                            kernel_size=self.kernel_size, 
                                            stride=self.stride, 
                                            padding=self.padding, 
                                            dilation=self.dilation, groups=config_channel)            
            self.separable_layer = nn.Conv2d(config_channel, config_channel, kernel_size=(1, 1))
            # parameters of separable layer guarentee employ the default values, hence the 
            # compute_image_size() is fed by using these defaults which won't change image size
            image_height, image_width = compute_image_size(image_height, image_width, 
                                                            kernel_size=(1, 1), 
                                                            stride=(1,1), 
                                                            padding=(0,0), 
                                                            dilation=(1,1))   
        else:
            self.conv_layer = nn.Conv2d(input_channel, config_channel, 
                                        kernel_size=self.kernel_size, 
                                        stride=self.stride, 
                                        padding=self.padding, 
                                        dilation=self.dilation)             
        self.batch_norm_layer = nn.BatchNorm2d(config_channel)    # sample size does not change
        self.relu_layer = nn.ReLU()                               # sample size does not change
        self.output_height = image_height
        self.output_width = image_width
        self.output_channel = config_channel
            
    def forward(self, x):
        if 'adjust_layer1' in self._modules:
            out = self.adjust_layer1(x)
            out = self.adjust_layer2(out)
            out = self.adjust_layer3(out)
        else:
            out = x   
        if self.is_depthwise_separable:
            out = self.depthwise_layer(out)
            out = self.separable_layer(out)
        else:
            out = self.conv_layer(out)            
        out = self.batch_norm_layer(out)
        out = self.relu_layer(out)
        return out  


class UpsamplingOperation(nn.Module):
    def __init__(self, input_height, input_width, output_height, output_width, upsample_mode='nearest'):        
        super(UpsamplingOperation, self).__init__()
        if not isinstance(input_height, int): raise Exception('image_height should be an integer')
        elif not input_height > 0: raise Exception('image_height should be positive')        
        if not isinstance(input_width, int): raise Exception('input_width should be an integer')
        elif not input_width > 0: raise Exception('input_width should be positive')
        if not isinstance(output_height, int): raise Exception('output_height should be an integer')
        elif not output_height > 0: raise Exception('output_height should be positive')
        if not isinstance(output_width, int): raise Exception('output_width should be an integer')
        elif not output_width > 0: raise Exception('output_width should be positive')        
        if input_height > output_height or input_width > output_width: 
            raise Exception('target size should exceed source size')
        if not isinstance(upsample_mode, str): raise Exception('upsample_mode should be a string')
        elif upsample_mode!='nearest' and upsample_mode!='linear' and upsample_mode!='bilinear' and upsample_mode!='trilinear':
            raise Exception('Unsupported upsample_mode, supported modes: nearest, linear, bilinear and trilinear')
        
        height_scale = output_height // input_height
        width_scale = output_width // input_width
        
        if height_scale > 1 and height_scale == width_scale:
            self.upsample_layer = nn.Upsample(scale_factor=height_scale, mode=upsample_mode)
            height_diff = output_height - math.floor(height_scale * input_height)
            width_diff = output_width - math.floor(width_scale * input_width)
        else:
            height_diff = output_height - input_height
            width_diff = output_width - input_width

        # (paddingLeft, paddingRight, paddingTop, paddingBottom)
        if height_diff != 0 or width_diff != 0:
            half_height_diff = height_diff // 2
            half_width_diff = width_diff //2                        
            self.padding_layer = nn.ZeroPad2d((half_width_diff, width_diff - half_width_diff, 
                                               half_height_diff, height_diff - half_height_diff))      
        
    def forward(self, x):
        if 'upsample_layer' in self._modules:
            out = self.upsample_layer(x)
        else:
            out = x        
        if 'padding_layer' in self._modules:
            out = self.padding_layer(out)        
        return out


class PoolingOperation(nn.Module):    
    def __init__(self, image_height, image_width, input_channel, config_channel,
                 pooling_mode, kernel_size=(2, 2), **options):        
        super(PoolingOperation, self).__init__()
        # for all pooling modes
        if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, Iterable):            
            raise Exception('kernel_size should be an iterable object containing two integers')    
            if len(kernel_size)==2 and all(isinstance(x, int) for x in kernel_size) and all(x>0 for x in kernel_size):
                self.kernel_size = kernel_size
            else: raise Exception('kernel_size should contain two positive integers')
        self.input_height = image_height
        self.input_width = image_width        
        self.pooling_indices = []       
        if not isinstance(pooling_mode, str):
            raise Exception('input pooling_mode is not of type str')
        elif pooling_mode=='max' or pooling_mode=='average' or pooling_mode=='fractional' or pooling_mode=='lp':
            self.pooling_mode = pooling_mode
        else:
            raise Exception('supported pooling_mode, supported modes are: max, average, fractional, lp')
        if pooling_mode=='max' or pooling_mode=='average' or pooling_mode=='lp':
            if not (isinstance(options.get("stride"), Iterable) or isinstance(options.get("stride"), int)):
                raise Exception('max/average/lp pooling requires input stride')
            stride = options.get("stride")
            if isinstance(stride, int): self.stride = (stride, stride)
            elif not isinstance(stride, Iterable):            
                raise Exception('stride should be an iterable object containing two integers')    
                if len(stride)==2 and all(isinstance(x, int) for x in stride) and all(x>0 for x in stride):
                    self.stride = stride
                else: raise Exception('stride should contain two positive integers')
        if pooling_mode=='max' or pooling_mode=='average':
            if not (isinstance(options.get("padding"), Iterable) or isinstance(options.get("padding"), int)):
                raise Exception('max/average pooling requires input padding')
            padding = options.get("padding")
            if isinstance(padding, int): self.padding = (padding, padding)
            elif not isinstance(padding, Iterable):            
                raise Exception('padding should be an iterable object containing two integers')    
                if len(padding)==2 and all(isinstance(x, int) for x in padding) and all(x>0 for x in padding):
                    self.padding = padding
                else: raise Exception('padding should contain two positive integers')
        if pooling_mode=='max':
            if not (isinstance(options.get("dilation"), Iterable) or isinstance(options.get("dilation"), int)):
                raise Exception('max pooling requires input dilation')
            dilation = options.get("dilation")
            if isinstance(dilation, int): self.dilation = (dilation, dilation)
            elif not isinstance(dilation, Iterable):            
                raise Exception('dilation should be an iterable object containing two integers')    
                if len(dilation)==2 and all(isinstance(x, int) for x in dilation) and all(x>0 for x in dilation):
                    self.dilation = dilation
                else: raise Exception('dilation should contain two positive integers')                  
        if pooling_mode=='max' or pooling_mode=='fractional':
            if isinstance(options.get("return_indices"), bool): 
                self.return_indices = options.get("return_indices")                
            else:
                raise Exception('max/fractional pooling requires input return_indices of type Boolean')     
        if pooling_mode=='max' or pooling_mode=='average' or pooling_mode=='lp':
            if isinstance(options.get("ceil_mode"), bool): 
                self.ceil_mode = options.get("ceil_mode")                
            else:
                raise Exception('max/average/lp pooling requires input ceil_mode of type Boolean')             
        if pooling_mode=='average':
            if isinstance(options.get("count_include_pad"), bool): 
                self.count_include_pad = options.get("count_include_pad")               
            else:
                raise Exception('average pooling requires input count_include_pad of type Boolean')    
        if pooling_mode=='fractional':
            if not isinstance(options.get("output_ratio"), Iterable):            
                raise Exception('fractional pooling requires input output_ratio')    
            elif all(isinstance(x, float) for x in options.get("output_ratio")) and all(0<=x and x<=1.0 for x in options.get("output_ratio")):
                self.output_ratio = options.get("output_ratio")
            else:
                raise Exception('output_ratio should be an iterable object containing floats within [0,1]')        
        if pooling_mode=='lp':
            if isinstance(options.get("norm_type"), float): 
                self.norm_type = options.get("norm_type")
            else:
                raise Exception('lp pooling requires input norm_type of type int')  
        # when current operation is the destination of a skip connection, the stacked channels 
        # are adjusted to the original (config_channel) channel number sepecified in channel setting 
        if isinstance(config_channel, int):
            # parameters of conv_layer employ the default values which won't change image size
            if input_channel != config_channel:
                self.adjust_layer1 = nn.Conv2d(input_channel, config_channel, kernel_size=(1, 1), stride=(1,1), padding=(0,0), dilation=(1,1))
                self.adjust_layer2 = nn.BatchNorm2d(config_channel)
                self.adjust_layer3 = nn.ReLU()
        else:
            config_channel = input_channel
        self.input_channel = config_channel
        self.output_channel = config_channel    
        if self.pooling_mode=="max":            
            self.pooling_layer = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation,
                                            return_indices=self.return_indices, ceil_mode=self.ceil_mode)            
            self.output_height, self.output_width=compute_image_size(image_height, image_width, kernel_size=self.kernel_size,                                                                        
                                                                    process_mode=self.pooling_mode, stride=self.stride, 
                                                                    padding=self.padding, dilation=self.dilation)
        elif self.pooling_mode=="average":            
            self.pooling_layer = nn.AvgPool2d(kernel_size=self.kernel_size, 
                                            stride=self.stride,
                                            padding=self.padding,                                            
                                            ceil_mode=self.ceil_mode,
                                            count_include_pad = self.count_include_pad)
            self.output_height, self.output_width=compute_image_size(image_height, image_width, kernel_size=self.kernel_size,                                                                        
                                                                    process_mode=self.pooling_mode, stride=self.stride, 
                                                                    padding=self.padding)
        elif self.pooling_mode=="fractional":
            self.pooling_layer = nn.FractionalMaxPool2d(kernel_size=self.kernel_size,                                            
                                            output_ratio=self.output_ratio,
                                            return_indices=self.return_indices)
            self.output_height, self.output_width=compute_image_size(image_height, image_width, kernel_size=self.kernel_size,                                                                        
                                                                    process_mode=self.pooling_mode, output_ratio=self.output_ratio)
        elif self.pooling_mode=="lp":
            self.pooling_layer = nn.LPPool2d(norm_type=self.norm_type,
                                            kernel_size=self.kernel_size, 
                                            stride=self.stride,
                                            ceil_mode=self.ceil_mode)
            self.output_height, self.output_width=compute_image_size(image_height, image_width, kernel_size=self.kernel_size,                                                                        
                                                                    stride=self.stride)        
                      
    def forward(self, x):
        if 'adjust_layer1' in self._modules:
            out = self.adjust_layer1(x)
            out = self.adjust_layer2(out)
            out = self.adjust_layer3(out)
        else:
            out = x        
        # unpooling layers in self.layers_cancel requires the linear indices returned by pooling layers
        # hence, the returned indices are saved as field member for future unpooling        
        if self.pooling_mode=="max" or self.pooling_mode=="fractional": 
            if self.return_indices:
                out, self.pooling_indices = self.pooling_layer(out)
            else:
                out = self.pooling_layer(out)
        elif self.pooling_mode=="average" or self.pooling_mode=="lp":            
            out = self.pooling_layer(out)
            # when pooling is average or lp, there's no indices returned by the
            # built-in pooling layers. The only mealns to generate unpooled data 
            # is nn.MaxUnpool2d which requires indices. Thus, it's generated manually.
            if self.pooling_mode=="average": padding = self.padding
            else: padding = (0,0)
            if self.return_indices:
                indices = generatePoolingIndices(self.input_height, self.input_width, 
                                                kernel_size=self.kernel_size,stride=self.stride,padding=padding)
                indices = torch.tensor(indices).reshape((1, 1, self.output_height, self.output_width))
                self.pooling_indices = indices.repeat(out.shape[0], out.shape[1], 1, 1)
        return out


class UnpoolingOperation(nn.Module):
    def __init__(self, input_height, input_width, output_height, output_width,
                 pooling_mode, kernel_size, **options):        
        super(UnpoolingOperation, self).__init__() 
        if not isinstance(pooling_mode, str):
            raise Exception('input pooling_mode is not of type str')
        elif pooling_mode=='max' or pooling_mode=='average' or pooling_mode=='fractional' or pooling_mode=='lp':
            self.pooling_mode = pooling_mode
        else:
            raise Exception('supported pooling_mode, supported modes are: max, average, fractional, lp')
        if pooling_mode=='max' or pooling_mode=='average' or pooling_mode=='lp':
            if not isinstance(options.get("stride"), Iterable):
                raise Exception('max/average/lp unpooling requires input stride')      
            elif all(isinstance(x, int) for x in options.get("stride")):
                self.stride = options.get("stride")
            else:
                raise Exception('stride should be an iterable object containing integers')           
        if pooling_mode=='max' or pooling_mode=='average':
            if not isinstance(options.get("padding"), Iterable):
                raise Exception('max/average unpooling requires input padding')            
            elif all(isinstance(x, int) for x in options.get("padding")):
                self.padding = options.get("padding") 
            else:
                raise Exception('padding should be an iterable object containing integers')
        if pooling_mode=='fractional':
            if not isinstance(options.get("output_ratio"), Iterable):            
                raise Exception('fractional pooling requires input output_ratio')    
            elif all(isinstance(x, float) for x in options.get("output_ratio")) and all(0<=x and x<=1.0 for x in options.get("output_ratio")):
                self.output_ratio = options.get("output_ratio")
                # when pooling mode is fractional, there's no stride and padding yielded by fractional
                # pooling, however, it does return indices for unpooling. if the unpooled size does not 
                # match the size of input, then unpooling will trigger exception even if output and indices 
                # are provided correctly. The unpooled size is solely determined by stride and padding. 
                # To yield unpooling size compitable with indices returned by fractional pooling,
                # stride can be initialized based on output_ratio, and padding is set to 0 by default,
                # then it is adjusted to meet the size compitability
                self.stride = (1 / self.output_ratio[0], 1 / self.output_ratio[1])
                self.padding = (0, 0)
                diff0, diff1 = 1, 1
                while diff0 != 0 or diff1 != 0:
                    unpool_size0, unpool_size1 = compute_image_size(input_height, input_width, kernel_size, process_mode='unpool', stride=self.stride, padding=self.padding)                    
                    diff0, diff1 = unpool_size0-output_height, unpool_size1 - output_width
                    if diff0 != 0 or diff1 != 0:
                        self.padding = (diff0/2, diff1/2)
            else:
                raise Exception('output_ratio should be an iterable object containing floats within [0,1]')
        if pooling_mode=='lp':               
            # lp pooling does require padding, however, unpooling does, and (0, 0) is found working
            self.padding = (0, 0) 
                
        self.kernel_size = kernel_size        
        self.unpool_layer = nn.MaxUnpool2d(kernel_size=kernel_size, stride=self.stride, padding=self.padding)
        self.unpool_height, self.unpool_width = compute_image_size(input_height, input_width, kernel_size, process_mode='unpool', stride=self.stride, padding=self.padding)
     
        height_diff = output_height - self.unpool_height
        width_diff = output_width - self.unpool_width
        # (paddingLeft, paddingRight, paddingTop, paddingBottom)
        if height_diff != 0 or width_diff != 0:
            half_height_diff = height_diff // 2
            half_width_diff = width_diff //2                        
            self.padding_layer = nn.ZeroPad2d((half_width_diff, width_diff - half_width_diff, 
                                               half_height_diff, height_diff - half_height_diff))      
    
    def forward(self, x, indices):
        out = self.unpool_layer(x, indices)
        if 'padding_layer' in self._modules:
            out = self.padding_layer(out)       
        return out


class IdentityLayer(nn.Module):
    def __init__(self, image_height, image_width, input_channel, config_channel=None):        
        super(IdentityLayer, self).__init__() 
        if not isinstance(image_height, int):
            raise Exception('image_height should be a positive integer')
        elif not isinstance(image_width, int):
            raise Exception('image_width should be a positive integer')
        elif not isinstance(input_channel, int):
            raise Exception('input_channel should be a positive integer')
        self.output_height = image_height
        self.output_width = image_width
        self.config_channel = config_channel
        self.output_channel = config_channel
        if isinstance(config_channel, int):
            # parameters of conv_layer employ the default values which won't change image size
            if input_channel != config_channel:
                self.adjust_layer1 = nn.Conv2d(input_channel, config_channel, kernel_size=(1, 1), stride=(1,1), padding=(0,0), dilation=(1,1))
                self.adjust_layer2 = nn.BatchNorm2d(config_channel)
                self.adjust_layer3 = nn.ReLU()
        else:
            config_channel = input_channel

    def forward(self, inputs):
        if 'adjust_layer1' in self._modules:
            out = self.adjust_layer1(inputs)
            out = self.adjust_layer2(out)
            out = self.adjust_layer3(out)
        else:
            out = inputs  
        return out


class Normal_Cell(nn.Module):
    def __init__(self, cell_ID, config_channel, GPU, 
                 pre_c_4_height, pre_c_4_width, pre_c_4_channel, 
                 pre_c_3_height, pre_c_3_width, pre_c_3_channel,
                 pre_c_1_height, pre_c_1_width, pre_c_1_channel):
        super(Normal_Cell, self).__init__()
        if not all(isinstance(x, int) for x in [cell_ID, pre_c_4_height, pre_c_4_width, pre_c_4_channel,
                                                pre_c_3_height, pre_c_3_width, pre_c_3_channel,
                                                pre_c_1_height, pre_c_1_width, pre_c_1_channel, config_channel]):
            raise Exception('all inputs except GPU should be int')
        self.GPU = GPU
        self.cell_ID = cell_ID
        self.L0_conv3_sep5 = []
        self.L0_conv5_conv5 = []
        self.L1_iden_sep5_sep5 = []
        self.L1_conv3_conv3 = []
        self.L2_sep5_iden_conv5 = []
        # bottom-left block in Fig. 9 of the paper
        # in block, maintain the output size as the smallest one yielded by ops
        self.L0_conv3_sep5.append(ConvOperation(pre_c_3_height, pre_c_3_width, pre_c_3_channel, config_channel, kernel_size=3, stride=1, padding=1, dilation=1, is_depthwise_separable=False))
        self.L0_conv3_sep5.append(ConvOperation(pre_c_3_height, pre_c_3_width, pre_c_3_channel, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=True))
        self.L0_conv3_sep5_H = min([op.output_height for op in self.L0_conv3_sep5])
        self.L0_conv3_sep5_W = min([op.output_width for op in self.L0_conv3_sep5])
        self.L0_conv3_sep5_C = min([op.output_channel for op in self.L0_conv3_sep5])
        # bottom-right block in Fig. 9 of the paper
        self.L0_conv5_conv5.append(ConvOperation(pre_c_1_height, pre_c_1_width, pre_c_1_channel, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=False))
        self.L0_conv5_conv5.append(ConvOperation(pre_c_4_height, pre_c_4_width, pre_c_4_channel, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=False))
        self.L0_conv5_conv5_H = min([op.output_height for op in self.L0_conv5_conv5])
        self.L0_conv5_conv5_W = min([op.output_width for op in self.L0_conv5_conv5])
        self.L0_conv5_conv5_C = min([op.output_channel for op in self.L0_conv5_conv5])
        # middle-left block in Fig. 9 of the paper
        self.L1_iden_sep5_sep5.append(IdentityLayer(self.L0_conv3_sep5_H, self.L0_conv3_sep5_W, self.L0_conv3_sep5_C, config_channel))
        self.L1_iden_sep5_sep5.append(ConvOperation(self.L0_conv3_sep5_H, self.L0_conv3_sep5_W, self.L0_conv3_sep5_C, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=True))
        self.L1_iden_sep5_sep5.append(ConvOperation(pre_c_3_height, pre_c_3_width, pre_c_3_channel, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=True))
        self.L1_iden_sep5_sep5_H = min([op.output_height for op in self.L1_iden_sep5_sep5])
        self.L1_iden_sep5_sep5_W = min([op.output_width for op in self.L1_iden_sep5_sep5])
        self.L1_iden_sep5_sep5_C = min([op.output_channel for op in self.L1_iden_sep5_sep5])
        # middle-right block in Fig. 9 of the paper
        self.L1_conv3_conv3.append(ConvOperation(self.L0_conv5_conv5_H, self.L0_conv5_conv5_W, self.L0_conv5_conv5_C, config_channel, kernel_size=3, stride=1, padding=1, dilation=1, is_depthwise_separable=False))
        self.L1_conv3_conv3.append(ConvOperation(pre_c_3_height, pre_c_3_width, pre_c_3_channel, config_channel, kernel_size=3, stride=1, padding=1, dilation=1, is_depthwise_separable=False))
        self.L1_conv3_conv3_H = min([op.output_height for op in self.L1_conv3_conv3])
        self.L1_conv3_conv3_W = min([op.output_width for op in self.L1_conv3_conv3])
        self.L1_conv3_conv3_C = min([op.output_channel for op in self.L1_conv3_conv3])
        # top block in Fig. 9 of the paper
        self.L2_sep5_iden_conv5.append(ConvOperation(self.L0_conv3_sep5_H, self.L0_conv3_sep5_W, self.L0_conv3_sep5_C, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=True))
        self.L2_sep5_iden_conv5.append(IdentityLayer(self.L1_iden_sep5_sep5_H, self.L1_iden_sep5_sep5_W, self.L1_iden_sep5_sep5_C, config_channel))
        self.L2_sep5_iden_conv5.append(ConvOperation(pre_c_1_height, pre_c_1_width, pre_c_1_channel, config_channel, kernel_size=5, stride=1, padding=2, dilation=1, is_depthwise_separable=False))
        self.L2_sep5_iden_conv5_H = min([op.output_height for op in self.L2_sep5_iden_conv5])
        self.L2_sep5_iden_conv5_W = min([op.output_width for op in self.L2_sep5_iden_conv5])
        self.L2_sep5_iden_conv5_C = min([op.output_channel for op in self.L2_sep5_iden_conv5])
        # find output H, W, C of current cell
        # in cell, maintain the output size as the largest one yielded by blocks
        self.output_height = max([self.L1_conv3_conv3_H, self.L2_sep5_iden_conv5_H])
        self.output_width = max([self.L1_conv3_conv3_W, self.L2_sep5_iden_conv5_W])
        self.output_channel = sum([self.L1_conv3_conv3_C, self.L2_sep5_iden_conv5_C])
        self.operations = nn.ModuleList(self.L0_conv3_sep5 + self.L0_conv5_conv5 + self.L1_iden_sep5_sep5 + self.L1_conv3_conv3 + self.L2_sep5_iden_conv5)
            
    def forward(self, pre_c_4_inputs, pre_c_3_inputs, pre_c_1_inputs):
        # bottom-left block in Fig. 9 of the paper
        L0_conv3_sep5_out = self.L0_conv3_sep5[0](pre_c_3_inputs) + self.L0_conv3_sep5[1](pre_c_3_inputs)
        # bottom-right block in Fig. 9 of the paper
        L0_conv5_conv5_outs = [self.L0_conv5_conv5[0](pre_c_1_inputs), 
                               self.L0_conv5_conv5[1](pre_c_4_inputs)]
        if pre_c_4_inputs.is_cuda:
            L0_conv5_conv5_out = torch.Tensor([0]).cuda(device=f"cuda:{pre_c_4_inputs.get_device()}")
            L1_iden_sep5_sep5_out = torch.Tensor([0]).cuda(device=f"cuda:{pre_c_4_inputs.get_device()}")
            L1_conv3_conv3_out = torch.Tensor([0]).cuda(device=f"cuda:{pre_c_4_inputs.get_device()}")
            L2_sep5_iden_conv5_out = torch.Tensor([0]).cuda(device=f"cuda:{pre_c_4_inputs.get_device()}")
            final_output = torch.Tensor([0]).cuda(device=f"cuda:{pre_c_4_inputs.get_device()}")
        else:
            L0_conv5_conv5_out = torch.Tensor([0])
            L1_iden_sep5_sep5_out = torch.Tensor([0])
            L1_conv3_conv3_out = torch.Tensor([0])
            L2_sep5_iden_conv5_out = torch.Tensor([0])
            final_output = torch.Tensor([0])
        for output in L0_conv5_conv5_outs:
            if output.shape[2] > self.L0_conv5_conv5_H or output.shape[3] > self.L0_conv5_conv5_W:
                output = downsample(output, self.L0_conv5_conv5_H, self.L0_conv5_conv5_W, self.GPU)
            L0_conv5_conv5_out = L0_conv5_conv5_out + output
        # middle-left block in Fig. 9 of the paper
        L1_iden_sep5_sep5_outs = [self.L1_iden_sep5_sep5[0](L0_conv3_sep5_out), 
                                  self.L1_iden_sep5_sep5[1](L0_conv3_sep5_out), 
                                  self.L1_iden_sep5_sep5[2](pre_c_3_inputs)]
        # all operations inside a block should be fed inputs of the same dimensions
        # inculding the channels, otherwise the element-wise addition won't work
        for output in L1_iden_sep5_sep5_outs:
            if output.shape[2] > self.L1_iden_sep5_sep5_H or output.shape[3] > self.L1_iden_sep5_sep5_W:
                output = downsample(output, self.L1_iden_sep5_sep5_H, self.L1_iden_sep5_sep5_W, self.GPU)
            L1_iden_sep5_sep5_out = L1_iden_sep5_sep5_out + output
        # middle-right block in Fig. 9 of the paper
        L1_conv3_conv3_outs = [self.L1_conv3_conv3[0](L0_conv5_conv5_out), 
                               self.L1_conv3_conv3[1](pre_c_3_inputs)]
        for output in L1_conv3_conv3_outs:
            if output.shape[2] > self.L1_conv3_conv3_H or output.shape[3] > self.L1_conv3_conv3_W:
                output = downsample(output, self.L1_conv3_conv3_H, self.L1_conv3_conv3_W, self.GPU)
            L1_conv3_conv3_out = L1_conv3_conv3_out + output
        # top block in Fig. 9 of the paper
        L2_sep5_iden_conv5_outs = [self.L2_sep5_iden_conv5[0](L0_conv3_sep5_out), 
                                   self.L2_sep5_iden_conv5[1](L1_iden_sep5_sep5_out), 
                                   self.L2_sep5_iden_conv5[2](pre_c_1_inputs)]
        for output in L2_sep5_iden_conv5_outs:
            if output.shape[2] > self.L2_sep5_iden_conv5_H or output.shape[3] > self.L2_sep5_iden_conv5_W:
                output = downsample(output, self.L2_sep5_iden_conv5_H, self.L2_sep5_iden_conv5_W, self.GPU)
            L2_sep5_iden_conv5_out = L2_sep5_iden_conv5_out + output
        # build final output
        # in cell, maintain the output size as the largest one yielded by blocks
        for output in [L1_conv3_conv3_out, L2_sep5_iden_conv5_out]:
            height, width = output.shape[2], output.shape[3]
            height_scale = self.output_height // height
            width_scale = self.output_width // width
            if height_scale > 1 and height_scale == width_scale:
                output = F.interpolate(output, scale_factor=height_scale, mode='nearest')   # interpolate                 
                height_diff = self.output_height - output.shape[2]
                width_diff = self.output_width - output.shape[3]
            else:
                height_diff = self.output_height - height
                width_diff = self.output_width - width
            # (paddingLeft, paddingRight, paddingTop, paddingBottom)
            if height_diff != 0 or width_diff != 0:
                half_height_diff = height_diff // 2
                half_width_diff = width_diff //2
                output = F.pad(output, (half_width_diff, width_diff-half_width_diff, half_height_diff, height_diff-half_height_diff), "constant", 0)
            if sum(final_output.shape) == 1:
                final_output = output
            else:
                final_output = torch.cat([final_output, output], 1)
        return final_output


class NASRT(nn.Module):
    # optimal architecture found by NASMS in paper submitted to Neuralcomputing
    def __init__(self, input_height, input_width, input_channel, class_num, config_channel=36, N=3, GPU='cuda:0'):
        super(NASMP, self).__init__()
        if not all(isinstance(x, int) for x in [input_height, input_width, input_channel, class_num]):
            raise Exception('all inputs except GPU should be int')
        self.operations = []
        self.operations.append(Normal_Cell(0, config_channel, GPU, 
                                            input_height, input_width, input_channel, 
                                            input_height, input_width, input_channel,
                                            input_height, input_width, input_channel))
        self.operations.append(Normal_Cell(1, config_channel, GPU,
                                            input_height, input_width, input_channel, 
                                            input_height, input_width, input_channel,
                                            self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel))
        self.operations.append(Normal_Cell(2, config_channel, GPU,
                                            input_height, input_width, input_channel, 
                                            input_height, input_width, input_channel,
                                            self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel,))
        self.operations.append(PoolingOperation(self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel, config_channel,
                                                'max', kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False))
        cell_ID = 3
        for cell_id in range(cell_ID, 3 + N, 1):
            self.operations.append(Normal_Cell(cell_id, config_channel, GPU,
                                                self.operations[-4].output_height, self.operations[-4].output_width, self.operations[-4].output_channel,
                                                self.operations[-3].output_height, self.operations[-3].output_width, self.operations[-3].output_channel,
                                                self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel))
            cell_ID = cell_ID + 1
        self.operations.append(PoolingOperation(self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel, config_channel,
                                                'max', kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False))
        for cell_id in range(cell_ID, cell_ID + N, 1):
            self.operations.append(Normal_Cell(cell_id, config_channel, GPU,
                                                self.operations[-4].output_height, self.operations[-4].output_width, self.operations[-4].output_channel,
                                                self.operations[-3].output_height, self.operations[-3].output_width, self.operations[-3].output_channel,
                                                self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel))
            cell_ID = cell_ID + 1
        self.operations.append(PoolingOperation(self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel, config_channel,
                                                'max', kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False))
        for cell_id in range(cell_ID, cell_ID + N, 1):
            self.operations.append(Normal_Cell(cell_id, config_channel, GPU,
                                                self.operations[-4].output_height, self.operations[-4].output_width, self.operations[-4].output_channel,
                                                self.operations[-3].output_height, self.operations[-3].output_width, self.operations[-3].output_channel,
                                                self.operations[-1].output_height, self.operations[-1].output_width, self.operations[-1].output_channel))
            cell_ID = cell_ID + 1
        output_height, output_width = self.operations[-1].output_height, self.operations[-1].output_width
        # global_average
        if self.operations[-1].output_channel != class_num:
            self.operations.append(nn.Conv2d(self.operations[-1].output_channel, class_num, kernel_size=(1, 1), stride=(1,1), padding=(0,0), dilation=(1,1)))
        self.operations.append(nn.AvgPool2d(kernel_size=(output_height, output_width), stride=None, padding=(0,0), ceil_mode=False, count_include_pad=True))
        self.operations = nn.ModuleList(self.operations)

    def forward(self, inputs):
        cell_outputs = []
        for op in self.operations:
            if isinstance(op, Normal_Cell):
                if op.cell_ID >= 3:
                    cell_outputs.append(op(cell_outputs[-4], cell_outputs[-3], cell_outputs[-1]))
                elif 0 < op.cell_ID and op.cell_ID < 3:
                    cell_outputs.append(op(inputs, inputs, cell_outputs[-1]))
                elif op.cell_ID == 0:
                    cell_outputs.append(op(inputs, inputs, inputs))
            else:
                cell_outputs.append(op(cell_outputs[-1]))
        out = cell_outputs[-1]
        out.squeeze_(2)
        out.squeeze_(2)
        return out
