import torch
import torch.nn.functional as F
from torch import nn
import collections
from itertools import repeat
from misc import misc
import torchvision

def padding_same(kernel_size = (3,3), stride = (1,1), dilation = (1,1)):
    def parse(x, n=2):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    def pad_along_dim(kernel_size, stride, dilation, dim=0):
        padding = (dilation[dim] * (kernel_size[dim] - 1) + 1 - stride[dim]) / 2
        return int(padding)

    kernel_size = parse(kernel_size)
    stride = parse(stride)
    dilation = parse(dilation)
    
    padding = ( pad_along_dim(kernel_size, stride, dilation, dim=0),
                pad_along_dim(kernel_size, stride, dilation, dim=1))

    return padding


class PolarBlock(torch.nn.Module):
    def __init__(self, norm_layer, in_depth, out_depth, kernel_size, padding, use_bias=True, do_pool=True, pool = (2,1), dilation=(1,2)) -> None:
        super().__init__()

        if not do_pool:
            dilation=(1,1)

        self.do_pool = do_pool

        self.padding1 = padding
        self.padding2 = padding_same(kernel_size=kernel_size, dilation=dilation)

        self.conv1 = nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, bias=use_bias)
        self.conv2 = nn.Conv2d(out_depth, out_depth, kernel_size=kernel_size, dilation=dilation, bias=use_bias)
        self.norm = norm_layer(out_depth)
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=pool)

    def mixed_padding(self, x:torch.Tensor, padding):
        """Circular padding on the width, zero on height"""
        # pad: (padding_left, padding_right, padding_top, padding_bottom)
        # padding: (H,W)
        # Circular left, right
        x = F.pad(x, (padding[1], padding[1], 0, 0), mode='circular')

        # Pad top, bottom
        x = F.pad(x, (0, 0, padding[0], padding[0]), mode='constant', value=0)

        return x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.mixed_padding(x, self.padding1)
        x = self.relu(self.conv1(x))
        x = self.mixed_padding(x, self.padding2)
        x = self.relu(self.norm(self.conv2(x)))

        if self.do_pool:
            x = self.pool(x)

        return x
    
class PolarBlockHead(torch.nn.Module):
    def __init__(self, in_depth, out_depth, kernel_size, norm_layer=None, use_bias=True, do_pool=True) -> None:
        super().__init__()

        if do_pool:
            dilation=(2,1)
        else:
            dilation=(1,1)

        self.do_pool = do_pool

        self.padding1 = padding_same(kernel_size=kernel_size, dilation=dilation)

        self.conv1 = nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, dilation=dilation, bias=use_bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_depth)
        else:
            self.norm = lambda x: x
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=(1,2))

    def mixed_padding(self, x:torch.Tensor, padding):
        """Circular padding on the width, zero on height"""
        # pad: (padding_left, padding_right, padding_top, padding_bottom)
        # padding: (H,W)
        # Circular left, right
        x = F.pad(x, (padding[1], padding[1], 0, 0), mode='circular')

        # Pad bottom
        x = F.pad(x, (0, 0, 0, padding[0]), mode='constant', value=0)

        # Pad TOP
        x = F.pad(x, (0, 0, padding[0], 0), mode='reflect')
        # AVOID ARTIFACTS IN THE MIDDLE REGION

        return x

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.mixed_padding(x, self.padding1)

        x = self.relu(self.norm(self.conv1(x)))

        if self.do_pool:
            x = self.pool(x)

        return x


class CalV5(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        self.writer = None

        self.kernel_size = kernel_size

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer
        self.cell_size = 2**n_layers_fe
        self.use_bias = use_bias

        cart_fe = []
        polar_fe = []

        in_depth = input_nc
        out_depth = nf

        padding = padding_same(kernel_size=kernel_size)

        for _ in range(n_layers_fe):
            cart_fe.append(
                self.cart_block(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True,
                )
            )
            polar_fe.append(
                PolarBlock(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True,
                )
            )
            in_depth = out_depth
            out_depth = out_depth * 2

        cart_fe.append(
            self.cart_block(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        polar_fe.append(
            PolarBlock(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        
        self.cart_fe = nn.Sequential(*cart_fe)
        self.polar_fe = nn.Sequential(*polar_fe)

        self.feature_depths = in_depth
        in_depth = self.feature_depths * 2 # Feature concatenation

    

        # CALIB HEAD OUT: 
        out_depth = calib_out_depth

        self.cal_head = self.build_calib_head(
            in_depth=in_depth,
            out_depth=out_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
        )

        # LOGVAR HEAD OUT: 
        out_depth = calib_out_depth

        self.logvar_head = self.build_calib_head(
            in_depth=in_depth,
            out_depth=out_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
        )

        self.c2p_grids = {} # Cache for c2p transformations
        self.p2c_grids = {} # Cache for p2c transformations


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        # Shared Encoder.
        fe_cart:torch.Tensor = self.cart_fe(x_cart)
        fe_polar:torch.Tensor = self.polar_fe(x_polar)

        rho = fe_polar.shape[2]
        if rho not in self.p2c_grids:
            self.p2c_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=x_polar.shape[-1],
                batch_size=x_polar.shape[0],
                inverse=True,
                v2 = True,
            ).to(device=x_polar.device)

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=x_polar.shape[-1],
                batch_size=x_polar.shape[0],
                v2 = True,
            ).to(device=x_polar.device)

        cropped_p2c_grid = self.p2c_grids[rho]
        cropped_c2p_grid = self.c2p_grids[rho]

        fe_polar_c = misc.cartesian_grid_sample(fe_polar, cropped_p2c_grid) # N x 1 x Hp x Wp
        fe_cart_p = misc.polar_grid_sample(fe_cart, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        # if self.writer is not None:
        #     from misc.misc import tensor_to_heatmap
        #     for i in range(10):
        #         self.writer.add_image(f"polar_map/{0}",  tensor_to_heatmap(fe_polar[0,i,...]), i)
        #         self.writer.add_image(f"p2c_map/{0}",  tensor_to_heatmap(fe_polar_c[0,i,...]), i)
        #         self.writer.add_image(f"cart_map/{0}",  tensor_to_heatmap(fe_cart[0,i,...]), i)
        #         self.writer.add_image(f"cart_img/{0}",  x_cart[0,...], i)
        #         self.writer.add_image(f"polar_img/{0}",  x_polar[0,...], i)

        fe_cat_c = torch.cat([fe_cart, fe_polar_c], dim=1)
        fe_cat_p = torch.cat([fe_polar, fe_cart_p], dim=1)
        
        out_logvar = self.logvar_head(fe_cat_p) # N x cell_size^2+1 x Hc/cell_size x Wc/cell_size

        out_cal = self.cal_head(fe_cat_p) # N x cell_size^2 x Hc/cell_size x Wc/cell_size
        return out_logvar, out_cal


    def cart_block(self, in_depth, out_depth, kernel_size, norm_layer, padding=0, use_bias=True, do_pool=True):
        block =[
            nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(out_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            norm_layer(out_depth),
            nn.ReLU(True),
            ]
        if do_pool:
            block.append(nn.MaxPool2d(kernel_size=(2,2)))
        return nn.Sequential(*block)
    

    def build_logvar_head(self, in_depth, out_depth, norm_layer, kernel_size, n_layers=3, use_bias=True):
        """ 
            Head input: N x 256 x 50(Hp/cell) x Wp of va_vecs
            ->
            Head output: N x 1 x Hp/cell x Wp/Hp
        """
        head_out = out_depth
        out_depth = in_depth // 2

        blocks = []
        for _ in range(n_layers):
            blocks.append(PolarBlockHead(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                kernel_size=kernel_size, use_bias=use_bias, do_pool=True,
            ))

            in_depth = out_depth
            out_depth = min( out_depth * 2, 256)

        blocks += [
                PolarBlockHead(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                    kernel_size=kernel_size, use_bias=use_bias, do_pool=False,
                )
            ]

        for _ in range(n_layers):
            out_depth = max(in_depth // 2, 1)
            blocks += [
                PolarBlockHead(
                    norm_layer=None, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, use_bias=use_bias, do_pool=False,
                )
            ]
            in_depth = out_depth

        blocks += [
            nn.Conv2d(out_depth, head_out, kernel_size=1, padding=0, bias=use_bias),
        ]
        
        return nn.Sequential(*blocks)
    


    def build_calib_head(self, in_depth, out_depth, norm_layer, kernel_size, n_layers=3, use_bias=True):
        """ 
            Head input: N x 256 x 50(Hp/cell) x Wp of va_vecs
            ->
            Head output: N x 1 x Hp/cell x Wp/Hp
        """
        head_out = out_depth
        out_depth = in_depth // 2

        blocks = []
        for _ in range(n_layers):
            blocks.append(PolarBlockHead(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                kernel_size=kernel_size, use_bias=use_bias, do_pool=True,
            ))

            in_depth = out_depth
            out_depth = min( out_depth * 2, 256)

        blocks += [
                PolarBlockHead(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                    kernel_size=kernel_size, use_bias=use_bias, do_pool=False,
                )
            ]

        for _ in range(n_layers):
            out_depth = max(in_depth // 2, 1)
            blocks += [
                PolarBlockHead(
                    norm_layer=None, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, use_bias=use_bias, do_pool=False,
                )
            ]
            in_depth = out_depth

        blocks += [
            nn.Conv2d(out_depth, head_out, kernel_size=1, padding=0, bias=use_bias),
        ]
        
        return nn.Sequential(*blocks)
    



class CalV6(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        self.writer = None

        self.kernel_size = kernel_size

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer
        self.cell_size = 2**n_layers_fe
        self.use_bias = use_bias

        cart_fe = []
        polar_fe = []

        in_depth = input_nc
        out_depth = nf

        padding = padding_same(kernel_size=kernel_size)

        for _ in range(n_layers_fe):
            cart_fe.append(
                self.cart_block(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True,
                )
            )
            polar_fe.append(
                PolarBlock(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True,
                )
            )
            in_depth = out_depth
            out_depth = out_depth * 2

        cart_fe.append(
            self.cart_block(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        polar_fe.append(
            PolarBlock(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        
        self.cart_fe = nn.Sequential(*cart_fe)
        self.polar_fe = nn.Sequential(*polar_fe)

        self.feature_depths = in_depth
        in_depth = self.feature_depths * 2 # Feature concatenation


        self.second_fe = self.build_second_fe(
            in_depth=in_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
            n_layers=2,
        )

        # CALIB HEAD OUT: 
        self.cal_head = self.build_calib_head(
            in_depth=256*25,
            out_depth=calib_out_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
            n_layers=n_layers_fc,
        )

        self.c2p_grids = {} # Cache for c2p transformations
        self.p2c_grids = {} # Cache for p2c transformations


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        # Shared Encoder.
        fe_cart:torch.Tensor = self.cart_fe(x_cart)
        fe_polar:torch.Tensor = self.polar_fe(x_polar)

        rho = fe_polar.shape[2]
        if rho not in self.p2c_grids:
            self.p2c_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=x_polar.shape[-1],
                batch_size=x_polar.shape[0],
                inverse=True,
                v2 = True,
            ).to(device=x_polar.device)

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=x_polar.shape[-1],
                batch_size=x_polar.shape[0],
                v2 = True,
            ).to(device=x_polar.device)

        cropped_p2c_grid = self.p2c_grids[rho]
        cropped_c2p_grid = self.c2p_grids[rho]

        fe_polar_c = misc.cartesian_grid_sample(fe_polar, cropped_p2c_grid) # N x 1 x Hp x Wp
        fe_cart_p = misc.polar_grid_sample(fe_cart, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        # if self.writer is not None:
        #     from misc.misc import tensor_to_heatmap
        #     for i in range(10):
        #         self.writer.add_image(f"polar_map/{0}",  tensor_to_heatmap(fe_polar[0,i,...]), i)
        #         self.writer.add_image(f"p2c_map/{0}",  tensor_to_heatmap(fe_polar_c[0,i,...]), i)
        #         self.writer.add_image(f"cart_map/{0}",  tensor_to_heatmap(fe_cart[0,i,...]), i)
        #         self.writer.add_image(f"cart_img/{0}",  x_cart[0,...], i)
        #         self.writer.add_image(f"polar_img/{0}",  x_polar[0,...], i)

        fe_cat_c = torch.cat([fe_cart, fe_polar_c], dim=1)
        fe_cat_p = torch.cat([fe_polar, fe_cart_p], dim=1)

        fe_vec2:torch.Tensor = self.second_fe(fe_cat_p)

        fe_vec2 = fe_vec2.mean(-1)
        fe_vec2 = fe_vec2.reshape(fe_vec2.shape[0], -1, 1)
        
        out_cal:torch.Tensor = self.cal_head(fe_vec2) # N x cell_size^2 x Hc/cell_size x Wc/cell_size

        out_cal = out_cal.squeeze(-1)
        return out_cal


    def cart_block(self, in_depth, out_depth, kernel_size, norm_layer, padding=0, use_bias=True, do_pool=True):
        block =[
            nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(out_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            norm_layer(out_depth),
            nn.ReLU(True),
            ]
        if do_pool:
            block.append(nn.MaxPool2d(kernel_size=(2,2)))
        return nn.Sequential(*block)
    

    def build_second_fe(self, in_depth, norm_layer, kernel_size, n_layers=3, use_bias=True):
        """ 
            Head input: N x 256 x 50(Hp/cell) x Wp of va_vecs
            ->
            Head output: N x 1 x Hp/cell x Wp/Hp
        """
        out_depth = in_depth // 2

        blocks = []
        for _ in range(n_layers + 1):
            blocks.append(PolarBlockHead(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                kernel_size=kernel_size, use_bias=use_bias, do_pool=True,
            ))

            in_depth = out_depth
            out_depth = min( out_depth * 2, 256)

        return nn.Sequential(*blocks)
       
        
    def build_calib_head(self, in_depth, out_depth, norm_layer, kernel_size, n_layers=3, use_bias=True):
        head = []
        ch_in = in_depth
        
        for l in range(n_layers - 1):
            ch_out = out_depth * (n_layers - l -1)
            head += [
                nn.Conv1d(ch_in, ch_out, kernel_size=1, bias=use_bias, padding=0),
                nn.ReLU(True)
            ]
            ch_in = ch_out

        head += [
                nn.Conv1d(ch_in, out_depth, kernel_size=1, bias=use_bias, padding=0),
            ]

        return nn.Sequential(*head)


    
class CalV7(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        self.writer = None

        self.kernel_size = kernel_size

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer
        self.cell_size = 2**n_layers_fe
        self.use_bias = use_bias

        cart_fe = []
        polar_fe = []

        in_depth = input_nc
        out_depth = nf

        padding = padding_same(kernel_size=kernel_size)

        for _ in range(n_layers_fe):
            cart_fe.append(
                self.cart_block(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True,
                )
            )
            polar_fe.append(
                PolarBlock(
                    norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=True, pool=(2,2), dilation=(1,1),
                )
            )
            in_depth = out_depth
            out_depth = out_depth * 2

        cart_fe.append(
            self.cart_block(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        polar_fe.append(
            PolarBlock(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=in_depth,
                kernel_size=kernel_size, padding=padding, use_bias=use_bias, do_pool=False,
            )
        )
        
        self.cart_fe = nn.Sequential(*cart_fe)
        self.polar_fe = nn.Sequential(*polar_fe)

        self.feature_depths = in_depth
        in_depth = self.feature_depths * 2 # Feature concatenation


        self.second_fe = self.build_second_fe(
            in_depth=in_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
            padding=padding,
            n_layers=1,
        )

        # CALIB HEAD OUT: 
        self.cal_head = self.build_calib_head(
            in_depth=256*6,
            out_depth=calib_out_depth,
            norm_layer=self.norm_layer,
            kernel_size=kernel_size,
            use_bias=use_bias,
            n_layers=n_layers_fc,
        )

        self.c2p_grids = {} # Cache for c2p transformations
        self.p2c_grids = {} # Cache for p2c transformations


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        # Shared Encoder.
        fe_cart:torch.Tensor = self.cart_fe(x_cart)
        fe_polar:torch.Tensor = self.polar_fe(x_polar)

        rho = fe_polar.shape[2]
        if rho not in self.p2c_grids:
            self.p2c_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=fe_polar.shape[-1],
                batch_size=fe_polar.shape[0],
                inverse=True,
                v2 = True,
            ).to(device=fe_polar.device)

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=rho*2,
                input_width=rho*2,
                polar_width=fe_polar.shape[-1],
                batch_size=fe_polar.shape[0],
                v2 = True,
            ).to(device=fe_polar.device)

        cropped_p2c_grid = self.p2c_grids[rho]
        cropped_c2p_grid = self.c2p_grids[rho]

        fe_polar_c = misc.cartesian_grid_sample(fe_polar, cropped_p2c_grid) # N x 1 x Hp x Wp
        fe_cart_p = misc.polar_grid_sample(fe_cart, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        # if self.writer is not None:
        #     from misc.misc import tensor_to_heatmap
        #     for i in range(10):
        #         self.writer.add_image(f"polar_map/{0}",  tensor_to_heatmap(fe_polar[0,i,...]), i)
        #         self.writer.add_image(f"p2c_map/{0}",  tensor_to_heatmap(fe_polar_c[0,i,...]), i)
        #         self.writer.add_image(f"cart_map/{0}",  tensor_to_heatmap(fe_cart[0,i,...]), i)
        #         self.writer.add_image(f"cart_img/{0}",  x_cart[0,...], i)
        #         self.writer.add_image(f"polar_img/{0}",  x_polar[0,...], i)

        fe_cat_c = torch.cat([fe_cart, fe_polar_c], dim=1)
        fe_cat_p = torch.cat([fe_polar, fe_cart_p], dim=1)

        fe_vec2:torch.Tensor = self.second_fe(fe_cat_p)

        fe_vec2 = fe_vec2.mean(-1)
        fe_vec2 = fe_vec2.reshape(fe_vec2.shape[0], -1, 1)
        
        out_cal:torch.Tensor = self.cal_head(fe_vec2) # N x cell_size^2 x Hc/cell_size x Wc/cell_size

        out_cal = out_cal.squeeze(-1)
        return out_cal


    def cart_block(self, in_depth, out_depth, kernel_size, norm_layer, padding=0, use_bias=True, do_pool=True):
        block =[
            nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(out_depth, out_depth, kernel_size=kernel_size, padding=padding, bias=use_bias),
            norm_layer(out_depth),
            nn.ReLU(True),
            ]
        if do_pool:
            block.append(nn.MaxPool2d(kernel_size=(2,2)))
        return nn.Sequential(*block)
    

    def build_second_fe(self, in_depth, norm_layer, padding, kernel_size, n_layers=3, use_bias=True):
        out_depth = in_depth

        blocks = []
        for _ in range(n_layers + 1):
            blocks.append(PolarBlock(
                norm_layer=norm_layer, in_depth=in_depth, out_depth=out_depth, padding=padding,
                kernel_size=kernel_size, use_bias=use_bias, do_pool=True, dilation=(1,1), pool=(2,2)
            ))

            in_depth = out_depth
            out_depth = min( out_depth * 2, 256)

        return nn.Sequential(*blocks)
       
        
    def build_calib_head(self, in_depth, out_depth, norm_layer, kernel_size, n_layers=3, use_bias=True):
        head = []
        ch_in = in_depth
        
        for l in range(n_layers - 1):
            ch_out = out_depth * (n_layers - l)
            head += [
                nn.Conv1d(ch_in, ch_out, kernel_size=1, bias=use_bias, padding=0),
                nn.ReLU(True)
            ]
            ch_in = ch_out

        head += [
                nn.Conv1d(ch_in, out_depth, kernel_size=1, bias=use_bias, padding=0),
            ]

        return nn.Sequential(*head)
    

class SwingV1(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        self.swing = torchvision.models.swin_transformer.swin_v2_t(num_classes=calib_out_depth)

        self.writer = None


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        
        out = self.swing(x_cart)
        return out
    
from functools import partial

class ConvNextV1(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        self.net = torchvision.models.convnext_tiny(num_classes=calib_out_depth)

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        
        out = self.net(x_cart)
        return out

class ConvNextV2(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        self.net = torchvision.models.convnext_tiny(num_classes=calib_out_depth)
        lastconv_output_channels = 768
        self.net.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels, calib_out_depth*2),
            nn.GELU(),
            nn.Linear(calib_out_depth*2, calib_out_depth)
        )

        for m in self.net.classifier.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        
        out = self.net(x_cart)
        return out
    
class ConvNextV3(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        self.net = torchvision.models.convnext_tiny(num_classes=calib_out_depth)
        lastconv_output_channels = 768
        self.net.classifier = nn.Sequential(
           # norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels, calib_out_depth*2),
            nn.GELU(),
            nn.Linear(calib_out_depth*2, calib_out_depth)
        )

        for m in self.net.classifier.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        
        out = self.net(x_cart)
        return out

class ConvNext_mini(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting = [
            torchvision.models.convnext.CNBlockConfig(96, 192, 1),
            torchvision.models.convnext.CNBlockConfig(192, 384, 1),
            torchvision.models.convnext.CNBlockConfig(384, 768, 3),
            torchvision.models.convnext.CNBlockConfig(768, None, 1),
        ]
        stochastic_depth_prob = 0.1
        self.net = torchvision.models.convnext.ConvNeXt(block_setting, stochastic_depth_prob,num_classes=calib_out_depth)
        lastconv_output_channels = 768
        self.net.classifier = nn.Sequential(
           # norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels, calib_out_depth*2),
            nn.GELU(),
            nn.Linear(calib_out_depth*2, calib_out_depth)
        )

        for m in self.net.classifier.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        
        out = self.net(x_cart)
        return out


class ConvNext_mini2(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, polar=False) -> None:
        super().__init__()

        self.polar = polar

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting = [
            torchvision.models.convnext.CNBlockConfig(96, 192, 3), # 1
            torchvision.models.convnext.CNBlockConfig(192, 384, 6), # 3
            torchvision.models.convnext.CNBlockConfig(384, None, 1), # 1
        ]
        stochastic_depth_prob = 0.1
        self.net = torchvision.models.convnext.ConvNeXt(block_setting, stochastic_depth_prob,num_classes=calib_out_depth)
        lastconv_output_channels = block_setting[-1].input_channels
        self.net.classifier = nn.Sequential(
           # norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels, calib_out_depth*2),
            nn.GELU(),
            nn.Linear(calib_out_depth*2, calib_out_depth)
        )

        for m in self.net.classifier.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        if self.polar:
            out = self.net(x_polar)
        else:
            out = self.net(x_cart)
        return out
    
class DummyModule(nn.Module):
    def forward(self, x):
        return x

class ConvNext_mini3(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, polar=False) -> None:
        super().__init__()

        self.polar = polar

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting = [
            torchvision.models.convnext.CNBlockConfig(58, 96, 3), # 1
            torchvision.models.convnext.CNBlockConfig(96, 192, 3), # 1
            torchvision.models.convnext.CNBlockConfig(192, 384, 6), # 3
            torchvision.models.convnext.CNBlockConfig(384, None, 1), # 1
        ]
        stochastic_depth_prob = 0.1
        self.net = torchvision.models.convnext.ConvNeXt(block_setting, stochastic_depth_prob,num_classes=calib_out_depth)

        self.net.avgpool = nn.AdaptiveAvgPool2d((None,1))

        lastconv_output_channels = block_setting[-1].input_channels
        self.net.classifier = nn.Sequential(
           # norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels * 6, calib_out_depth*2),
            nn.GELU(),
            nn.Linear(calib_out_depth*2, calib_out_depth)
        )

        for m in self.net.classifier.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        if self.polar:
            x = x_polar
        else:
            x = x_cart
            
        x_fe = self.net.features(x)
        x_avg = self.net.avgpool(x_fe)
        out = self.net.classifier(x_avg)
        
        return out
    

class ConvNext_mix(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, polar=False) -> None:
        super().__init__()

        self.polar = polar

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting = [
            torchvision.models.convnext.CNBlockConfig(96, 192, 2), # 1
            torchvision.models.convnext.CNBlockConfig(192, 384, 4), # 3
            torchvision.models.convnext.CNBlockConfig(384, None, 1), # 1
        ]
        stochastic_depth_prob = 0.1
        
        self.net_cart = torchvision.models.convnext.ConvNeXt(block_setting, stochastic_depth_prob)
        self.net_cart.classifier = None
        self.net_cart = self.net_cart.features
        
        self.net_polar = torchvision.models.convnext.ConvNeXt(block_setting, stochastic_depth_prob)
        self.net_polar.classifier = None
        self.net_polar = self.net_polar.features
        
        lastconv_output_channels = block_setting[-1].input_channels * 2
        self.head = nn.Sequential(
           # norm_layer(lastconv_output_channels), # TODO togliere
            nn.Flatten(1), 
            nn.Linear(lastconv_output_channels, calib_out_depth*40),
            nn.GELU(),
            nn.Linear(calib_out_depth*40, calib_out_depth)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.c2p_grids = {}

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor,) -> torch.Tensor:
        cart_feat = self.net_cart(x_cart)
        polar_feat = self.net_polar(x_polar)       
        
        rho = polar_feat.shape[2]

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=cart_feat.shape[-2],
                input_width=cart_feat.shape[-1],
                polar_width=polar_feat.shape[-1],
                batch_size=polar_feat.shape[0],
                v2 = True,
            ).to(device=polar_feat.device)

        cropped_c2p_grid = self.c2p_grids[rho]

        fe_cart_p = misc.polar_grid_sample(cart_feat, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc
        

        # if self.writer is not None:
        #     from misc.misc import tensor_to_heatmap
        #     for i in range(10):
        #         self.writer.add_image(f"polar_map/{0}",  tensor_to_heatmap(fe_polar[0,i,...]), i)
        #         self.writer.add_image(f"p2c_map/{0}",  tensor_to_heatmap(fe_polar_c[0,i,...]), i)
        #         self.writer.add_image(f"cart_map/{0}",  tensor_to_heatmap(fe_cart[0,i,...]), i)
        #         self.writer.add_image(f"cart_img/{0}",  x_cart[0,...], i)
        #         self.writer.add_image(f"polar_img/{0}",  x_polar[0,...], i)

        fe_cat_p = torch.cat([polar_feat, fe_cart_p], dim=1)
        
        fe_cat_p = self.avgpool(fe_cat_p)

        out = self.head(fe_cat_p)
        return out
    
    
class HeadBlock(nn.Module):
    def __init__(self, input_shape, intermediate_neurons, old_out=0, out_neurons=1) -> None:
        super().__init__()
        
        neurons = [input_shape] + intermediate_neurons + [out_neurons]
        
        self.layers = [nn.Flatten(1)]
        
        old_position = len(neurons)-2
        self.insertion_idx = None
        
        for i in range(1,len(neurons)):
            if i == old_position:
                self.insertion_idx = len(self.layers)
                self.layers += [nn.Linear(neurons[i-1] + old_out, neurons[i])]
            else:
                self.layers += [nn.Linear(neurons[i-1], neurons[i])]            
            
            if i < len(neurons)-1:
                self.layers += [nn.GELU()]
                
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x, old=None):
        
        for i, layer in enumerate(self.layers):
            if i == self.insertion_idx and old is not None:
                x = torch.cat([x,old],-1)
                x = layer(x)
            else:
                x = layer(x)
            
        return x
            
import models.my_convnext as my_convnext
    
    

    
class ConvNext_mix_feat(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, stop_gradient=False, fully_local=True) -> None:
        super().__init__()

        self.stop_gradient = stop_gradient
        self.fully_local = fully_local

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting_cart = [
            my_convnext.My_CNBlockConfig(96, 192, 2), # 1
            my_convnext.My_CNBlockConfig(192, 384, 4, dilation=(2,2), stride=1, padding=1), # 3
            my_convnext.My_CNBlockConfig(384, None, 1), # 1
        ]
        # block_setting_polar = [
        #     my_convnext.My_CNBlockConfig(96, 192, 2), # 1
        #     my_convnext.My_CNBlockConfig(192, 384, 4, dilation=(2,1), stride=(1,2), padding=(1,0)), # 3
        #     my_convnext.My_CNBlockConfig(384, None, 1), # 1
        # ]
        block_setting_polar = [
            my_convnext.My_CNBlockConfig(96, 192, 2, polar=True), # 1
            my_convnext.My_CNBlockConfig(192, 384, 4, polar=True, dilation=(2,1), stride=(1,2), padding=(1,0)), # 3
            my_convnext.My_CNBlockConfig(384, None, 1, polar=True), # 1
        ]
        stochastic_depth_prob = 0.1
        
        self.net_cart = my_convnext.ConvNeXt_FE(block_setting_cart, stochastic_depth_prob)
        
        self.net_polar = my_convnext.ConvNeXt_FE(block_setting_polar, stochastic_depth_prob)
        
        lastconv_output_channels = block_setting_polar[-1].input_channels * 2
        # self.head = nn.Sequential(
        #    # norm_layer(lastconv_output_channels), # TODO togliere
        #     nn.Flatten(1), 
        #     nn.Linear(lastconv_output_channels, calib_out_depth*2),
        #     nn.GELU(),
        #     nn.Linear(calib_out_depth*2, calib_out_depth)
        # )
        
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))

        self.features_height = 25
        intermediate_neurons = [40]
        
        self.head = []
        for i in range(self.features_height):
            old_out = 1 if i>0 and not self.fully_local else 0 
            self.head.append(
                HeadBlock(lastconv_output_channels, intermediate_neurons, old_out=old_out, out_neurons=1)
            )
        self.head = nn.ModuleList(self.head)
        

        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.c2p_grids = {}

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor, gt_va:torch.Tensor = None) -> torch.Tensor:
        cart_feat = self.net_cart(x_cart)
        polar_feat = self.net_polar(x_polar)       
        
        rho = polar_feat.shape[2]

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=cart_feat.shape[-2],
                input_width=cart_feat.shape[-1],
                polar_width=polar_feat.shape[-1],
                batch_size=polar_feat.shape[0],
                v2 = True,
            ).to(device=polar_feat.device)

        cropped_c2p_grid = self.c2p_grids[rho]

        fe_cart_p = misc.polar_grid_sample(cart_feat, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        fe_cat_p = torch.cat([polar_feat, fe_cart_p], dim=1)
        
        fe_cat_p = self.avgpool(fe_cat_p)
        
        out:torch.Tensor = None
        
        outs = []
        for i in range(self.features_height):
            if self.fully_local:
                out = self.head[i](fe_cat_p[:,:,i,0])
            elif gt_va is not None and i > 0:
                out = self.head[i](fe_cat_p[:,:,i,0], gt_va[:,[i]])
            elif not self.stop_gradient or out is None:
                out = self.head[i](fe_cat_p[:,:,i,0], out)
            else:
                detached_out = out.detach()
                out = self.head[i](fe_cat_p[:,:,i,0], detached_out)
            outs.append(out)
        

        out = torch.cat(outs, -1)
        # out = out % (2*torch.pi)
        # out = torch.minimum(out, 2*torch.pi - out)
        out = out.clamp(0,torch.pi)
        return out
    
    
    
class ConvNext_mix_feat_small(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, stop_gradient=False, fully_local=True) -> None:
        super().__init__()

        self.stop_gradient = stop_gradient
        self.fully_local = fully_local

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting_cart = [
            my_convnext.My_CNBlockConfig(96, 192, 1), # 1
            my_convnext.My_CNBlockConfig(192, 384, 3, dilation=(2,2), stride=1, padding=1), # 3
            my_convnext.My_CNBlockConfig(384, None, 1), # 1
        ]
        block_setting_polar = [
            my_convnext.My_CNBlockConfig(96, 192, 1, polar=True), # 1
            my_convnext.My_CNBlockConfig(192, 384, 3, polar=True, dilation=(2,1), stride=(1,2), padding=(1,0)), # 3
            my_convnext.My_CNBlockConfig(384, None, 1, polar=True), # 1
        ]
        stochastic_depth_prob = 0.1
        
        self.net_cart = my_convnext.ConvNeXt_FE(block_setting_cart, stochastic_depth_prob)
        
        self.net_polar = my_convnext.ConvNeXt_FE(block_setting_polar, stochastic_depth_prob)
        
        lastconv_output_channels = block_setting_polar[-1].input_channels * 2
        
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))

        self.features_height = 25
        intermediate_neurons = [40]
        
        self.head = []
        for i in range(self.features_height):
            old_out = 1 if i>0 and not self.fully_local else 0 
            self.head.append(
                HeadBlock(lastconv_output_channels, intermediate_neurons, old_out=old_out, out_neurons=1)
            )
        self.head = nn.ModuleList(self.head)
        

        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.c2p_grids = {}

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor, gt_va:torch.Tensor = None) -> torch.Tensor:
        cart_feat = self.net_cart(x_cart)
        polar_feat = self.net_polar(x_polar)       
        
        rho = polar_feat.shape[2]

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=cart_feat.shape[-2],
                input_width=cart_feat.shape[-1],
                polar_width=polar_feat.shape[-1],
                batch_size=polar_feat.shape[0],
                v2 = True,
            ).to(device=polar_feat.device)

        cropped_c2p_grid = self.c2p_grids[rho]

        fe_cart_p = misc.polar_grid_sample(cart_feat, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        fe_cat_p = torch.cat([polar_feat, fe_cart_p], dim=1)
        
        fe_cat_p = self.avgpool(fe_cat_p)
        
        out:torch.Tensor = None
        
        outs = []
        for i in range(self.features_height):
            if self.fully_local:
                out = self.head[i](fe_cat_p[:,:,i,0])
            elif gt_va is not None and i > 0:
                out = self.head[i](fe_cat_p[:,:,i,0], gt_va[:,[i]])
            elif not self.stop_gradient or out is None:
                out = self.head[i](fe_cat_p[:,:,i,0], out)
            else:
                detached_out = out.detach()
                out = self.head[i](fe_cat_p[:,:,i,0], detached_out)
            outs.append(out)
        

        out = torch.cat(outs, -1)
        return out

class ConvNext_polar_feat(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, polar=False, stop_gradient=False, fully_local=True) -> None:
        super().__init__()

        self.polar = polar
        self.stop_gradient = stop_gradient
        self.fully_local = fully_local

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        
        if self.polar:
            block_setting = [
                my_convnext.My_CNBlockConfig(96, 192, 4), # 1
                my_convnext.My_CNBlockConfig(192, 384, 8, dilation=(2,1), stride=(1,2), padding=(1,0)), # 3
                my_convnext.My_CNBlockConfig(384, None, 2), # 1
            ]
        else:
            block_setting = [
                my_convnext.My_CNBlockConfig(96, 192, 2), # 1
                my_convnext.My_CNBlockConfig(192, 384, 4, dilation=(2,2), stride=1, padding=1), # 3
                my_convnext.My_CNBlockConfig(384, None, 1), # 1
            ]
        stochastic_depth_prob = 0.1
        
        self.net_fe = my_convnext.ConvNeXt_FE(block_setting, stochastic_depth_prob)
        self.net_fe.features.append(nn.Conv2d(384, 384*2, kernel_size=1, padding=0))
        
        lastconv_output_channels = block_setting[-1].input_channels * 2
        # self.head = nn.Sequential(
        #    # norm_layer(lastconv_output_channels), # TODO togliere
        #     nn.Flatten(1), 
        #     nn.Linear(lastconv_output_channels, calib_out_depth*2),
        #     nn.GELU(),
        #     nn.Linear(calib_out_depth*2, calib_out_depth)
        # )
        
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))

        self.features_height = 25
        intermediate_neurons = [40]
        
        self.head = []
        for i in range(self.features_height):
            old_out = 1 if i>0 and not self.fully_local else 0 
            self.head.append(
                HeadBlock(lastconv_output_channels, intermediate_neurons, old_out=old_out, out_neurons=1)
            )
        self.head = nn.ModuleList(self.head)
        

        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.c2p_grids = {}

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor, gt_va:torch.Tensor = None) -> torch.Tensor:
        if self.polar:
            feats = self.net_fe(x_polar)
            
        else:
            raise NotImplementedError()
            # feats = self.net_fe(x_cart)
            # rho = polar_feat.shape[2]
            # if rho not in self.c2p_grids:
            #     self.c2p_grids[rho] = misc.build_c2p_grid(
            #         input_height=feats.shape[-2],
            #         input_width=feats.shape[-1],
            #         polar_width=polar_feat.shape[-1],
            #         batch_size=polar_feat.shape[0],
            #         v2 = True,
            #     ).to(device=polar_feat.device)
            # cropped_c2p_grid = self.c2p_grids[rho]

            # fe_cart_p = misc.polar_grid_sample(feats, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc
 
        fe_cat_p = self.avgpool(feats)
        
        out:torch.Tensor = None
        
        outs = []
        for i in range(self.features_height):
            if self.fully_local:
                out = self.head[i](fe_cat_p[:,:,i,0])
            elif gt_va is not None and i > 0:
                out = self.head[i](fe_cat_p[:,:,i,0], gt_va[:,[i]])
            elif not self.stop_gradient or out is None:
                out = self.head[i](fe_cat_p[:,:,i,0], out)
            else:
                detached_out = out.detach()
                out = self.head[i](fe_cat_p[:,:,i,0], detached_out)
            outs.append(out)
    
        out = torch.cat(outs, -1)
        return out
    
    
class ConvNext_mix_feat_deep(nn.Module):

    def __init__(self, n_layers_fe=3, n_layers_fc=3, use_bias=True, nf=32, 
        input_nc=1, kernel_size = 5, norm_layer=None, calib_out_depth = 1, stop_gradient=False, fully_local=True) -> None:
        super().__init__()

        self.stop_gradient = stop_gradient
        self.fully_local = fully_local

        if norm_layer is None:
            norm_layer = partial(torchvision.models.convnext.LayerNorm2d, eps=1e-6)

        self.writer = None
        block_setting_cart = [
            my_convnext.My_CNBlockConfig(96, 192, 2), # 1
            my_convnext.My_CNBlockConfig(192, 384, 4), # 3
            my_convnext.My_CNBlockConfig(384, None, 1), # 1
        ]
        block_setting_polar = [
            my_convnext.My_CNBlockConfig(96, 192, 2, polar=True), # 1
            my_convnext.My_CNBlockConfig(192, 384, 4, polar=True), # 3
            my_convnext.My_CNBlockConfig(384, None, 1, polar=True), # 1
        ]
        stochastic_depth_prob = 0.1
        
        self.net_cart = my_convnext.ConvNeXt_FE(block_setting_cart, stochastic_depth_prob, stem=4)
        
        self.net_polar = my_convnext.ConvNeXt_FE(block_setting_polar, stochastic_depth_prob, stem=4)
        
        lastconv_output_channels = block_setting_polar[-1].input_channels * 2
        
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))

        self.features_height = 25
        intermediate_neurons = [40]
        
        self.head = []
        for i in range(self.features_height):
            old_out = 1 if i>0 and not self.fully_local else 0 
            self.head.append(
                HeadBlock(lastconv_output_channels, intermediate_neurons, old_out=old_out, out_neurons=1)
            )
        self.head = nn.ModuleList(self.head)
        

        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.c2p_grids = {}

    def forward(self, x_cart:torch.Tensor, x_polar:torch.Tensor, gt_va:torch.Tensor = None) -> torch.Tensor:
        cart_feat = self.net_cart(x_cart)
        polar_feat = self.net_polar(x_polar)       
        
        rho = polar_feat.shape[2]

        if rho not in self.c2p_grids:
            self.c2p_grids[rho] = misc.build_c2p_grid(
                input_height=cart_feat.shape[-2],
                input_width=cart_feat.shape[-1],
                polar_width=polar_feat.shape[-1],
                batch_size=polar_feat.shape[0],
                v2 = True,
            ).to(device=polar_feat.device)

        cropped_c2p_grid = self.c2p_grids[rho]

        fe_cart_p = misc.polar_grid_sample(cart_feat, cropped_c2p_grid, border_remove=0) # N x 1 x Hc x Wc

        fe_cat_p = torch.cat([polar_feat, fe_cart_p], dim=1)
        
        fe_cat_p = self.avgpool(fe_cat_p)
        
        out:torch.Tensor = None
        
        outs = []
        for i in range(self.features_height):
            if self.fully_local:
                out = self.head[i](fe_cat_p[:,:,i,0])
            elif gt_va is not None and i > 0:
                out = self.head[i](fe_cat_p[:,:,i,0], gt_va[:,[i]])
            elif not self.stop_gradient or out is None:
                out = self.head[i](fe_cat_p[:,:,i,0], out)
            else:
                detached_out = out.detach()
                out = self.head[i](fe_cat_p[:,:,i,0], detached_out)
            outs.append(out)
        

        out = torch.cat(outs, -1)
        return out