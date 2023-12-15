import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d, deform_conv2d


class DeformConv(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=None):
        super(DeformConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        channels_ = groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)


class ResidualBlock(nn.Module):
    def __init__(self, planes, ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.conv2 = DeformConv(planes, planes, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(planes, affine=True)
        self.norm2 = nn.InstanceNorm2d(planes, affine=True)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x_sc = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + x_sc


class QueryExpansionModule(nn.Module):
    def __init__(self, hidden_num=768, n_block=8, input_size=128, outout_size=192, patch_size=16):
        super(QueryExpansionModule, self).__init__()

        self.hidden_num = hidden_num #768
        
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_num) for _ in range(n_block)])
        self.noise_mlp = nn.Sequential(nn.Linear(hidden_num // 8, hidden_num // 4), nn.LayerNorm(hidden_num // 4),
                                        nn.ReLU(),
                                        nn.Linear(hidden_num // 4, hidden_num // 2), nn.LayerNorm(hidden_num // 2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_num // 2, hidden_num))

        self.norm = nn.LayerNorm(hidden_num)
        self.embed = nn.Linear(hidden_num, hidden_num)
        
        # self.outer_query_index 144 return true false 

    def get_index(self,h,w,h_pad,w_pad):
        mask = torch.ones(size=[h+2*h_pad, w+2*w_pad]).long()
        mask[h_pad:-h_pad, w_pad:-w_pad] = 0
        mask = mask.flatten(0)
        return mask == 0, mask == 1

    def forward(self, src_query,pic_h_w,pad_h_w,mask):
        h,w = pic_h_w[:]
        h_pad,w_pad = pad_h_w[:]
        inner_query_index, outer_query_index = self.get_index(h,w,h_pad,w_pad)

        src_query = src_query.permute(1, 0, 2)
        b, n, c = src_query.size()

        ori_src_query = src_query
       
        noise = torch.randn(size=(b, (h+2*h_pad)*(w+2*w_pad), c // 8), dtype=torch.float32).to(src_query.device)
        # 1 144 96
        initial_query = self.noise_mlp(noise)

        initial_query[:, inner_query_index] = src_query

        x = initial_query.permute(0, 2, 1).reshape(b, c, (h+2*h_pad), (w+2*w_pad)).contiguous()
        # 1 786 12 12
        for layer in self.res_blocks:
            x = layer(x)
        # CNN 大小没有变
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x[:, inner_query_index, :] = ori_src_query
        x = self.embed(x)
        x = x[:,outer_query_index,:].permute(1, 0, 2)
        return x

if __name__ == '__main__':
    m1 = QueryExpansionModule()
    x1 = torch.randn([1, 64, 768])
    y1 = m1(x1)
    print(y1.size())
