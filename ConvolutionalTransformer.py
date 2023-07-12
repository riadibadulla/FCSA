import math

import torch.nn as nn
import torch
import torch
from torch.nn.functional import unfold, fold, pad
from torch.nn.functional import conv2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size : int
        Size of the patch (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The emmbedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = int((img_size // patch_size) ** 2)
        # self.n_patches = (1+(self.img_size-self.patch_size)//(self.patch_size//2))**2
    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # Get the batch size and number of channels
        batch_size = x.shape[0]

        # Reshape patches to the format (batch_size, num_patches, channels, height, width)
        x = x.reshape(batch_size, -1, self.patch_size, self.patch_size)
        return x


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0., n_patches=197):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.number_of_patches = n_patches

        self.q = nn.Conv2d(in_channels=self.number_of_patches, out_channels=self.number_of_patches, kernel_size=11, stride=1, padding="same", bias=False, groups=self.number_of_patches)
        self.v = nn.Conv2d(in_channels=self.number_of_patches, out_channels=self.number_of_patches, kernel_size=11,
                           stride=1, padding="same", bias=False, groups=self.number_of_patches)
        self.k = nn.Conv2d(in_channels=self.number_of_patches, out_channels=self.number_of_patches, kernel_size=11,
                           stride=1, padding="same", bias=False, groups=self.number_of_patches)
        #weight matrics
        self.q.weight.data = torch.ones((self.number_of_patches, 1, 11, 11))
        self.k.weight.data = torch.ones((self.number_of_patches, 1, 11, 11))
        self.v.weight.data = torch.ones((self.number_of_patches, 1, 11, 11))
        self.maxpool = nn.MaxPool2d(2)
        #kaiming initialisation
        torch.nn.init.kaiming_uniform_(self.q.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.v.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.k.weight, a=math.sqrt(5))

        self.attn_drop = nn.Dropout(attn_p)
        self.softmax = nn.Softmax(dim=2)
        # self.proj = nn.Conv2d(in_channels=self.number_of_patches, out_channels=self.number_of_patches, kernel_size=7,
        #                    stride=1, padding="same", bias=True, groups=self.number_of_patches)

    def perform_convolution_no_sum(self,input_tensor,kernel):
        _, _, ker_height, ker_width = kernel.shape
        # if (ker_height%2==0):
        #     kernel = torch.nn.functional.pad(kernel, (0, 1, 0, 1))
        batch_size, in_channels, height, width = input_tensor.shape
        kernel = kernel.unsqueeze(1).repeat(1, in_channels, 1, 1, 1)
        _, out_channels, _, kernel_height, kernel_width = kernel.shape
        # Compute padding
        padding_height = kernel_height // 2
        padding_width = kernel_width // 2

        new_input_size = input_tensor.shape[-1]
        input_tensor = input_tensor.view(batch_size, in_channels, 1, new_input_size, new_input_size)
        input_tensor = input_tensor.expand(-1, -1, out_channels, -1, -1)

        # Use unfold to create the sliding windows
        input_tensor = input_tensor.unfold(3, kernel_height, 1).unfold(4, kernel_width, 1)
        input_tensor = input_tensor.contiguous().view(batch_size, in_channels, out_channels, -1, kernel_height,
                                            kernel_width)
        kernel = kernel.view(batch_size, out_channels, in_channels, 1, kernel_height, kernel_width)
        kernel = kernel.expand(-1, -1, -1, input_tensor.shape[3], -1, -1)

        input_tensor = torch.einsum('...ij,...ij->...', input_tensor, kernel)

        return input_tensor.unsqueeze(-1)/new_input_size

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim_x, dim_y = x.shape
        #TODO: Bring back the maxpool
        # q = self.maxpool(self.q(x))
        # k = self.maxpool(self.k(x))
        q = self.q(x)
        k = self.k(x)


        alpha_matrix = self.perform_convolution_no_sum(q,k)
        del q, k
        alpha_matrix = self.softmax(alpha_matrix)
        x = self.v(x) # replaced v with x
        #Get output matrices
        output_mat = torch.empty((n_samples, n_tokens, dim_x, dim_y)).to(device)
        for sample in range(n_samples):
            output_mat[sample,:,:,:] = conv2d(x[sample,:,:,:].unsqueeze(0), alpha_matrix[sample,:,:,:,:], padding="same")
        return output_mat


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, mlp_p=0.):
        super().__init__()

        # self.conv1 = torch.nn.ConvTranspose2d(in_features, in_features, kernel_size=2,stride=2,groups=in_features)
        self.conv1 = nn.Conv2d(in_features,in_features,kernel_size=11,padding="same",groups=in_features)
        self.act = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.batch_norm2 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=11, padding="same",groups=in_features)

        self.drop = nn.Dropout2d(mlp_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        # x = self.fc1(
        #         x
        # ) # (n_samples, n_patches + 1, hidden_features)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=2.0, qkv_bias=True, p=0., attn_p=0.,n_patches=197,current_depth=0,last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.current_depth = current_depth
        self.image_sizes_at_each_depth = [110,80,50,20]
        self.current_image_size = self.image_sizes_at_each_depth[current_depth]
        self.patch_embed = PatchEmbed(
            img_size=self.current_image_size,
            patch_size=10)
        self.n_patches = self.patch_embed.n_patches
        if current_depth == 0:
            self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.n_patches, dim,dim)
            )
        if self.last_layer:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim,dim))
            self.n_patches += 1
        self.norm1 = nn.LayerNorm([dim,dim], eps=1e-6)
        # self.norm1 = nn.BatchNorm2d(n_patches)
        self.attention_heads = nn.ModuleList(
            [Attention( dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p, n_patches=self.n_patches)
                for _ in range(n_heads)
            ]
        )
        self.norm2 = nn.LayerNorm([dim,dim], eps=1e-6)
        # self.norm2 = nn.BatchNorm2d(n_patches)
        self.mlp = MLP(
                in_features=self.n_patches,
                hidden_features=int(self.n_patches * mlp_ratio),
                out_features=self.n_patches,
                mlp_p=attn_p
        )
        if not self.last_layer:
            self.pool = nn.AdaptiveAvgPool2d(self.image_sizes_at_each_depth[current_depth+1])


    def forward(self, x):


        x = self.patch_embed(x)
        if self.current_depth == 0:
            x = x + self.pos_embed
        if self.last_layer:
            n_samples = x.shape[0]
            cls_token = self.cls_token.expand(n_samples, -1, -1,-1)
            x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        new_x = torch.zeros_like(x)
        new_x += x
        for attention_layer in self.attention_heads:
            new_x += attention_layer(self.norm1(x))
        del x
        # x = x + self.attn(self.norm1(new_x))
        new_x = new_x + self.mlp(self.norm2(new_x))
        if not self.last_layer:
            input_shape = list(new_x.shape)

            # Calculate new height and width
            new_height = new_width = int(math.sqrt(input_shape[1]) * input_shape[2])

            # Reshape
            new_x = new_x.view(input_shape[0], int(math.sqrt(input_shape[1])), input_shape[2], -1, input_shape[3]).permute(
                0, 1, 3, 2, 4)
            # Further reshape
            new_x = new_x.contiguous().view(input_shape[0], new_height, new_width)
            new_x = self.pool(new_x)
        return new_x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            n_classes=100,
            embed_dim=768,
            depth=5,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()
        #TODO: Think of custom embedding,rather than hard codding it
        self.learn_colours = nn.Conv2d(in_chans,1,kernel_size=9,padding="same")
        self.gelu = nn.GELU()

        embed_dim = patch_size
        # self.patch_embed = PatchEmbed(
        #         img_size=img_size,
        #         patch_size=patch_size,
        #         in_chans=in_chans,
        #         embed_dim=embed_dim,
        # )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim,embed_dim))
        # self.pos_embed = nn.Parameter(
        #         torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim,embed_dim)
        # )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                    current_depth = i,
                    last_layer = True if i == depth-1 else False
                )
                for i in range(depth)
            ]
        )

        # self.blocks[-1].last_layer = True

        self.norm = nn.LayerNorm([embed_dim,embed_dim], eps=1e-6)

        self.head = nn.Sequential(nn.Flatten(),
                                    nn.Linear(100,120),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(120,100))


    def forward(self, x):
        x = self.learn_colours(x)
        x = self.gelu(x)
        n_samples = x.shape[0]

        # cls_token = self.cls_token.expand(
        #         n_samples, -1, -1,-1
        # )  # (n_samples, 1, embed_dim)
        # x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        # x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        # x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x[:, 0].unsqueeze(1)  # just the CLS token
        x = self.head(x)
        return x

if __name__ == "__main__":
    at = Attention( 10, n_heads=1, qkv_bias=False, attn_p=0, proj_p=0, n_patches=11)
    input = torch.ones(1,11,10,10)
    b = at(input)
    # kernel = torch.ones(1,3,6,6)
    # out = at.perform_convolution_no_sum(input,kernel)
    # print(out.shape)
