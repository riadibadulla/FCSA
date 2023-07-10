import torch

input_tensor = torch.ones(2,3,10,10)
kernel = torch.ones(2,3,3,3,3)


batch_size, in_channels, height, width = input_tensor.shape
_, out_channels, _, kernel_height, kernel_width = kernel.shape

# Compute padding
padding_height = kernel_height // 2
padding_width = kernel_width // 2

# Pad the input tensor
input_padded = torch.nn.functional.pad(input_tensor, (padding_width, padding_width, padding_height, padding_height))
new_input_size = input_padded.shape[-1]
input_padded = input_padded.repeat(1,out_channels,1,1).reshape(batch_size,in_channels,out_channels,new_input_size,new_input_size)
# Use unfold to create the sliding windows
windows = input_padded.unfold(3, kernel_height, 1).unfold(4, kernel_width, 1)
windows = windows.contiguous().view(batch_size, in_channels, out_channels,width * height, kernel_height, kernel_width)
kernel_reshaped = kernel.repeat(1,1,height*width,1,1).reshape(batch_size,out_channels,in_channels,height*width,kernel_height,kernel_width)
# kernle_reshaped = kernel.repeat(1,width * height,1,1).reshape(batch_size,in_channels,width * height,kernel_height,kernel_width)
output = (windows * kernel_reshaped).sum(dim=-1).sum(dim=-1).reshape(batch_size,in_channels,out_channels,height,width)
print(output)
