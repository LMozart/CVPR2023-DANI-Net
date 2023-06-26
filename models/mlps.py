import torch

# Implementation borrowed from: https://github.com/krrish94/nerf-pytorch
class DepthMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        encode_fn=None,          # Encoder   
        num_encoding_fn_input=10,# Encode Dimension
        include_input_input=2,    # denote images coordinates (u, v)
    ):
        super(DepthMLP, self).__init__()
        self.dim_uv = include_input_input * (1 + 2 * num_encoding_fn_input)
        self.skip_connect_every = skip_connect_every + 1
        self.encode_fn = encode_fn

        # Branch 1
        self.layers_depth = torch.nn.ModuleList()
        self.layers_depth.append(torch.nn.Linear(self.dim_uv, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_depth.append(torch.nn.Linear(self.dim_uv + hidden_size, hidden_size))
            else:
                self.layers_depth.append(torch.nn.Linear(hidden_size, hidden_size))

        # Branch Output
        self.fc_depth = torch.nn.Linear(hidden_size, 1)

        # Activation Function
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        # Inputs.
        input = self.encode_fn(input)

        # Compute Features.
        xyz = input[..., :self.dim_uv]
        x = xyz
        for i in range(len(self.layers_depth)):
            if i == self.skip_connect_every:
                x = self.layers_depth[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_depth[i](x)
            x = self.relu(x)
        
        # Predict Depth.
        depth = self.fc_depth(x)
        return depth


class MaterialMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        encode_fn=None,            # Encoder
        num_encoding_fn_input=10,  # Encode Dimension
        include_input_input=2,     # Images Coordinates (u, v)
        output_ch=1
    ):
        super(MaterialMLP, self).__init__()
        self.dim_uv = include_input_input * (1 + 2 * num_encoding_fn_input)
        self.skip_connect_every = skip_connect_every + 1
        self.encode_fn = encode_fn

        # Branch 1
        self.layers_mat = torch.nn.ModuleList()
        self.layers_mat.append(torch.nn.Linear(self.dim_uv, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_mat.append(torch.nn.Linear(self.dim_uv + hidden_size, hidden_size))
            else:
                self.layers_mat.append(torch.nn.Linear(hidden_size, hidden_size))                            
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        
        # Branch 2
        self.layers_coeff = torch.nn.ModuleList()
        self.layers_coeff.append(torch.nn.Linear(hidden_size, hidden_size // 2))
        for i in range(3):
            self.layers_coeff.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        
        # Branch Output
        self.fc_spec_coeff = torch.nn.Linear(hidden_size // 2, output_ch)
        self.fc_diff = torch.nn.Linear(hidden_size // 2, 3)

        # Activation Function
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        # Inputs.
        input = self.encode_fn(input)
        uv = input[..., :self.dim_uv]
        x  = uv

        # Compute Features.
        for i in range(len(self.layers_mat)):
            if i == self.skip_connect_every:
                x = self.layers_mat[i](torch.cat((uv, x), -1))
            else:
                x = self.layers_mat[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        
        # Predict Coefficients.
        x = self.layers_coeff[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_coeff)):
            x = self.layers_coeff[i](x)
            x = self.relu(x)
            
        diff = torch.abs(self.fc_diff(x))
        spec_coeff = torch.abs(self.fc_spec_coeff(x))
        
        # Output Recording.
        coeff_dict = {"diff": diff, 
                      "spec_coeff": spec_coeff}
        return coeff_dict