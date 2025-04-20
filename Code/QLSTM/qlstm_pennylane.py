import torch
import torch.nn as nn
import pennylane as qml

class HybridActivationBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

class QLSTM(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                n_qubits=4,
                n_qlayers=1,
                batch_first=True,
                return_sequences=False,
                return_state=False,
                backend="default.qubit",
                use_hybrid_activation=False):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.use_hybrid_activation = use_hybrid_activation

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit(inputs, weights, wires):
            qml.templates.AngleEmbedding(inputs, wires=wires)
            qml.templates.BasicEntanglerLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        weight_shapes = {"weights": (n_qlayers, n_qubits)}

        self.qlayer_forget = qml.QNode(lambda inputs, weights: _circuit(inputs, weights, self.wires_forget), self.dev_forget, interface="torch")
        self.qlayer_input = qml.QNode(lambda inputs, weights: _circuit(inputs, weights, self.wires_input), self.dev_input, interface="torch")
        self.qlayer_update = qml.QNode(lambda inputs, weights: _circuit(inputs, weights, self.wires_update), self.dev_update, interface="torch")
        self.qlayer_output = qml.QNode(lambda inputs, weights: _circuit(inputs, weights, self.wires_output), self.dev_output, interface="torch")

        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.clayer_out = nn.Linear(n_qubits, self.hidden_size)

        if self.use_hybrid_activation:
            self.hybrid_forget = HybridActivationBlock(self.hidden_size, 2 * self.hidden_size)
            self.hybrid_input = HybridActivationBlock(self.hidden_size, 2 * self.hidden_size)
            self.hybrid_update = HybridActivationBlock(self.hidden_size, 2 * self.hidden_size)
            self.hybrid_output = HybridActivationBlock(self.hidden_size, 2 * self.hidden_size)
        else:
            self.hybrid_forget = nn.Identity()
            self.hybrid_input = nn.Identity()
            self.hybrid_update = nn.Identity()
            self.hybrid_output = nn.Identity()

    def forward(self, x, init_states=None):
        if self.batch_first:
            batch_size, seq_length, _ = x.size()
        else:
            seq_length, batch_size, _ = x.size()

        hidden_seq = []
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        if init_states:
            h_t, c_t = init_states[0][0], init_states[1][0]

        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.hybrid_forget(self.clayer_out(self.VQC['forget'](y_t))))
            i_t = torch.sigmoid(self.hybrid_input(self.clayer_out(self.VQC['input'](y_t))))
            g_t = torch.tanh(self.hybrid_update(self.clayer_out(self.VQC['update'](y_t))))
            o_t = torch.sigmoid(self.hybrid_output(self.clayer_out(self.VQC['output'](y_t))))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)
