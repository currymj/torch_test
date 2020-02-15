#include <iostream>
#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net():
    fc1(register_module("fc1", torch::nn::Linear(784, 64))),
    fc2(register_module("fc2", torch::nn::Linear(64, 64))),
    fc3(register_module("fc3", torch::nn::Linear(64, 10)))
    {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x.view({-1, 784})));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc2(x));
        x = torch::log_softmax(fc3(x), /*dim=*/1);
        return x;
    }

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

int main() {
    auto net = Net();

    auto dataset = torch::data::datasets::MNIST("./data/MNIST/raw");
    auto dataloader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()));

    torch::optim::SGD optimizer(net.parameters(), 0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        for (auto& batch : *dataloader) {
            optimizer.zero_grad();
            torch::Tensor predictions = net.forward(batch.data);
            torch::Tensor loss = torch::nll_loss(predictions, batch.target);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << loss.item<float>() << " ";
            }
        }
    }
    }
