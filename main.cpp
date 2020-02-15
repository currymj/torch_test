#include <iostream>
#include <torch/torch.h>
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest.h"
#include "net.h"

int main(int argc, char** argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int res = context.run(); // run
    if (context.shouldExit()) {
        return res;
    }

    auto net = Net();

    auto dataset = torch::data::datasets::MNIST("./data/MNIST/raw");
    auto dataloader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()));

    torch::optim::SGD optimizer(net.parameters(), 0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        for (auto& batch : *dataloader) {
            optimizer.zero_grad();
            torch::Tensor predictions = net(batch.data);
            torch::Tensor loss = torch::nll_loss(predictions, batch.target);

            // does this work the same as python torch.autograd.grad?
            torch::Tensor grads = torch::autograd::grad({loss}, {predictions}, {}, /*retain_graph=*/true)[0];

            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << loss.item<float>() << std::endl;
            }
        }
    }
}

